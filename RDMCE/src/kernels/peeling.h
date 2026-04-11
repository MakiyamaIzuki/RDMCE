#pragma once
#ifndef PEELING_H
#define PEELING_H

#include "common.cuh"
#include "context_gpu.cuh"
#include "graph_gpu.cuh"
#include "mce_gpu.cuh"
#include "queue.h"

namespace gkp
{

#define val auto const
#define var auto

constexpr uint32_t INVALID_VID = 0xffff'ffffU;

// The number of threads per block should be not greater than 1024, and be a multiple of 32.
// Only threads with threadIdx.x < 32 can obtain the summation.
template <typename Addable>
    requires((std::integral<Addable> || std::floating_point<Addable>) && sizeof(Addable) >= 4)
__forceinline__ __device__ Addable SumBlockwise(Addable a)
{
#if DEBUG
    if (blockDim.x > 1024 || blockDim.y > 1 || blockDim.z > 1 || (blockDim.x & 0x1f))
    {
        LOG("Incompatible configuration\tblockDim = {",
            blockDim.x,
            ", ",
            blockDim.y,
            ", ",
            blockDim.z,
            "}");
        exit(__LINE__);
    }
    if (__activemask() != 0xffff'ffffU)
    {
        LOG("Divergent deadlock\tactivemast = ", std::format("%x", __activemask()));
        exit(__LINE__);
    }
#endif // DEBUG
    a += __shfl_xor_sync(0xffff'ffffU, a, 1);
    a += __shfl_xor_sync(0xffff'ffffU, a, 2);
    a += __shfl_xor_sync(0xffff'ffffU, a, 4);
    a += __shfl_xor_sync(0xffff'ffffU, a, 8);
    a += __shfl_xor_sync(0xffff'ffffU, a, 16);
    if (blockDim.x == 32)
        return a;
    __shared__ Addable t[32];
    val wid = threadIdx.x >> 5;
    if ((threadIdx.x & 0x1f) == 0)
        t[wid] = a;
    __syncthreads();
    if (wid == 0)
    {
        a = threadIdx.x < ((blockDim.x + 31) >> 5) ? t[threadIdx.x] : static_cast<Addable>(0);
        a += __shfl_xor_sync(0xffff'ffffU, a, 1);
        a += __shfl_xor_sync(0xffff'ffffU, a, 2);
        a += __shfl_xor_sync(0xffff'ffffU, a, 4);
        a += __shfl_xor_sync(0xffff'ffffU, a, 8);
        a += __shfl_xor_sync(0xffff'ffffU, a, 16);
    }
    __syncthreads(); // In case that this kernel is successively invoked, ...
    return a;
}

template <typename VERTEX>
__global__ void filter1(
    GraphGpu const g, auto const* __restrict__ degree, Queue<VERTEX> d1, Queue<VERTEX> d2)
{
    val tid = blockDim.x * blockIdx.x + threadIdx.x;
    val stride = blockDim.x * gridDim.x;
    val end = ((g.num_vertices_ + 31) >> 5) << 5;
    for (var i = tid; i < end; i += stride)
    {
        d1.addWarpwise(i, i < g.num_vertices_ && degree[i] == 1);
        d2.addWarpwise(i, i < g.num_vertices_ && degree[i] == 2);
    }
}

template <typename VERTEX>
__global__ void filter2(
    GraphGpu const g,
    auto* survival,
    auto* degree,
    Queue<VERTEX> d2,
    Queue<VERTEX> triangle,
    uint32_t* count)
{
    val tid = blockDim.x * blockIdx.x + threadIdx.x;
    val stride = blockDim.x * gridDim.x;
    for (var i = tid; i < g.num_vertices_; i += stride)
    {
        if (i < g.num_vertices_ && survival[i] && degree[i] <= 0)
            survival[i] = 0;
    }
    __syncthreads();

    for (var i = tid; i < d2.size(); i += stride)
    {
        bool isTriangle = false;
        if (val v = d2[i]; degree[v] == 2)
        {
            uint32_t nghb1 = INVALID_VID, nghb2 = INVALID_VID;
            for (var j = g.rowoffset_[v]; j < g.rowoffset_[v + 1]; ++j)
            {
                if (val u = g.colidx_[j]; survival[u])
                {
                    nghb1 = u;
                    nghb2 = nghb1;
                }
            }
            // Look through N(n_1), for n_2 might be invalid.
            for (var j = g.rowoffset_[nghb1]; j < g.rowoffset_[nghb1 + 1]; ++j)
            {
                if (g.colidx_[j] == nghb2)
                    isTriangle = true;
            }
            triangle.addWarpwise(v, isTriangle, __activemask());
            int32_t dif = 0;
            if (!isTriangle)
            {
                int32_t t;
                t = atomicSub(degree + v, 2);
                dif += min(max(t, 0), 2);
                if (nghb1 != INVALID_VID && survival[nghb1])
                    t = max(atomicSub(degree + nghb1, 2), 0);
                if (nghb2 != INVALID_VID && survival[nghb2])
                    dif += max(atomicSub(degree + nghb2, 2), 0);
                dif = SumBlockwise(dif);
            }
            if (threadIdx.x == 0)
                atomicAdd(count, dif);
        }
    }
}

#undef var
#undef val

} // namespace gkp

#endif // PEELING_H
