#pragma once
#ifndef PEELING_H
#define PEELING_H

#include "common.cuh"
#include "context_gpu.cuh"
#include "cuda.h"
#include "cuda_runtime.h"
#include "gkp.h"
#include "graph_gpu.cuh"
#include "mce_gpu.cuh"
#include "queue.h"

#include <cstdint>
#include <type_traits>

namespace gkp
{

#define val auto const
#define var auto

template <typename VERTEX>
__global__ void peeling_init(
    GraphGpu const g, auto const* __restrict__ degree, Queue<VERTEX> d1, Queue<VERTEX> d2)
{
    val tid = blockDim.x * blockIdx.x + threadIdx.x;
    val stride = blockDim.x * gridDim.x;
    val end = ((g.num_vertices_ + 31) >> 5) << 5;
    for (var i = tid; i < end; i += stride) {
        d1.addWarpwise(i, i < g.num_vertices_ && degree[i] == 1);
        d2.addWarpwise(i, i < g.num_vertices_ && degree[i] == 2);
    }
}

template <typename VERTEX>
__global__ void peelingFilter(
    GraphGpu const g,
    auto const* __restrict__ degree,
    Queue<VERTEX> frontier,
    Queue<VERTEX> d1,
    Queue<VERTEX> d2)
{
    val tid = blockDim.x * blockIdx.x + threadIdx.x;
    val stride = blockDim.x * gridDim.x;
    val end = ((frontier.size() + 31) >> 5) << 5;
    for (var i = tid; i < end; i += stride) {
        d1.addWarpwise(frontier[i], i < frontier.size() && degree[i] == 1);
        d2.addWarpwise(frontier[i], i < frontier.size() && degree[i] == 2);
    }
}

template <typename VERTEX>
__global__ void peelingLeaf(
    GraphGpu const g, auto* degree, auto* survival, Queue<VERTEX> d1, uint32_t* count)
{
    val tid = blockDim.x * blockIdx.x + threadIdx.x;
    val stride = blockDim.x * gridDim.x;
    val end = ((d1.size() + 31) >> 5) << 5;
    uint32_t cnt = 0;
    for (var i = tid; i < end; i += stride) {
        if (val v = d1[i]; survival[v]) {
            survival[v] = false;
            cnt += degree[v] == 1 ? atomicSub(degree, 1) : 0;
        }
    }
    cnt = sumBlockwise(cnt);
    if (threadIdx.x == 0) atomicAdd(count, cnt);
}

template <typename VERTEX>
__global__ void peelingTriangle(
    GraphGpu const g, auto* degree, auto* survival, Queue<VERTEX> d2, uint32_t* count)
{
    val tid = blockDim.x * blockIdx.x + threadIdx.x;
    val stride = blockDim.x * gridDim.x;
    val end = ((d2.size() + 31) >> 5) << 5;
    uint32_t cnt = 0;
    for (var i = tid; i < end; i += stride) {
        if (val v = d2[i]; degree[v] == 2) {
            uint32_t nghb1 = INVALID_VID, nghb2 = INVALID_VID;
            for (var j = g.rowoffset_[v]; j < g.rowoffset_[v + 1]; ++j) {
                val u = g.colidx_[j];
                if (survival[u]) {
                    nghb2 = nghb1;
                    nghb1 = u;
                }
            }
        }
    }
    if (threadIdx.x == 0) atomicAdd(count, cnt);
}

template <typename VERTEX>
__global__ void peeling_bridge(
    GraphGpu const g,
    auto* survival,
    auto* degree,
    Queue<VERTEX> d2,
    Queue<VERTEX> next,
    Queue<VERTEX> triangle,
    uint32_t* count)
{
    val tid = blockDim.x * blockIdx.x + threadIdx.x;
    val stride = blockDim.x * gridDim.x;
    for (var i = tid; i < g.num_vertices_; i += stride) {
        if (i < g.num_vertices_ && survival[i] && degree[i] <= 0) survival[i] = 0;
    }
    __syncthreads();

    for (var i = tid; i < d2.size(); i += stride) {
        bool isTriangle = false;
        if (val v = d2[i]; degree[v] == 2) {
            uint32_t nghb1 = INVALID_VID, nghb2 = INVALID_VID;
            for (var j = g.rowoffset_[v]; j < g.rowoffset_[v + 1]; ++j) {
                if (val u = g.colidx_[j]; survival[u]) {
                    nghb1 = u;
                    nghb2 = nghb1;
                }
            }
            // Look through N(n_1), for n_2 might be invalid.
            for (var j = g.rowoffset_[nghb1]; j < g.rowoffset_[nghb1 + 1]; ++j) {
                if (g.colidx_[j] == nghb2) isTriangle = true;
            }
            triangle.addWarpwise(v, isTriangle, __activemask());

            int32_t dif = 0;
            if (!isTriangle) {
                int32_t t;
                t = atomicSub(degree + v, 2);
                dif += min(max(t, 0), 2);
                if (nghb1 != INVALID_VID && survival[nghb1]) {
                    t = atomicSub(degree + nghb1, 1);
                    if (t > 0) ++dif;
                    if (t > 3) nghb1 = INVALID_VID;
                }
                if (nghb2 != INVALID_VID && survival[nghb2]) {
                    t = atomicSub(degree + nghb2, 1);
                    if (t > 0) ++dif;
                    if (t > 3) nghb2 = INVALID_VID;
                }
                dif = sumBlockwise(dif);
            }
            if (threadIdx.x == 0) atomicAdd(count, dif);
            next.addWarpwise(nghb1, nghb1 != INVALID_VID, __activemask());
            next.addWarpwise(nghb2, nghb2 != INVALID_VID, __activemask());
        }
    }
}

#undef var
#undef val

} // namespace gkp

#endif // PEELING_H
