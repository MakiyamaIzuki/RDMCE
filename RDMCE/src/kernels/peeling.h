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

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <thrust/swap.h>
#include <type_traits>

namespace gkp
{

#define val auto const
#define var auto

template <typename Vid>
__global__ void peelingFirstFilter(
    GraphGpu const g, auto const* __restrict__ degree, Queue<Vid> d1, Queue<Vid> d2)
{
    val tid = blockDim.x * blockIdx.x + threadIdx.x;
    val stride = blockDim.x * gridDim.x;
    val end = (g.num_vertices_ + 31) & ~31ULL;
    for (var i = tid; i < end; i += stride) {
        d1.addWarpwise(i, i < g.num_vertices_ && degree[i] == 1);
        d2.addWarpwise(i, i < g.num_vertices_ && degree[i] == 2);
    }
}

template <typename Vid>
__global__ void peelLeaf(
    GraphGpu const g,
    auto* degree,
    auto* survival,
    Queue<Vid> const d1,
    Queue<Vid> nd1,
    Queue<Vid> nd2,
    uint32_t* count)
{
    val tid = blockDim.x * blockIdx.x + threadIdx.x;
    val stride = blockDim.x * gridDim.x;
    val size = d1.size();
    val end = (size + 31) & ~31ULL;
    if (tid == 0) LOG("");
    uint32_t cnt = 0;
    for (var i = tid; i < end; i += stride) {
        int32_t t = -1;
        var u = INVALID_VID;
        if (val v = i < size ? d1[i] : 0; i < size && survival[v]) {
            survival[v] = 0;
            if (atomicExch(degree + v, 0) == 1) {
                ++cnt;
                for (var j = g.rowoffset_[v]; j < g.rowoffset_[v + 1]; ++j) {
                    if (u = g.colidx_[j]; survival[u]) {
                        t = atomicSub(degree + u, 1);
                        cnt += t > 0 ? 1 : 0;
                        if (t == 1) survival[u] = 0;
                    }
                }
            }
        }
        nd1.addWarpwise(u, t == 2);
        nd2.addWarpwise(u, t == 3);
    }
    cnt = sumBlockwise(cnt);
    if (threadIdx.x == 0) atomicAdd(count, cnt);
}

__forceinline__ __device__ void peelIsolatedTriangle(
    auto* degree, auto* survival, auto a, auto b, auto c, auto& cnt)
{
    if (atomicCAS(degree + a, 2, -1) != 2) return;
    if (atomicCAS(degree + b, 2, -1) != 2) {
        atomicExch(degree + a, 2);
        // LOG("Restore... ", c, " ", b, " ", a);
        return;
    }
    if (atomicCAS(degree + c, 2, -1) != 2) {
        atomicExch(degree + b, 2);
        atomicExch(degree + a, 2);
        // LOG("Restore... ", c, " ", b, " ", a);
        return;
    }
    // LOG(a, " ", b, " ", c);
    ++cnt;
    degree[a] = degree[b] = degree[c] = 0;
    survival[a] = survival[b] = survival[c] = 0;
}

__forceinline__ __device__ void peelAttachedTriangle(
    auto* degree, auto* survival, auto n1, auto n2, auto c, auto& cnt, auto& od1, auto& od2)
{
    if ((od1 = atomicSub(degree + n1, 1)) < 3) {
        atomicAdd(degree + n1, 1);
        od1 = -1;
        LOG("Restore... ", c, " ", n2, " ", n1);
        return;
    }
    if ((od2 = atomicSub(degree + n2, 1)) < 3) {
        atomicAdd(degree + n2, 1);
        atomicAdd(degree + n1, 1);
        od2 = -1;
        od1 = -1;
        // LOG("Restore... ", c, " ", n2, " ", n1);
        return;
    }
    if (atomicCAS(degree + c, 2, 0) != 2) {
        atomicAdd(degree + n2, 1);
        atomicAdd(degree + n1, 1);
        od2 = -1;
        od1 = -1;
        // LOG("Restore... ", c, " ", n2, " ", n1);
        return;
    }
    // LOG(a, " ", b, " ", c);
    // LOG("d(v_a) = ", degree[a], " d(v_b) = ", degree[b]);
    ++cnt;
    survival[c] = 0;
}

template <typename Vid>
__global__ void peelBridge(
    GraphGpu const g,
    auto* degree,
    auto* survival,
    Queue<Vid> const d2,
    Queue<Vid> next1,
    Queue<Vid> next2,
    uint32_t* count2,
    uint32_t* count3)
{
    val tid = blockDim.x * blockIdx.x + threadIdx.x;
    val stride = blockDim.x * gridDim.x;
    val size = d2.size();
    val end = (size + 31) & ~31ULL;
    if (tid == 0) LOG("");
    uint32_t cnt2 = 0U, cnt3 = 0U;
    for (var i = tid; i < end; i += stride) {
        uint32_t nghb1 = INVALID_VID, nghb2 = INVALID_VID;
        bool isTriangle = false;
        int32_t od1 = -1, od2 = -1;
        if (val v = i < size ? d2[i] : 0; i < size && survival[v]) {
            for (var j = g.rowoffset_[v]; j < g.rowoffset_[v + 1]; ++j) {
                if (val u = g.colidx_[j]; survival[u]) {
                    nghb2 = nghb1;
                    nghb1 = u;
                }
            }
            if (nghb1 == INVALID_VID) {
                // No alive neighbor is found; counting is done by other threads.
                survival[v] = 0;
                degree[v] = 0;
            }
            else {
                for (var j = g.rowoffset_[nghb1]; j < g.rowoffset_[nghb1 + 1]; ++j) {
                    if (g.colidx_[j] == nghb2) {
                        isTriangle = true;
                        break;
                    }
                }

                if (isTriangle) {
                    if (v < nghb1 && v < nghb2 && degree[nghb1] == 2 && degree[nghb2] == 2)
                        peelIsolatedTriangle(degree, survival, nghb1, nghb2, v, cnt3);
                    if (degree[nghb1] > 2 && degree[nghb2] > 2)
                        peelAttachedTriangle(degree, survival, nghb1, nghb2, v, cnt3, od1, od2);
                }
                else {
                    survival[v] = 0;
                    val t = atomicExch(degree + v, 0);
                    if (t < 0) LOG("t = ", t);
                    cnt2 += max(t, 0);
                    if (nghb1 != INVALID_VID && survival[nghb1]) {
                        od1 = atomicSub(degree + nghb1, 1);
                        if (od1 > 0) ++cnt2;
                        if (od1 <= 1) survival[nghb1] = 0;
                    }
                    if (nghb2 != INVALID_VID && survival[nghb2]) {
                        od2 = atomicSub(degree + nghb2, 1);
                        if (od2 > 0) ++cnt2;
                        if (od2 <= 1) survival[nghb2] = 0;
                    }
                }
            }
        }
        next1.addWarpwise(nghb1, od1 == 2);
        next2.addWarpwise(nghb1, od1 == 3);
        next1.addWarpwise(nghb2, od2 == 2);
        next2.addWarpwise(nghb2, od2 == 3);
    }
    cnt2 = sumBlockwise(cnt2);
    cnt3 = sumBlockwise(cnt3);
    if (threadIdx.x == 0) {
        atomicAdd(count2, cnt2);
        atomicAdd(count3, cnt3);
    }
}

#undef var
#undef val

} // namespace gkp

#endif // PEELING_H
