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
#include <type_traits>

namespace gkp
{

#define val auto const
#define var auto

template <typename VERTEX>
__global__ void peelingFirstFilter(
    GraphGpu const g, auto const* __restrict__ degree, Queue<VERTEX> d1, Queue<VERTEX> d2)
{
    val tid = blockDim.x * blockIdx.x + threadIdx.x;
    val stride = blockDim.x * gridDim.x;
    val end = ((g.num_vertices_ + 31) >> 5) << 5;
    for (var i = tid; i < end; i += stride) {
        d1.addWarpwise(i, i < g.num_vertices_ && degree[i] == 1);
        d2.addWarpwise(i, i < g.num_vertices_ && degree[i] == 2);
    }
    if (tid == 0) {
        LOG("d1.size = ", d1.size(), " d2.size = ", d2.size());
    }
}

template <typename VERTEX>
__global__ void peelingFilter(
    GraphGpu const g,
    auto* degree,
    auto* survival,
    Queue<VERTEX> frontier,
    Queue<VERTEX> d1,
    Queue<VERTEX> d2)
{
    val tid = blockDim.x * blockIdx.x + threadIdx.x;
    val stride = blockDim.x * gridDim.x;
    val end = ((frontier.size() + 31) >> 5) << 5;
    if (tid == 0) LOG("");
    for (var i = tid; i < end; i += stride) {
        val v = i < frontier.size() ? frontier[i] : 0;
        if (degree[v] < 0) degree[v] = 0;
        if (degree[v] == 0) survival[v] = 0;
        d1.addWarpwise(v, i < frontier.size() && degree[v] == 1);
        d2.addWarpwise(v, i < frontier.size() && degree[v] == 2);
    }
    if (tid == 0) {
        LOG("d1.size = ", d1.size(), " d2.size = ", d2.size());
    }
}

template <typename VERTEX>
__global__ void peelingLeaf(
    GraphGpu const g,
    auto* degree,
    auto* survival,
    Queue<VERTEX> d1,
    uint32_t* count,
    Queue<VERTEX> next)
{
    val tid = blockDim.x * blockIdx.x + threadIdx.x;
    val stride = blockDim.x * gridDim.x;
    val end = ((d1.size() + 31) >> 5) << 5;
    if (tid == 0) LOG("");
    uint32_t cnt = 0;
    for (var i = tid; i < end; i += stride) {
        var u = INVALID_VID;
        if (val v = i < d1.size() ? d1[i] : 0; i < d1.size() && survival[v]) {
            survival[v] = 0;
            cnt += atomicSub(degree + v, 1) > 1 ? 1 : 0;
            for (var j = g.rowoffset_[v]; j < g.rowoffset_[v + 1]; ++j) {
                u = g.colidx_[j];
                if (degree[u] <= 0 || survival[u] == 0) {
                    u = INVALID_VID;
                }
                else {
                    val t = atomicSub(degree + u, 1);
                    if (t == 1) survival[u] = 0;
                    cnt += t > 0 ? 1 : 0;
                    if (t != 2 && t != 3) u = INVALID_VID;
                }
            }
        }
        next.addWarpwise(u, u != INVALID_VID);
    }
    cnt = sumBlockwise(cnt);
    if (tid == 0) {
        d1.clear();
        atomicAdd(count, cnt);
    }
}

template <typename VERTEX>
__global__ void peelingTriangle(
    GraphGpu const g,
    auto* degree,
    auto* survival,
    Queue<VERTEX> d3,
    uint32_t* count,
    Queue<VERTEX> next)
{
    val tid = blockDim.x * blockIdx.x + threadIdx.x;
#if 0
    val stride = blockDim.x * gridDim.x;
    uint32_t cnt = 0;
    if (tid == 0) LOG("");
    for (var i = tid; i < d3.size(); i += stride) {
        uint32_t nghb1 = INVALID_VID, nghb2 = INVALID_VID;
        if (val v = d3[i]; degree[v] == 2) {
            for (var j = g.rowoffset_[v]; j < g.rowoffset_[v + 1]; ++j) {
                val u = g.colidx_[j];
                if (survival[u]) {
                    nghb2 = nghb1;
                    nghb1 = u;
                }
            }

            if (nghb1 != INVALID_VID && nghb2 != INVALID_VID) {
                if (degree[nghb1] > 2 && degree[nghb2] > 2 || v < nghb1 && v < nghb2) {
                    int32_t t = atomicSub(degree + v, 2);
                    if (t != 2) {
                        LOG("A vertex classified as of degree 2 turns out to be not degree 2. v = ", v,
                            " degree[v]: ", t, " -> ", t - 2, " n_1 = ", nghb1, " n_2 = ", nghb2);
                        assert(0);
                    }
                    ++cnt;
                    survival[v] = 0;
                    t = atomicSub(degree + nghb1, 1);
                    if (t == 2) {
                        degree[nghb1] = 0;
                        survival[nghb1] = 0;
                        nghb1 = INVALID_VID;
                    }
                    else if (t != 3) {
                        nghb1 = INVALID_VID;
                    }
                    t = atomicSub(degree + nghb2, 1);
                    if (t == 2) {
                        degree[nghb2] = 0;
                        survival[nghb2] = 0;
                        nghb2 = INVALID_VID;
                    }
                    else if (t != 3) {
                        nghb2 = INVALID_VID;
                    }
                }
                else {
                    nghb1 = INVALID_VID;
                    nghb2 = INVALID_VID;
                }
            }
        }
        next.addWarpwise(nghb1, nghb1 != INVALID_VID, __activemask());
        next.addWarpwise(nghb2, nghb2 != INVALID_VID, __activemask());
    }
    cnt = sumBlockwise(cnt);
#endif
    if (tid == 0) {
        d3.clear();
        //    atomicAdd(count, cnt);
    }
}

template <typename VERTEX>
__global__ void peelingBridge(
    GraphGpu const g,
    auto* degree,
    auto* survival,
    Queue<VERTEX> d2,
    Queue<VERTEX> next,
    uint32_t* count)
{
    val tid = blockDim.x * blockIdx.x + threadIdx.x;
    val stride = blockDim.x * gridDim.x;
    if (tid == 0) LOG("");
    uint32_t cnt = 0;
    for (var i = tid; i < d2.size(); i += stride) {
        uint32_t nghb1 = INVALID_VID, nghb2 = INVALID_VID;
        if (val v = d2[i]; survival[v]) {
            bool isTriangle = false;
            for (var j = g.rowoffset_[v]; j < g.rowoffset_[v + 1]; ++j) {
                if (val u = g.colidx_[j]; survival[u]) {
                    nghb2 = nghb1;
                    nghb1 = u;
                }
            }
            // Go through N(n_1), for n_2 might be invalid.
            for (var j = g.rowoffset_[nghb1]; j < g.rowoffset_[nghb1 + 1]; ++j) {
                if (g.colidx_[j] == nghb2) isTriangle = true;
            }

            if (!isTriangle) {
                int32_t t;
                survival[v] = 0;
                t = atomicSub(degree + v, 2);
                cnt += max(t, 0);
                if (nghb1 != INVALID_VID && survival[nghb1]) {
                    t = atomicSub(degree + nghb1, 1);
                    if (t > 0) ++cnt;
                    if (t <= 1) survival[nghb1] = 0;
                    if (t != 2 && t != 3) nghb1 = INVALID_VID;
                }
                if (nghb2 != INVALID_VID && survival[nghb2]) {
                    t = atomicSub(degree + nghb2, 1);
                    if (t > 0) ++cnt;
                    if (t <= 1) survival[nghb2] = 0;
                    if (t != 2 && t != 3) nghb2 = INVALID_VID;
                }
            }
        }
        next.addWarpwise(nghb1, nghb1 != INVALID_VID, __activemask());
        next.addWarpwise(nghb2, nghb2 != INVALID_VID, __activemask());
    }
    cnt = sumBlockwise(cnt);
    if (tid == 0) {
        d2.clear();
        atomicAdd(count, cnt);
    }
}

#undef var
#undef val

} // namespace gkp

#endif // PEELING_H
