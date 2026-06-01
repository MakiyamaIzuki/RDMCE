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

struct PGThreshold
{
    static constexpr int32_t alpha = 32;
    static constexpr int32_t beta = 1024;
};

__global__ void pt(
    GraphGpu const g,
    auto const* __restrict__ degree_in,
    auto* __restrict__ degree_out,
    auto* __restrict__ survival)
{
    val tid = blockDim.x * blockIdx.x + threadIdx.x;
    val stride = blockDim.x * gridDim.x;
    for (var i = tid; i < g.num_vertices_; i += stride) {
        if (degree_in[i] > 1 && degree_in[i] < PGThreshold::alpha) {
            int32_t diff = 0;
            for (var j = g.rowoffset_[i]; j < g.rowoffset_[i + 1]; ++j) {
                val n = g.colidx_[j];
                if (degree_in[n] == 1) {
                    diff += 1;
                    survival[n] = 0;
                }
            }
            degree_out[i] = degree_in[i] - diff;
        }
    }
//    __syncthreads();
}

__global__ void pw(
    GraphGpu const g,
    auto const* __restrict__ degree_in,
    auto* __restrict__ degree_out,
    auto* __restrict__ survival)
{
    val wid = (blockDim.x * blockIdx.x + threadIdx.x) >> 5;
    val lid = threadIdx.x & 0x1f;
    val stride = blockDim.x * gridDim.x >> 5;
    for (var i = wid; i < g.num_vertices_; i += stride) {
        if (degree_in[i] >= PGThreshold::alpha && degree_in[i] < PGThreshold::beta) {
            int32_t diff = 0;
            for (var j = g.rowoffset_[i] + lid; j < g.rowoffset_[i + 1]; j += 32) {
                val n = g.colidx_[j];
                if (degree_in[n] == 1) {
                    diff += 1;
                    survival[n] = 0;
                }
            }
            diff = sumWarpwise(diff);
            if (lid == 0) degree_out[i] = degree_in[i] - diff;
        }
    }
}

__global__ void pb(
    GraphGpu const g,
    auto const* __restrict__ degree_in,
    auto* __restrict__ degree_out,
    auto* __restrict__ survival)
{
    val tid = blockDim.x * blockIdx.x + threadIdx.x;
    val stride = blockDim.x * gridDim.x;
    val end = g.num_vertices_ / 2;
    for (var i = g.num_vertices_ - 1; i >= end; --i) {
        if (degree_in[i] >= PGThreshold::beta) {
            int32_t diff = 0;
            for (var j = g.rowoffset_[i] + tid; j < g.rowoffset_[i + 1]; j += stride) {
                val n = g.colidx_[j];
                if (degree_in[n] == 1) {
                    diff += 1;
                    survival[n] = 0;
                }
            }
            diff = sumBlockwise(diff);
            if (threadIdx.x == 0) degree_out[i] = degree_in[i] - diff;
        }
    }
}

template <typename Vid>
__global__ void peelingFirstFilter(
    GraphGpu const g, auto const* __restrict__ degree, Queue<Vid> d1, Queue<Vid> d2)
{
    val tid = blockDim.x * blockIdx.x + threadIdx.x;
    val stride = blockDim.x * gridDim.x;
    val end = ((g.num_vertices_ + 31) >> 5) << 5;
    for (var i = tid; i < end; i += stride) {
        d1.addWarpwise(i, i < g.num_vertices_ && degree[i] == 1);
        d2.addWarpwise(i, i < g.num_vertices_ && degree[i] == 2);
    }
}

template <typename Vid>
__global__ void peelingFilter(
    GraphGpu const g,
    auto* degree,
    auto* tag,
    Queue<Vid> const frontier,
    Queue<Vid> d1,
    Queue<Vid> d2)
{
    val tid = blockDim.x * blockIdx.x + threadIdx.x;
    val stride = blockDim.x * gridDim.x;
    val end = (frontier.size() + 31) & ~31ULL;
    if (tid == 0) LOG("");
    for (var i = tid; i < end; i += stride) {
        val v = i < frontier.size() ? frontier[i] : 0;
        if (i < frontier.size() && tag[v]) {
            val d = degree[v];
            if (d < 0) degree[v] = 0;
            if (d <= 0) tag[v] = 0;
            d1.addWarpwise(v, d == 1 && atomicCAS(tag + v, 1, 3) == 1);
            d2.addWarpwise(v, d == 2 && atomicCAS(tag + v, 1, 3) == 1);
        }
        else {
            d1.addWarpwise(v, false);
            d2.addWarpwise(v, false);
        }
    }
}

template <typename Vid>
__global__ void peelingLeaf(
    GraphGpu const g,
    auto* degree,
    auto* survival,
    Queue<Vid> const d1,
    uint32_t* count,
    Queue<Vid> next)
{
    val tid = blockDim.x * blockIdx.x + threadIdx.x;
    val stride = blockDim.x * gridDim.x;
    val end = (d1.size() + 31) & ~31ULL;
    if (tid == 0) LOG("");
    uint32_t cnt = 0;
    for (var i = tid; i < end; i += stride) {
        var affected = INVALID_VID;
        if (val v = i < d1.size() ? d1[i] : 0; i < d1.size() && survival[v]) {
            survival[v] = 0;
            val t = atomicExch(degree + v, 0);
            if (t == 1) {
                ++cnt;
                for (var j = g.rowoffset_[v]; j < g.rowoffset_[v + 1]; ++j) {
                    if (val u = g.colidx_[j]; survival[u]) {
                        val t = atomicSub(degree + u, 1);
                        cnt += t > 0 ? 1 : 0;
                        switch (t) {
                        case 1:
                            survival[u] = 0;
                            break;
                        case 2:
                        case 3:
                            affected = u;
                            break;
                        }
                    }
                }
            }
        }
        next.addWarpwise(affected, affected != INVALID_VID);
    }
    cnt = sumBlockwise(cnt);
    if (threadIdx.x == 0) atomicAdd(count, cnt);
}

template <typename Vid>
__global__ void peelingTriangle(
    GraphGpu const g, auto* degree, auto* survival, Queue<Vid> const tri, uint32_t* count)
{
    val tid = blockDim.x * blockIdx.x + threadIdx.x;
    val stride = blockDim.x * gridDim.x;
    uint32_t cnt = 0;
    if (tid == 0) LOG("");
    for (var i = tid; i < tri.size(); i += stride) {
        uint32_t nghb1 = INVALID_VID, nghb2 = INVALID_VID;
        if (i < tri.size()) {
            val v = tri[i];
            for (var j = g.rowoffset_[v]; j < g.rowoffset_[v + 1]; ++j) {
                if (val u = g.colidx_[j]; degree[u] == 2) {
                    nghb2 = nghb1;
                    nghb1 = u;
                }
            }

            if (v > nghb1 && v > nghb2) {
                var t = atomicExch(degree + v, 0);
                if (t != 2) {
                    LOG("Vertex ", v, " in tri turns out to be not degree 2. v = ", v,
                        " degree[v]: ", t, " -> ", t - 2, " n_1 = ", nghb1, " n_2 = ", nghb2);
                    assert(0);
                }
                if (atomicExch(degree + nghb1, 0) != 2) {
                    LOG("Vertex ", nghb1, " (N of ", v, ") is not of degree 2.");
                    assert(0);
                }
                if (atomicExch(degree + nghb2, 0) != 2) {
                    LOG("Vertex ", nghb2, " (N of ", v, ") is not of degree 2.");
                    assert(0);
                }
                ++cnt;
                survival[v] = 0;
                degree[nghb1] = degree[nghb2] = 0;
                survival[nghb1] = survival[nghb2] = 0;
            }
        }
    }
    cnt = sumBlockwise(cnt);
    if (threadIdx.x == 0) atomicAdd(count, cnt);
}

template <typename Vid>
__global__ void peelingBridge(
    GraphGpu const g,
    auto* degree,
    auto* survival,
    Queue<Vid> const d2,
    Queue<Vid> next,
    Queue<Vid> tri,
    uint32_t* count)
{
    val tid = blockDim.x * blockIdx.x + threadIdx.x;
    val stride = blockDim.x * gridDim.x;
    val end = (d2.size() + 31) & ~31ULL;
    if (tid == 0) LOG("");
    uint32_t cnt = 0;
    for (var i = tid; i < end; i += stride) {
        uint32_t nghb1 = INVALID_VID, nghb2 = INVALID_VID;
        bool isTriangle = false;
        val v = i < d2.size() ? d2[i] : 0;
        if (i < d2.size() && survival[v]) {
            for (var j = g.rowoffset_[v]; j < g.rowoffset_[v + 1]; ++j) {
                if (val u = g.colidx_[j]; survival[u]) {
                    nghb2 = nghb1;
                    nghb1 = u;
                }
            }
            if (nghb1 == INVALID_VID) {
                survival[v] = 0;
                degree[v] = 0;
                LOG("Vertex ", v, " in d2 has no neighbor.");
            }
            else {
                for (var j = g.rowoffset_[nghb1]; j < g.rowoffset_[nghb1 + 1]; ++j) {
                    if (g.colidx_[j] == nghb2) {
                        isTriangle = true;
                        nghb1 = nghb2 = INVALID_VID;
                        break;
                    }
                }

                if (!isTriangle) {
                    int32_t t;
                    survival[v] = 0;
                    // `t` can be negative, for a cascaded vertex may be added into frontier queue
                    // more than one times.
                    t = atomicExch(degree + v, 0);
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
        }
        tri.addWarpwise(v, isTriangle);
        next.addWarpwise(nghb1, nghb1 != INVALID_VID);
        next.addWarpwise(nghb2, nghb2 != INVALID_VID);
    }
    cnt = sumBlockwise(cnt);
    if (threadIdx.x == 0) atomicAdd(count, cnt);
}

#undef var
#undef val

} // namespace gkp

#endif // PEELING_H
