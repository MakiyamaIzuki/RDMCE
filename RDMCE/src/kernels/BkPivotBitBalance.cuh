#pragma once
#include "common.cuh"
#include "context_gpu.cuh"
#include "gkp.h"
#include "graph_gpu.cuh"
#include "mce_gpu.cuh"
#include "peeling.h"
#include "queue.h"

#include <cassert>
#include <cerrno>
#include <chrono>
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/pair.h>
#include <thrust/swap.h>
#include <vector>

#ifndef TASK_SHARE_BOUND
#define TASK_SHARE_BOUND 24
#endif

// 1. User defined vertex
#define bitvec_t uint64_t

// 2. User defined enumeration tree node
struct Metadata
{
    bitvec_t* bitmap_base;
    size_t p_size, x_size;
    bitvec_t* rpv_base;
    int32_t task_count;
    int32_t pending_count;
};

// 3. User defined context manager
class ContextManagerBkpb : public ContextManager
{
public:
    // BufferManager bitmaps_;
    // BufferManager rpxv_;
    BufferManager buffers_;
    BufferManager metas_;

    vid_t* current_id;
    vid_t* active_warps_;
    // vid_t *debug_val;
};

// 4. User defined kernel function

__device__ void SetTaskById(
    bitvec_t* bitmap_base, size_t p_size, bitvec_t* rpv_src, bitvec_t* rpv_dst, int remain_bits)
{

    size_t sizeOfItem = round_up(p_size, sizeof(bitvec_t) * 8) / sizeof(bitvec_t) / 8;

    size_t p_id;
    for (int i = 0; i < sizeOfItem; i++) {
        rpv_dst[i] = rpv_src[i];
        rpv_dst[i + sizeOfItem] = rpv_src[i + sizeOfItem] & ~rpv_src[i + 2 * sizeOfItem];
    }

    for (int i = sizeOfItem - 1; i >= 0 && remain_bits > 0; i--) {
        bitvec_t vec = rpv_src[i + 2 * sizeOfItem];
        if (__popcll(vec) < remain_bits) {
            remain_bits -= __popcll(vec);
            rpv_dst[i + sizeOfItem] |= vec;
        }
        else {
            while (__popcll(vec) > remain_bits) {
                vec = vec - (vec & -vec);
            }
            remain_bits = 0;
            rpv_dst[i] |= (vec & -vec);
            rpv_dst[i + sizeOfItem] |= vec;
            p_id = __ffsll(vec) - 1 + i * sizeof(bitvec_t) * 8;
        }
    }

    for (int i = 0; i < sizeOfItem; i++) {
        rpv_dst[i + sizeOfItem] = rpv_dst[i + sizeOfItem] & bitmap_base[p_id * sizeOfItem + i];
    }
}

__device__ acc_t
IterKernelAuto(ContextManagerBkpb& context, bitvec_t* bitmap_base, size_t p_size, size_t x_size)
{
    const size_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    bitvec_t* rpv = (bitvec_t*)context.buffers_.GetElement(warp_id);
    Metadata* meta = (Metadata*)context.metas_.GetElement(warp_id);
    acc_t mc_local = 0;
    int level = 0, level_sharing = -1, level_to_share = -1;
    size_t sizeOfItem = round_up(p_size, sizeof(bitvec_t) * 8) / sizeof(bitvec_t) / 8;
    const size_t itemPerVec = sizeof(bitvec_t) * 8;
    if (get_lane_id() == 0) {
        meta->task_count = 0;
        meta->pending_count = 0;
    }
    bool has_task = true;
    while (true) {
        if (level_sharing < level_to_share) {
            if (get_lane_id() == 0) { // share workload
                auto pending_count = meta->pending_count;
                if (pending_count == 0) {
                    level_sharing++;
                    bitvec_t* pp_base = rpv + (3 * level_sharing + 1) * sizeOfItem;
                    bitvec_t* vpp_base = rpv + (3 * level_sharing + 2) * sizeOfItem;
                    int task_count = 0;
                    for (int i = 0; i < sizeOfItem; i++)
                        task_count += __popcll(pp_base[i] & vpp_base[i]);
                    if (task_count > 0) {
                        meta->bitmap_base = bitmap_base;
                        meta->p_size = p_size;
                        meta->x_size = x_size;
                        meta->rpv_base = rpv + (3 * level_sharing) * sizeOfItem;
                        meta->pending_count = task_count;
                        __threadfence();
                        meta->task_count = task_count;
                        // atomicAdd(context.debug_val, task_count);
                        // printf("Set: %d %d\n", task_count, task_count);
                    }
                }
            }
            level_sharing = __shfl_sync(FULL_MASK, level_sharing, 0);
        }

        if (!has_task) {
            if (meta->pending_count == 0)
                break;
            else {
                if (get_lane_id() == 0) {
                    auto task_count = meta->task_count;
                    if (task_count > 0) {
                        auto old_value = atomicCAS(&meta->task_count, task_count, task_count - 1);
                        if (old_value == task_count) {
                            SetTaskById(
                                meta->bitmap_base, meta->p_size, meta->rpv_base,
                                rpv + 3 * sizeOfItem * (level_sharing + 1), task_count);
                            __threadfence();
                            auto pending_count = atomicSub(&meta->pending_count, 1);
                            //  printf("Acquire: %d %d\n", task_count - 1, pending_count - 1);
                            level = level_sharing + 1;
                            has_task = true;
                        }
                    }
                }

                has_task = __shfl_sync(FULL_MASK, has_task, 0);
                if (!has_task) {
                    cuda_sleep();
                    continue;
                }
            }
        }

        // find p
        // bitvec_t *rr_base = rpv + (3 * level) * sizeOfItem;
        // bitvec_t *pp_base = rpv + (3 * level + 1) * sizeOfItem;

        // vid_t pivot_id = 0;
        // size_t pid = INF;

        // size_t min_non_neighbor_p = INF;

        // for(int i = get_lane_id(); i < p_size + x_size; i += warpSize){
        //   size_t non_neighbor_p = 0;
        //   bool is_p_or_x = true;
        //   for(int j = 0; j < sizeOfItem && is_p_or_x; j++){
        //     if(rr_base[j] & ~bitmap_base[i * sizeOfItem + j])
        //       is_p_or_x = false;
        //     non_neighbor_p += __popcll(pp_base[j] & ~bitmap_base[i * sizeOfItem + j]);
        //   }
        //   if(is_p_or_x && non_neighbor_p < min_non_neighbor_p){
        //     min_non_neighbor_p = non_neighbor_p;
        //     pivot_id = i;
        //   }
        // }
        // size_t mnnp = warp_min(min_non_neighbor_p);
        // mnnp = __shfl_sync(FULL_MASK, mnnp, 0);

        // if(mnnp == INF) // is maximal
        //   mc_local ++;
        // else if(mnnp > 0){
        //   size_t cur_p_size = 0;
        //   unsigned match_mask = __ballot_sync(FULL_MASK, mnnp == min_non_neighbor_p);
        //   pivot_id = __shfl_sync(FULL_MASK, pivot_id, __ffs(match_mask) - 1);
        //   if (get_lane_id() == 0)
        //   {
        //     bitvec_t *vpp_base = rpv + (3 * level + 2) * sizeOfItem;
        //     for (size_t i = 0; i < sizeOfItem; i++)
        //     {
        //       bitvec_t res = ~bitmap_base[pivot_id * sizeOfItem + i] & pp_base[i];
        //       if (pid == INF && res != (bitvec_t)0ULL)
        //         pid = i * itemPerVec + __ffsll(res) - 1;
        //       vpp_base[i] = res;
        //       cur_p_size += __popcll(pp_base[i]);
        //     }
        //   }
        //   pid = __shfl_sync(FULL_MASK, pid, 0);
        //   cur_p_size = __shfl_sync(FULL_MASK, cur_p_size, 0);

        //   if(pid != INF && cur_p_size >= TASK_SHARE_BOUND){
        //     level_to_share = level;
        //   }
        // }

        bitvec_t* rr_base = rpv + (3 * level) * sizeOfItem;
        bitvec_t* pp_base = rpv + (3 * level + 1) * sizeOfItem;

        vid_t pivot_id = 0;
        size_t pid = INF;

        size_t cur_p_size = 0;
        if (get_lane_id() == 0) {
            for (size_t i = 0; i < sizeOfItem; i++)
                cur_p_size += __popcll(pp_base[i]);
        }
        cur_p_size = __shfl_sync(FULL_MASK, cur_p_size, 0);
        size_t min_non_neighbor_p = INF;

        // if (cur_p_size != 0){
        //   for(int i = get_lane_id(); i < p_size; i += warpSize){
        //     int pos = i / sizeof(bitvec_t) / 8;
        //     int offset = i % (sizeof(bitvec_t) * 8);
        //     bool is_p = i < p_size && ((pp_base[pos] & ((bitvec_t)1ULL << offset)) != 0ULL);

        //     size_t non_neighbor_p = 0;
        //     for (int j=0; is_p && j<sizeOfItem; j++)
        //       non_neighbor_p += __popcll(pp_base[j] & ~bitmap_base[i * sizeOfItem + j]);
        //     unsigned mask = __ballot_sync(__activemask(), non_neighbor_p == 1);
        //     if(get_lane_id() == 0 && mask != 0){
        //       bitvec_t new_r = (bitvec_t)mask << (i % (sizeof(bitvec_t) * 8));
        //       rr_base[i / sizeof(bitvec_t) / 8] |= new_r;
        //       pp_base[i / sizeof(bitvec_t) / 8] &= ~new_r;
        //       cur_p_size -= __popcll(new_r);
        //     }
        //   }
        //   cur_p_size = __shfl_sync(FULL_MASK, cur_p_size, 0);
        // }

        if (cur_p_size != 0) {
            for (int i = get_lane_id(); i < p_size + x_size && min_non_neighbor_p != 0;
                 i += warpSize)
            {
                size_t non_neighbor_p = 0;
                bool is_p_or_x = true;
                for (int j = 0; j < sizeOfItem; j++) {
                    if (rr_base[j] & ~bitmap_base[i * sizeOfItem + j]) {
                        is_p_or_x = false;
                        break;
                    }
                    non_neighbor_p += __popcll(pp_base[j] & ~bitmap_base[i * sizeOfItem + j]);
                }
                if (is_p_or_x && non_neighbor_p < min_non_neighbor_p) {
                    min_non_neighbor_p = non_neighbor_p;
                    pivot_id = i;
                }
            }

            size_t mnnp = warp_min(min_non_neighbor_p);
            mnnp = __shfl_sync(FULL_MASK, mnnp, 0);

            if (mnnp > 0) {
                unsigned match_mask = __ballot_sync(FULL_MASK, mnnp == min_non_neighbor_p);
                pivot_id = __shfl_sync(FULL_MASK, pivot_id, __ffs(match_mask) - 1);
                if (get_lane_id() == 0) {
                    bitvec_t* vpp_base = rpv + (3 * level + 2) * sizeOfItem;
                    for (size_t i = 0; i < sizeOfItem; i++) {
                        bitvec_t res = ~bitmap_base[pivot_id * sizeOfItem + i] & pp_base[i];
                        if (pid == INF && res != (bitvec_t)0ULL)
                            pid = i * itemPerVec + __ffsll(res) - 1;
                        vpp_base[i] = res;
                    }
                }
                pid = __shfl_sync(FULL_MASK, pid, 0);

                if (pid != INF && cur_p_size >= TASK_SHARE_BOUND) {
                    level_to_share = level;
                }
            }
        }
        else {
            for (int i = get_lane_id(); i < p_size + x_size && min_non_neighbor_p != 0;
                 i += warpSize)
            {
                bool is_p_or_x = true;
                for (int j = 0; j < sizeOfItem; j++) {
                    if (rr_base[j] & ~bitmap_base[i * sizeOfItem + j]) {
                        is_p_or_x = false;
                        break;
                    }
                }
                if (is_p_or_x) min_non_neighbor_p = 0;
            }
            bool is_maximal = __all_sync(FULL_MASK, min_non_neighbor_p == INF);
            if (get_lane_id() == 0 && is_maximal) mc_local++;

            // bool is_maximal = true;
            // for (int i = get_lane_id(); i < p_size + x_size && is_maximal; i += warpSize)
            // {
            //   bitvec_t vec = (bitvec_t)0ULL;
            //   for (int j = 0; j < sizeOfItem && vec == (bitvec_t)0ULL; j++)
            //   {
            //     vec |= (~bitmap_base[i * sizeOfItem + j] & rr_base[j]);
            //   }

            //   if (vec == (bitvec_t)0ULL)
            //   {
            //     is_maximal = false;
            //   }
            // }
            // is_maximal = __all_sync(FULL_MASK, is_maximal);
            // if (get_lane_id() == 0 && is_maximal)
            // {
            //   mc_local++;
            // }
        }

        if (pid == INF) {
            // find next node
            if (get_lane_id() == 0) {
                while (level > (level_sharing + 1) && pid == INF) {
                    level--;
                    bitvec_t* pp_base = rpv + (3 * level + 1) * sizeOfItem;
                    bitvec_t* vpp_base = rpv + (3 * level + 2) * sizeOfItem;
                    for (size_t i = 0; i < sizeOfItem && pid == INF; i++) {
                        bitvec_t res = vpp_base[i] & pp_base[i];
                        if (res != (bitvec_t)0ULL) pid = i * itemPerVec + __ffsll(res) - 1;
                    }
                }
            }

            pid = __shfl_sync(FULL_MASK, pid, 0);
            level = __shfl_sync(FULL_MASK, level, 0);
            level_to_share = min(level_to_share, level - 1);

            if (pid == INF) {
                has_task = false;
                continue;
            }
        }

        // generate new node
        level++;
#ifdef EXPAND_R
        rr_base = rpv + (3 * level - 3) * sizeOfItem;
        pp_base = rpv + (3 * level - 2) * sizeOfItem;
        bitvec_t* rr_next_base = rr_base + 3 * sizeOfItem;
        bitvec_t* pp_next_base = pp_base + 3 * sizeOfItem;
        if (get_lane_id() == 0) {
            cur_p_size = 0;
            pp_base[pid / itemPerVec] &= ~((bitvec_t)1ULL << (pid % itemPerVec));
            for (int i = 0; i < sizeOfItem; i++) {
                rr_next_base[i] = rr_base[i];
                pp_next_base[i] = pp_base[i] & bitmap_base[pid * sizeOfItem + i];
                cur_p_size += __popcll(pp_next_base[i]);
            }
            rr_next_base[pid / itemPerVec] |= (bitvec_t)1ULL << (pid % itemPerVec);
        }
        cur_p_size = __shfl_sync(FULL_MASK, cur_p_size, 0);
        for (int i = get_lane_id(); i < p_size && cur_p_size > 1; i += warpSize) {
            int pos = i / sizeof(bitvec_t) / 8;
            int offset = i % (sizeof(bitvec_t) * 8);
            bool is_p = (pp_next_base[pos] & ((bitvec_t)1ULL << offset)) != 0ULL;

            size_t non_neighbor_p = 0;
            for (int j = 0; is_p && j < sizeOfItem; j++)
                non_neighbor_p += __popcll(pp_next_base[j] & ~bitmap_base[i * sizeOfItem + j]);
            unsigned mask = __ballot_sync(__activemask(), non_neighbor_p == 1);
            if (get_lane_id() == 0 && mask != 0) {
                bitvec_t new_r = (bitvec_t)mask << (i % (sizeof(bitvec_t) * 8));
                rr_next_base[i / sizeof(bitvec_t) / 8] |= new_r;
                pp_next_base[i / sizeof(bitvec_t) / 8] &= ~new_r;
            }
        }
#else
        // no prune
        if (get_lane_id() == 0) {
            bitvec_t* rr_base = rpv + (3 * level - 3) * sizeOfItem;
            bitvec_t* pp_base = rpv + (3 * level - 2) * sizeOfItem;
            bitvec_t* rr_next_base = rr_base + 3 * sizeOfItem;
            bitvec_t* pp_next_base = pp_base + 3 * sizeOfItem;

            pp_base[pid / itemPerVec] &= ~((bitvec_t)1ULL << (pid % itemPerVec));
            for (int i = 0; i < sizeOfItem; i++) {
                rr_next_base[i] = rr_base[i];
                pp_next_base[i] = pp_base[i] & bitmap_base[pid * sizeOfItem + i];
            }
            rr_next_base[pid / itemPerVec] |= (bitvec_t)1ULL << (pid % itemPerVec);
        }
#endif
    }
    return mc_local;
}

__device__ bool GetTaskFromGlobalId(
    GraphGpu& graph,
    ContextManagerBkpb& context,
    size_t& p_size,
    size_t& x_size,
    bitvec_t*& bitmap_base)
{
    const size_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;

    size_t lane_id = get_lane_id();
    vid_t cur_id;
    if (lane_id == 0) cur_id = atomicAdd(context.current_id, 1);
    cur_id = __shfl_sync(FULL_MASK, cur_id, 0);
    vid_t *p_base, *x_base;
#ifndef LEVEL_2
    if (cur_id >= graph.GetNumVertices()) return false;

    cur_id = graph.GetNumVertices() - cur_id - 1;
    vid_t* Nv;
    size_t Nv_size;
    graph.GetNeighbors(cur_id, Nv, Nv_size);
    if (lane_id == 0) x_size = binary_search_first_greater(cur_id, Nv, Nv_size);
    x_size = __shfl_sync(FULL_MASK, x_size, 0);
    x_base = Nv;
    p_size = Nv_size - x_size;
    p_base = Nv + x_size;
#else
    if (cur_id >= graph.GetNumEdges() * 2) return false;
    cur_id = graph.GetNumEdges() * 2 - cur_id - 1;
    // if(lane_id == 0 && cur_id % 100000000 == 0)
    //   printf("cur_id: %d\n", cur_id);
    vid_t *Nv0, *Nv1;
    vid_t v0, v1;
    size_t Nv0_size, Nv1_size;
    size_t Nv_size = 0;
    graph.GetEdge(cur_id, v0, v1);
    if (v0 > v1) {
        p_size = 0;
        x_size = 1;
        return true;
    }
    graph.GetNeighbors(v0, Nv0, Nv0_size);
    graph.GetNeighbors(v1, Nv1, Nv1_size);

    vid_t* rpx_base = (vid_t*)context.buffers_.GetElement(warp_id);

    for (size_t v0_idx = lane_id; v0_idx < Nv0_size; v0_idx += warpSize) {
        vid_t nv0 = Nv0[v0_idx];
        size_t pos = binary_search(nv0, Nv1, Nv1_size);
        size_t match_mask = __ballot_sync(__activemask(), pos != INF);
        if (pos != INF) {
            size_t offset = count_bit(match_mask) - 1 - count_bit((match_mask >> lane_id) - 1);
            size_t pos = Nv_size + offset;
            rpx_base[pos] = nv0;
        }
        Nv_size += __popc(match_mask);
    }
    Nv_size = __shfl_sync(FULL_MASK, Nv_size, 0);

    if (lane_id == 0) {
        x_size = binary_search_first_greater(v1, rpx_base, Nv_size);
    }
    x_size = __shfl_sync(FULL_MASK, x_size, 0);
    x_base = rpx_base;
    p_size = Nv_size - x_size;
    p_base = rpx_base + x_size;
#endif

    if (p_size > 0) {
        size_t sizeOfItem = round_up(p_size, sizeof(bitvec_t) * 8) / sizeof(bitvec_t) / 8;
        size_t bitmap_size = (p_size + x_size) * round_up(p_size, sizeof(bitvec_t) * 8) / 8;

        // assert((x_size + p_size) * sizeof(vid_t) + bitmap_size <
        // context.buffers_.GetElementSize());

        if ((x_size + p_size) * sizeof(vid_t) + bitmap_size >= context.buffers_.GetElementSize()) {
            if (get_lane_id() == 0) {
                printf("x_size: %lu, p_size: %lu\n", x_size, p_size);
            }

            x_size = 1;
            p_size = 0;
            return true;
        }
        else
            bitmap_base = (bitvec_t*)context.buffers_.GetElementRearPtr(warp_id, bitmap_size);

        BuildBitmap(graph, bitmap_base, p_base, p_size, x_base, x_size);

        if (get_lane_id() == 0) {
            bitvec_t* rpv_vec_base = (bitvec_t*)context.buffers_.GetElement(warp_id);
            const size_t itemPerVec = sizeof(bitvec_t) * 8;
            for (int i = 0; i < sizeOfItem; i++) {
                rpv_vec_base[i] = (bitvec_t)0ULL;
                if (i != sizeOfItem - 1 || p_size % itemPerVec == 0)
                    rpv_vec_base[sizeOfItem + i] = rpv_vec_base[sizeOfItem * 2 + i] =
                        (bitvec_t)~0ULL;
                else
                    rpv_vec_base[sizeOfItem + i] = rpv_vec_base[sizeOfItem * 2 + i] =
                        (bitvec_t)~0ULL >> (itemPerVec - p_size % itemPerVec);
            }
        }
    }

    return true;
}

__launch_bounds__(32 * WARP_PER_BLOCK, 1) __global__
    void BkpbKernel(GraphGpu graph, ContextManagerBkpb context)
{
    acc_t mc_local = 0;
    const size_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const size_t num_warps = gridDim.x * blockDim.x / warpSize;
    size_t lane_id = get_lane_id();

    vid_t* rpx_base = (vid_t*)context.buffers_.GetElement(warp_id);
    Metadata* meta_base = (Metadata*)context.metas_.GetElement(warp_id);
    bitvec_t* bitmap_base; // = (bitvec_t *)context.bitmaps_.GetElement(warp_id);
    bitvec_t* rpv_vec_base = (bitvec_t*)context.buffers_.GetElement(warp_id);
    size_t p_size, x_size;

    uint64_t start_time = clock64();
    if (lane_id == 0) {
        atomicAdd(context.active_warps_, 1);
        meta_base->pending_count = 0;
        __threadfence();
    }
    // stage 1
    while (true) {
        if (!GetTaskFromGlobalId(graph, context, p_size, x_size, bitmap_base)) break;
        if (p_size == 0) {
            if (x_size == 0 && lane_id == 0) mc_local++;
        }
        else
            mc_local += IterKernelAuto(context, bitmap_base, p_size, x_size);
    }

    if (lane_id == 0) {
        auto num = atomicSub(context.active_warps_, 1);
        // printf("active warp rate: %lf | active warps: %lu\n", (double)num / num_warps, num);
    }

    // stage 2
    vid_t cur_warp_id = warp_id;

    while (true) {
        cur_warp_id = (cur_warp_id + 1) % num_warps;
        Metadata* meta = (Metadata*)context.metas_.GetElement(cur_warp_id);
        bool has_task = false;
        if (lane_id == 0) {
            auto task_count = meta->task_count;
            if (task_count != 0) {
                auto old_value = atomicCAS(&meta->task_count, task_count, task_count - 1);
                if (old_value == task_count) { // get a task
                    has_task = true;
                    SetTaskById(
                        meta->bitmap_base, meta->p_size, meta->rpv_base, rpv_vec_base, task_count);
                    bitmap_base = meta->bitmap_base;
                    p_size = meta->p_size;
                    x_size = meta->x_size;
                    __threadfence();
                    atomicSub(&meta->pending_count, 1);
                    atomicAdd(context.active_warps_, 1);
                }
            }
        }
        has_task = __shfl_sync(FULL_MASK, has_task, 0);
        if (has_task) {
            p_size = __shfl_sync(FULL_MASK, p_size, 0);
            x_size = __shfl_sync(FULL_MASK, x_size, 0);
            bitmap_base = (bitvec_t*)__shfl_sync(FULL_MASK, (uint64_t)bitmap_base, 0);
            mc_local += IterKernelAuto(context, bitmap_base, p_size, x_size);
            if (lane_id == 0) {
                auto num = atomicSub(context.active_warps_, 1);
                // cuda_sleep();
                // printf("active warp rate: %lf | active warps: %lu\n", (double)num / num_warps,
                // num);
            }
            cur_warp_id--;
        }
        else
            cuda_sleep();

        bool exit_flag;
        if (lane_id == 0) {

            __threadfence();
            // printf("%d " ,*context.active_warps_);
            exit_flag = *context.active_warps_ == 0;
        }
        exit_flag = __shfl_sync(FULL_MASK, exit_flag, 0);
        if (exit_flag) break;
    }

    if (lane_id == 0 && mc_local != 0) {
        acc_t res = atomicAdd((unsigned long long*)context.mc_num_, mc_local);
        // printf("mc_num: %llu\n", res);
    }
    // if(threadIdx.x == 0 && blockIdx.x == 0)
    //   printf("total shared tasks size: %d\n", *context.debug_val);
}

#define val auto const
#define var auto
// The number of threads per block should be not greater than 1024, and be a multiple of 32.
// Only threads with threadIdx.x < 32 can obtain the summation.
template <typename Addable>
    requires(std::integral<Addable> || std::floating_point<Addable>) && (sizeof(Addable) >= 4)
__forceinline__ __device__ Addable Sum(Addable a)
{
    a += __shfl_xor_sync(0xffff'ffff, a, 1);
    a += __shfl_xor_sync(0xffff'ffff, a, 2);
    a += __shfl_xor_sync(0xffff'ffff, a, 4);
    a += __shfl_xor_sync(0xffff'ffff, a, 8);
    a += __shfl_xor_sync(0xffff'ffff, a, 16);
    __shared__ Addable t[32];
    val WID = threadIdx.x >> 5;
    if ((threadIdx.x & 0x1f) == 0) t[WID] = a;
    __syncthreads();
    if (WID == 0) {
        a = threadIdx.x < ((blockDim.x + 31) >> 5) ? t[threadIdx.x] : static_cast<Addable>(0);
        a += __shfl_xor_sync(0xffff'ffff, a, 1);
        a += __shfl_xor_sync(0xffff'ffff, a, 2);
        a += __shfl_xor_sync(0xffff'ffff, a, 4);
        a += __shfl_xor_sync(0xffff'ffff, a, 8);
        a += __shfl_xor_sync(0xffff'ffff, a, 16);
    }
    __syncthreads(); // In case that this kernel is successively invoked, ...
    return a;
}

__global__ void Peel0(GraphGpu const g, auto* counter, auto* survival, auto* degree)
{
    auto const TID = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t cnt = 0;
    for (uint32_t i = TID; i < g.num_vertices_; i += blockDim.x * gridDim.x) {
        if (degree[i] == 0) {
            ++cnt;
            survival[i] = 0;
        }
    }
    auto t = Sum(cnt);
    if (threadIdx.x == 0) atomicAdd(counter, t);
}

__global__ void SieveD1(GraphGpu const g, auto* degree, auto* next, auto* cursor)
{
    val TID = blockDim.x * blockIdx.x + threadIdx.x;
    val LID = TID & 0x1f;
    val STRIDE = blockDim.x * gridDim.x;
    val END = ((g.num_vertices_ + 31) >> 5) << 5;
    for (var i = TID; i < END; i += STRIDE) {
        val hit = i < g.num_vertices_ && degree[i] == 1;
        val mask = __ballot_sync(0xffff'ffffU, hit);
        val offset = __popc(mask & ((1U << LID) - 1));
        uint32_t cur;
        if (LID == 0) cur = atomicAdd(cursor, __popc(mask));
        cur = __shfl_sync(0xffff'ffffU, cur, 0);
        if (hit) next[cur + offset] = i;
    }
}

__global__ void Peel1_v2(
    GraphGpu const g,
    auto* counter,
    auto* survival,
    auto* degree,
    auto const* __restrict__ dying,
    auto const* __restrict__ dying_count,
    auto* next,
    auto* cursor)
{
    val TID = blockDim.x * blockIdx.x + threadIdx.x;
    val WID = TID >> 5;
    val LID = TID & 0x1f;
    val STRIDE = blockDim.x * gridDim.x;
    val END = ((*dying_count + 31) >> 5) << 5;
    uint32_t cnt = 0; // The decrease of the summation of degrees
    for (var i = TID; i < END; i += STRIDE) {
        var hit = 0xffff'ffffU;
        if (i < *dying_count) {
            val VID = dying[i];
            // TODO: Could it lead to negative degrees?
            if (atomicSub(degree + VID, 1) > 0) cnt += 1;
            survival[VID] = 0;
            for (var j = __ldg(g.rowoffset_ + VID), e = __ldg(g.rowoffset_ + VID + 1); j < e; ++j) {
                val nghb = __ldg(g.colidx_[j]);
                if (survival[nghb]) {
                    val fd = atomicSub(degree + nghb, 1);
                    if (fd > 0) cnt += 1;
                    if (fd == 2) {
                        hit = nghb;
                    }
                    else if (fd <= 1) {
                        survival[nghb] = 0;
                    }
                    // TODO: Could it lead to negative degrees?
                    if (fd < 0) printf("[GKP] NEGATIVE DEGREE!\n");
                }
            }
        }
        val writing = hit < 0xffff'ffffU;
        var mask = __ballot_sync(0xffff'ffffU, writing);
        val offset = __popc(mask & ((1U << LID) - 1));
        uint32_t cur;
        if (LID == 0) cur = atomicAdd(cursor, __popc(mask));
        cur = __shfl_sync(0xffff'ffffU, cur, 0);
        if (writing) next[cur + offset] = hit;
    }
    // TODO: Do it by a finalizing function, to avoid negative degrees.
    cnt = Sum(cnt);
    if (threadIdx.x == 0) atomicAdd(counter + 1, cnt);
}

__global__ void Sieve1_v3(
    GraphGpu const g, auto* flag, auto const* __restrict__ degree, auto* next, auto* cursor)
{
    val TID = blockDim.x * blockIdx.x + threadIdx.x;
    val LID = TID & 0x1f;
    val STRIDE = blockDim.x * gridDim.x;
    val END = ((g.num_vertices_ + 31) >> 5) << 5;
    for (var i = TID; i < END; i += STRIDE) {
        val deg = degree[i];
        val hit = i < g.num_vertices_ && deg < 3;
        uint32_t mask = __ballot_sync(0xffff'ffffU, hit);
        uint32_t offset = __popc(mask & ((1U << LID) - 1));
        flag[i] = (deg == 2) << 2 | (deg == 1) << 1 | 1;
        uint32_t cur = LID == 0 ? atomicAdd(cursor, __popc(mask)) : 0U;
        cur = __shfl_sync(0xffff'ffffU, cur, 0);
        if (hit) next[cur + offset] = i;
    }
}

__global__ void Sieve_v3(GraphGpu const g, auto* degree, auto* next, auto* cursor)
{
    val TID = blockDim.x * blockIdx.x + threadIdx.x;
    val LID = TID & 0x1f;
    val STRIDE = blockDim.x * gridDim.x;
    val END = ((g.num_vertices_ + 31) >> 5) << 5;
    for (var i = TID; i < END; i += STRIDE) {
        val hit = i < g.num_vertices_ && degree[i] < 3;
        val mask = __ballot_sync(0xffff'ffffU, hit);
        val offset = __popc(mask & ((1U << LID) - 1));
    }
}

__global__ void SieveD2(GraphGpu const g, auto* degree, auto* next, auto* cursor)
{
    val TID = blockDim.x * blockIdx.x + threadIdx.x;
    val LID = TID & 0x1f;
    val STRIDE = blockDim.x * gridDim.x;
    val END = ((g.num_vertices_ + 31) >> 5) << 5;
    for (var i = TID; i < END; i += STRIDE) {
        val hit = i < g.num_vertices_ && degree[i] == 2;
        val mask = __ballot_sync(0xffff'ffffU, hit);
        val offset = __popc(mask & ((1U << LID) - 1));
        uint32_t cur;
        if (LID == 0) cur = atomicAdd(cursor, __popc(mask));
        cur = __shfl_sync(0xffff'ffffU, cur, 0);
        if (hit) next[cur + offset] = i;
    }
}

__forceinline__ __device__ bool LockTriangle(auto* flag, uint32_t v1, uint32_t v2, uint32_t v3)
{
    if (v2 > v3) thrust::swap(v2, v3);
    val e1 = flag[v1];
    if ((e1 & 4) || atomicCAS(flag + v1, e1, e1 | 4U) != e1) return false;
    val e2 = flag[v2];
    if ((e2 & 4) || atomicCAS(flag + v2, e2, e2 | 4U) != e2) {
        atomicAnd(flag + v1, ~4U);
        return false;
    }
    val e3 = flag[v3];
    if ((e3 & 4) || atomicCAS(flag + v3, e3, e3 | 4U) != e3) {
        atomicAnd(flag + v1, ~4U);
        atomicAnd(flag + v2, ~4U);
        return false;
    }
    return true;
}

__forceinline__ __device__ bool NotTriangle(GraphGpu const& g, uint32_t nghb1, uint32_t nghb2)
{
    for (var i = g.rowoffset_[nghb1]; i < g.rowoffset_[nghb1 + 1]; ++i) {
        if (nghb2 == g.colidx_[i]) return false;
    }
    return true;
}

__global__ void Peel2(
    GraphGpu const g,
    auto* counter,
    auto* flag,
    auto* degree,
    auto const* dying,
    auto const* dying_count,
    auto* next,
    auto* cursor)
{
#define INVALID_VID 0xffff'ffffU
    val TID = blockDim.x * blockIdx.x + threadIdx.x;
    val WID = TID >> 5;
    val LID = TID & 0x1f;
    val STRIDE = blockDim.x * gridDim.x;
    val END = ((*dying_count + 31) >> 5) << 5;
    // cnt2: The decrease of the summation of degrees caused by **removing MAXIMAL 2-cliques**.
    // |Removed MAXIMAL 2-cliques| = cnt2 / 2
    // cnt3: How many 3-cliques reduced.
    uint32_t cnt2 = 0, cnt3 = 0;
    // flag: 1 = survival, 2 = non-triangle, 4 = locked, 8 = with the smallest index among a
    // triangle, 16 = in frontier The first time a vertex put into frontier: identify whether it
    // forms a triangle, update flag, and put it into frontier again.
    for (var i = TID; i < END; i += STRIDE) {
        uint32_t nghb1 = INVALID_VID, nghb2 = INVALID_VID;
        if (i < *dying_count) {
            val VID = dying[i];
            for (var j = g.rowoffset_[VID], e = g.rowoffset_[VID + 1]; j < e; ++j) {
                var u = g.colidx_[j];
                if (flag[u]) {
                    nghb2 = nghb1;
                    nghb1 = u;
                }
            }
            if (flag[VID] == 29) { // 29 = 16 + 8 + 4 + 1
                flag[VID] = 0;
                // It **must** be of deg 2, for only vertices of degree 2 at most are put into
                // frontier.
                if (atomicSub(degree + VID, 2) == 2) {
                    cnt3 += 1;
                    if (nghb1 != INVALID_VID) {
                        val fd = atomicSub(degree + nghb1, 1);
                        if (fd != 2 && fd != 3) nghb1 = INVALID_VID;
                    }
                    if (nghb2 != INVALID_VID) {
                        val fd = atomicSub(degree + nghb2, 1);
                        if (fd != 2 && fd != 3) nghb2 = INVALID_VID;
                    }
                }
            }
            else if (flag[VID] & 4) {
                // Do nothing
                nghb1 = nghb2 = INVALID_VID;
            }
            else if (flag[VID] == 19) { // 19 = 16 + 2 + 1
                flag[VID] = 0;
                var fd = atomicSub(degree + VID, 2);
                cnt2 += max(fd, 0);
                if (nghb1 != INVALID_VID) {
                    fd = atomicSub(degree + nghb1, 1);
                    if (fd < 2) {
                        degree[nghb1] = 0;
                        flag[nghb1] = 0;
                    }
                    if (fd > 0) cnt2 += 1;
                    if (fd != 3) nghb1 = INVALID_VID;
                }
                if (nghb2 != INVALID_VID) {
                    fd = atomicSub(degree + nghb2, 1);
                    if (fd < 2) {
                        degree[nghb2] = 0;
                        flag[nghb2] = 0;
                    }
                    if (fd > 0) cnt2 += 1;
                    if (fd != 3) nghb2 = INVALID_VID;
                }
            }
            else if (flag[VID] & 1) { // Let us find out whether d-2 vertex form a triangle or not.
                if (degree[VID] < 0) {
                    nghb1 = nghb2 = INVALID_VID;
                    degree[VID] = 0;
                    flag[VID] = 0;
                }
                else if (degree[VID] == 1) { // None of 2-cliques removed hereby is maximal clique.
                    nghb1 = nghb2 = INVALID_VID;
                    degree[VID] = 0;
                    flag[VID] = 0;
                    for (var j = g.rowoffset_[VID], e = g.rowoffset_[VID + 1]; j < e; ++j) {
                        var u = g.colidx_[j];
                        if (flag[u]) {
                            val fd = atomicSub(degree + u, 1);
                            if (fd == 2 || fd == 3) nghb1 = u;
                        }
                    }
                }
                else if (degree[VID] == 2) {
                    if (nghb1 == INVALID_VID || nghb2 == INVALID_VID ||
                        NotTriangle(g, nghb1, nghb2))
                    {
                        flag[VID] |= 2;
                        nghb1 = VID;
                        nghb2 = INVALID_VID;
                    }
                    else if (VID < nghb1 && VID < nghb2 && LockTriangle(flag, VID, nghb1, nghb2)) {
                        flag[VID] |= 8;
                        nghb1 = VID;
                        nghb2 = INVALID_VID;
                    }
                    else {
                        nghb1 = nghb2 = INVALID_VID;
                    }
                }
            }
            flag[VID] &= ~16U;
        }
        {
            var writing = nghb1 < INVALID_VID && !(atomicOr(flag + nghb1, 16) & 16);
            var mask = __ballot_sync(0xffff'ffffU, writing);
            var off = __popc(mask & ((1U << LID) - 1));
            uint32_t cur;
            if (LID == 0) cur = atomicAdd(cursor, __popc(mask));
            cur = __shfl_sync(0xffff'ffffU, cur, 0);
            if (writing) next[cur + off] = nghb1;
        }
        {
            var writing = nghb2 < INVALID_VID && !(atomicOr(flag + nghb2, 16) & 16);
            var mask = __ballot_sync(0xffff'ffffU, writing);
            var off = __popc(mask & ((1U << LID) - 1));
            uint32_t cur;
            if (LID == 0) cur = atomicAdd(cursor, __popc(mask));
            cur = __shfl_sync(0xffff'ffffU, cur, 0);
            if (writing) next[cur + off] = nghb2;
        }
    }
    cnt2 = Sum(cnt2);
    if (threadIdx.x == 0) atomicAdd(counter + 1, cnt2);
    cnt3 = Sum(cnt3);
    if (threadIdx.x == 0) atomicAdd(counter + 2, cnt3);
#undef INVALID_VID
}

__global__ void FinalizePeeling(auto const vertex_count, auto* degree, auto* survival)
{
    val TID = blockDim.x * blockIdx.x + threadIdx.x;
    val STRIDE = blockDim.x * gridDim.x;
    for (var i = TID; i < vertex_count; i += STRIDE) {
        if (degree[i] < 0) {
            LOG("degree[", i, "] = ", degree[i]);
            degree[i] = 0;
            survival[i] = 0;
        }
        if (survival[i] == 0 && degree[i] != 0) {
            LOG("survival[", i, "] = 0 && degree[", i, "] =", degree[i]);
            degree[i] = 0;
        }
        if (survival[i] != 0 && degree[i] == 0) {
            LOG("survival[", i, "] = ", survival[i], " && degree[", i, "] = 0");
            degree[i] = 0;
        }
        if (survival[i] > 1) {
            LOG("survival[", i, "] = ", survival[i]);
            survival[i] = 1;
        }
    }
}

__global__ void Rebuild(
    GraphGpu const orig_g,
    GraphGpu g,
    auto const* __restrict__ degree,
    auto const* __restrict__ survival,
    auto const* __restrict__ new_vid,
    auto const* __restrict__ offset_offset)
{
    auto const TID = blockIdx.x * blockDim.x + threadIdx.x;
    auto const WID = TID >> 5;
    auto const LID = TID & 0x1f;
    auto const* __restrict__ rowoffset_ = orig_g.rowoffset_;
    auto const* __restrict__ colidx_ = orig_g.colidx_;
    if (TID == 0)
        g.rowoffset_[g.num_vertices_] =
            degree[orig_g.num_vertices_ - 1] + offset_offset[orig_g.num_vertices_ - 1];
    auto const STRIDE = gridDim.x * blockDim.x >> 5;
    for (int i = WID; i < orig_g.num_vertices_; i += STRIDE) {
        if (survival[i]) {
            if (LID == 0) g.rowoffset_[new_vid[i]] = offset_offset[i];
            auto cur = offset_offset[i];
            auto end = rowoffset_[i + 1];
            for (auto j = rowoffset_[i]; j < end; j += 32) {
                auto k = j + LID;
                auto u = k < end ? colidx_[k] : 0;
                auto active = k < end && survival[u];
                auto nvid = active ? new_vid[u] : 0;
                auto mask = __ballot_sync(0xffff'ffff, active);
                auto pos = __popc(mask & ((1U << LID) - 1));
                if (active) g.colidx_[cur + pos] = nvid;
                cur += __popc(mask);
            }
        }
    }
}
#undef var
#undef val

// 5. User defined solver wrapper
acc_t BkSolverWrapper(Graph& graph, size_t device_id)
{

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    size_t sm_num = prop.multiProcessorCount;
    printf(
        "\033[90m[GPU] SM: %lu  \tWARP_PER_BLOCK: %d\tBLOCK_PER_SM: %d\033[0m\n", sm_num,
        WARP_PER_BLOCK, BLOCK_PER_SM);
    printf("\033[90m[GPU] TASK_SHARE_BOUND: %d\033[0m\n", TASK_SHARE_BOUND);

    GraphGpu graph_gpu;
    ContextManagerBkpb context_gpu;

    graph_gpu.LoadFromCpu(graph);

    size_t num_warps = WARP_PER_SM * sm_num;
    size_t bitvec_size = round_up(graph.GetDegeneracy(), 8 * sizeof(bitvec_t)) / 8;

    size_t bitmap_size = round_up(graph.GetMaxDegree() * (bitvec_size + 2 * sizeof(vid_t)), 8);
    size_t rpxv_size = graph.GetDegeneracy() * bitvec_size * 3;

    size_t free_byte;
    size_t total_byte;
    CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte));
    context_gpu.metas_.Allocate(num_warps, sizeof(Metadata));

    size_t max_available_size = round_up((size_t)(free_byte * 0.95 / num_warps), 8);
    size_t max_usage = bitmap_size + rpxv_size;
    size_t total_size = min(max_usage, max_available_size);

    printf(
        "\033[90m[GPU] Context memory usage: %lf MB\033[0m\n",
        total_size * num_warps / 1024.0 / 1024.0);
    printf(
        "      \033[90m- Bitmap: %lf MB\n", (total_size - rpxv_size) * num_warps / 1024.0 / 1024.0);
    printf("      - Rpx buffer: %lf MB\033[0m\n", rpxv_size * num_warps / 1024.0 / 1024.0);

    if (max_usage < max_available_size)
        context_gpu.buffers_.Allocate(num_warps, max_usage);
    else {
        printf(
            "\033[90m[GPU] Context memory usage exceeds 95%% of available memory on GPUs, using "
            "unified memory instead\033[0m\n");
        context_gpu.buffers_.AllocateManaged(num_warps, max_usage);
    }

    CUDA_CHECK(cudaMalloc((void**)&context_gpu.mc_num_, sizeof(acc_t)));
    CUDA_CHECK(cudaMemset(context_gpu.mc_num_, 0, sizeof(acc_t)));
    CUDA_CHECK(cudaMalloc((void**)&context_gpu.current_id, sizeof(vid_t)));
    CUDA_CHECK(cudaMemset(context_gpu.current_id, 0, sizeof(vid_t)));
    CUDA_CHECK(cudaMalloc((void**)&context_gpu.active_warps_, sizeof(vid_t)));
    CUDA_CHECK(cudaMemset(context_gpu.active_warps_, 0, sizeof(vid_t)));

    // CUDA_CHECK(cudaMalloc((void **)&context_gpu.debug_val, sizeof(vid_t)));
    // CUDA_CHECK(cudaMemset(context_gpu.debug_val, 0, sizeof(vid_t)));

    auto t1 = std::chrono::high_resolution_clock::now();
    GraphGpu reduced_graph_gpu = graph_gpu;
    reduced_graph_gpu.degree_ = nullptr;
    uint32_t* survival;
    uint32_t* counter;
    uint32_t counter_h[8] { 0 };
    uint32_t* new_vid;
    int64_t* offset_offset;
    int64_t* new_degree;
    cudaMalloc(&survival, graph_gpu.num_vertices_ * sizeof(uint32_t));
    cudaMalloc(&new_vid, graph_gpu.num_vertices_ * sizeof(uint32_t));
    cudaMalloc(&offset_offset, graph_gpu.num_vertices_ * sizeof(*offset_offset));
    cudaMalloc(&new_degree, graph_gpu.num_vertices_ * sizeof(*new_degree));
    cudaMalloc(&counter, 8 * sizeof(uint32_t));
    cudaMemset(survival, 0, graph_gpu.num_vertices_ * sizeof(uint32_t));
    cudaMemset(new_vid, 0, graph_gpu.num_vertices_ * sizeof(uint32_t));
    cudaMemset(offset_offset, 0, graph_gpu.num_vertices_ * sizeof(*offset_offset));
    cudaMemset(new_degree, 0, graph_gpu.num_vertices_ * sizeof(*new_degree));
    cudaMemset(counter, 0, 8 * sizeof(uint32_t));
    CUDA_CHECK(cudaDeviceSynchronize());

    {
        std::vector<int32_t> temp(graph_gpu.num_vertices_, 1);
        cudaMemcpy(
            survival, temp.data(), graph_gpu.num_vertices_ * sizeof(int32_t),
            cudaMemcpyHostToDevice);
    }
    void* temp_storage = nullptr;
    size_t temp_storage_size = 0;

    cudaStream_t streams[3];
    for (int i = 0; i < 3; ++i) {
        cudaStreamCreateWithFlags(streams + i, cudaStreamNonBlocking);
    }
    gkp::pt<<<sm_num * BLOCK_PER_SM, 32 * 4, 0, streams[0]>>>(
        graph_gpu, graph_gpu.degree_, new_degree, survival, counter);
    gkp::pw<<<sm_num, 32 * 4, 0, streams[1]>>>(
        graph_gpu, graph_gpu.degree_, new_degree, survival, counter + 1);
    gkp::pb<<<1, 1024, 0, streams[2]>>>(
        graph_gpu, graph_gpu.degree_, new_degree, survival, counter + 2);
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < 3; ++i) {
        cudaStreamDestroy(streams[i]);
    }
    cudaMemcpy(counter_h, counter, 8 * sizeof(*counter_h), cudaMemcpyDeviceToHost);
    LOG("counter_h[0] = ", counter_h[0]);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        nullptr, temp_storage_size, new_degree, offset_offset, graph_gpu.num_vertices_));
    cudaMalloc(&temp_storage, temp_storage_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        temp_storage, temp_storage_size, new_degree, offset_offset, graph_gpu.num_vertices_));
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
        temp_storage, temp_storage_size, survival, new_vid, graph_gpu.num_vertices_));
    CUDA_CHECK(cudaGetLastError());
    cudaFree(temp_storage);
    uint32_t a, b;
    cudaMemcpy(&a, new_vid + graph_gpu.num_vertices_ - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(
        &b, survival + graph_gpu.num_vertices_ - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    LOG("New ids generated. a = ", a, " b = ", b);
    reduced_graph_gpu.num_vertices_ = static_cast<size_t>(a) + b;
    cudaMemcpy(
        &a, offset_offset + graph_gpu.num_vertices_ - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(
        &b, new_degree + graph_gpu.num_vertices_ - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    LOG("New degrees generated. a = ", a, " b = ", b);
    reduced_graph_gpu.num_edges_ = (static_cast<size_t>(a) + b) >> 1;
    cudaMalloc(
        &reduced_graph_gpu.rowoffset_, (reduced_graph_gpu.num_vertices_ + 1) * sizeof(vid_t));
    cudaMalloc(&reduced_graph_gpu.colidx_, (reduced_graph_gpu.num_edges_ << 1) * sizeof(vid_t));

    Rebuild<<<sm_num * BLOCK_PER_SM, 32 * WARP_PER_BLOCK>>>(
        graph_gpu, reduced_graph_gpu, new_degree, survival, new_vid, offset_offset);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(counter);
    cudaFree(new_degree);
    cudaFree(offset_offset);
    cudaFree(new_vid);
    cudaFree(survival);
    // graph_gpu.Free();
    // graph_gpu = reduced_graph_gpu;
    auto t2 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    printf("[GKP] time: %0.3lf ms\n", ms);

    BkpbKernel<<<sm_num * BLOCK_PER_SM, 32 * WARP_PER_BLOCK>>>(reduced_graph_gpu, context_gpu);
    // BkpbKernel<<<1, 32>>>(graph_gpu, context_gpu);

    CUDA_CHECK(cudaDeviceSynchronize());
    auto mc_num = context_gpu.GetMcNum() + counter_h[0] + counter_h[1] + counter_h[2];
    // free memory
    graph_gpu.Free();
    reduced_graph_gpu.Free();
    // context_gpu.bitmaps_.Free();
    // context_gpu.rpxv_.Free();
    context_gpu.buffers_.Free();
    context_gpu.metas_.Free();

    CUDA_CHECK(cudaFree(context_gpu.mc_num_));
    CUDA_CHECK(cudaFree(context_gpu.current_id));
    CUDA_CHECK(cudaFree(context_gpu.active_warps_));
    // CUDA_CHECK(cudaFree(context_gpu.debug_val));
    return mc_num;
}
