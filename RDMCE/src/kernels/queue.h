#pragma once
#ifndef GKP_QUEUE
#define GKP_QUEUE

#include "cuda.h"
#include "cuda_runtime.h"
#include "gkp.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <type_traits>

namespace gkp
{

#define val auto const
#define var auto
template <typename T>
struct Queue
{
    T* const base;
    // TODO: Do I really need circular queue? Or, I can set capacity large enough, can't I?
    int32_t const capacity;
    volatile int32_t* const wptr;
    volatile int32_t* const rptr;

    // Invoker should allocate, initialize and free memory.
    __host__ Queue(T* const base, int32_t const capacity, int32_t* const wptr, int32_t* const rptr)
        : base(base), capacity(capacity), wptr(wptr), rptr(rptr)
    {}

    __host__ ~Queue()
    {}

    __forceinline__ __device__ ssize_t size(void) const
    {
        return *wptr - *rptr;
    }

    __forceinline__ __device__ T& operator[](std::integral auto x)
    {
#if DEBUG
        if (x >= size() || static_cast<int64_t>(x) & (1ULL << 63)) {
            LOG("Out of range x = ", x);
            exit(__LINE__);
        }
#endif // DEBUG
        return base[x];
    }

    __forceinline__ __device__ T operator[](std::integral auto x) const
        requires(std::is_arithmetic_v<T>)
    {
#if DEBUG
        if (x >= size() || static_cast<int64_t>(x) & (1ULL << 63)) {
            LOG("Out of range x = ", x);
            exit(__LINE__);
        }
#endif // DEBUG
        return base[x];
    }

    __forceinline__ __device__ T const& operator[](std::integral auto x) const
        requires(!std::is_arithmetic_v<T>)
    {
#if DEBUG
        if (x >= size() || static_cast<int64_t>(x) & (1ULL << 63)) {
            LOG("Out of range x = ", x);
            exit(__LINE__);
        }
#endif // DEBUG
        return base[x];
    }

    template <typename U>
    __forceinline__ __device__ void addWarpwise(U&& x, bool const cond)
        requires(std::is_same_v<std::remove_cvref_t<U>, T>) // Without calculating leader thread
    {
#if DEBUG
        if (__activemask() != 0xffff'ffffU) {
            LOG("Divergent Deadlock mask = 0xffff'ffff",
                " activemask = ", std::format("%x", __activemask()));
            exit(__LINE__);
        }
#endif // DEBUG
        val hit = __ballot_sync(0xffff'ffffU, cond);
        if (hit == 0) return;
        val off = __popc(hit & ((1U << get_lane_id()) - 1U));
        int32_t pos = get_lane_id() == 0 ? atomicAdd(wptr, __popc(hit)) : 0;
        pos = __shfl_sync(0xffff'ffffU, pos, 0);
#if DEBUG
        if (pos + off >= capacity) {
            LOG("Overflow\capacity = ", capacity);
            exit(__LINE__);
        }
#endif // DEBUG
        if (cond) base[pos + off] = std::forward<U>(x);
    }

    template <typename U>
    __forceinline__ __device__ void addWarpwise(U&& x, bool const cond, uint32_t const mask)
        requires(std::is_same_v<std::remove_cvref_t<U>, T>)
    {
#if DEBUG
        if ((mask & __activemask()) != mask) {
            LOG("Divergent Deadlock mask = ", mask,
                " activemask = ", std::format("%x", __activemask()));
            exit(__LINE__);
        }
#endif // DEBUG
        val hit = __ballot_sync(mask, cond);
        if (hit == 0) return;
        val off = __popc(hit & ((1U << get_lane_id()) - 1U));
        val leader = __ffs(mask) - 1;
        int32_t pos = get_lane_id() == leader ? atomicAdd(wptr, __popc(hit)) : 0;
        pos = __shfl_sync(mask, pos, leader);
#if DEBUG
        if (pos + off >= capacity) {
            LOG("Overflow capacity = ", capacity);
            exit(__LINE__);
        }
#endif // DEBUG
        if (cond) base[pos + off] = std::forward<U>(x);
    }

    template <typename U>
        requires requires(T t, U u) { u = t; }
    __forceinline__ __device__ int32_t drainBlockwise(U& out)
    {
#if DEBUG
        if (blockDim.y > 1 || blockDim.z > 1) {
            LOG("Incompatible configuration blockDim = {", blockDim.x, ", ", blockDim.y, ", ",
                blockDim.z, "}");
            exit(__LINE__);
        }
#endif // DEBUG
        __shared__ int32_t cnt;
        __shared__ int32_t pos;
        while (threadIdx.x == 0) {
            cnt = std::min(blockDim.x, *wptr - *rptr);
            if (cnt == 0) break;
            pos = *rptr;
            val t = atomicCAS(rptr, pos, pos + cnt);
            if (t == pos) break;
        }
        __syncthreads();
        if (cnt == 0) return 0;
        if (threadIdx.x < cnt) out = std::move(base[pos + threadIdx.x]);
        return cnt;
    }
};

#undef var
#undef val

} // namespace gkp

#endif // GKP_QUEUE
