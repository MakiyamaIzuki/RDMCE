#pragma once
#ifndef GKP_QUEUE
#define GKP_QUEUE

#include "gkp.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cuda.h>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <type_traits>

namespace gkp
{

#define val auto const
#define var auto
template <typename T>
struct Queue
{
    static Queue<T> Generate(int32_t capacity)
    {
        T* b;
        int32_t *w, *r;
        CUDA_INVOKE(cudaMalloc, &b, capacity * sizeof(T));
        CUDA_INVOKE(cudaMalloc, &w, sizeof(int32_t));
        CUDA_INVOKE(cudaMemset, w, 0, sizeof(int32_t));
        CUDA_INVOKE(cudaMalloc, &r, sizeof(int32_t));
        CUDA_INVOKE(cudaMemset, r, 0, sizeof(int32_t));
        LOG("Constructing queue. base = ", b);
        return Queue(b, capacity, w, r);
    }

    static void Free(Queue<T> q)
    {
        CUDA_INVOKE(cudaFree, q.base);
        CUDA_INVOKE(cudaFree, q.wptr);
        CUDA_INVOKE(cudaFree, q.rptr);
    }

    T* base;
    // TODO: Do I really need circular queue? Or, I can set capacity large enough,
    // can't I?
    int32_t capacity;
    int32_t* wptr;
    int32_t* rptr;

    // Invoker should allocate, initialize and free memory.
    __host__ Queue(T* const base, int32_t const capacity, int32_t* const wptr, int32_t* const rptr)
        : base(base), capacity(capacity), wptr(wptr), rptr(rptr)
    {}

    Queue(Queue const&) = default;

    __host__ Queue(Queue&& rhs)
    {
        base = rhs.base;
        rhs.base = nullptr;
        capacity = rhs.capacity;
        rhs.capacity = 0;
        wptr = rhs.wptr;
        rhs.wptr = nullptr;
        rptr = rhs.rptr;
        rhs.rptr = nullptr;
    }

    __host__ auto& operator=(Queue&& rhs)
    {
        base = rhs.base;
        rhs.base = nullptr;
        capacity = rhs.capacity;
        rhs.capacity = 0;
        wptr = rhs.wptr;
        rhs.wptr = nullptr;
        rptr = rhs.rptr;
        rhs.rptr = nullptr;
        return *this;
    }

    __host__ ~Queue()
    {}

#if __CUDA_ARCH__
    __forceinline__ __device__ void clear(void)
    {
        atomicExch(wptr, 0);
        atomicExch(rptr, 0);
    }
#else
    inline void clear(void)
    {
        // TODO: Use async function
        CUDA_INVOKE(cudaMemset, wptr, 0, sizeof(*wptr));
        CUDA_INVOKE(cudaMemset, rptr, 0, sizeof(*wptr));
    }
#endif

#if __CUDA_ARCH__
    __forceinline__ __device__ ssize_t size(void) const
    {
        return *wptr - *rptr;
    }
#else
    inline ssize_t size(void) const
    {
        int32_t buf[2];
        CUDA_INVOKE(cudaMemcpy, buf, wptr, sizeof(int32_t), cudaMemcpyDeviceToHost);
        CUDA_INVOKE(cudaMemcpy, buf + 1, rptr, sizeof(int32_t), cudaMemcpyDeviceToHost);
        return buf[0] - buf[1];
    }
#endif

    __forceinline__ __device__ T& operator[](std::integral auto x)
    {
#if DEBUG
        if (x >= size() || static_cast<int64_t>(x) & (1ULL << 63)) {
            LOG("Out of range. base = ", base, " x = ", x, " size = ", size());
            assert(0);
        }
#endif // DEBUG
        return base[x];
    }

    __forceinline__ __device__ T operator[](std::integral auto x) const
        requires(std::is_arithmetic_v<T>)
    {
#if DEBUG
        if (x >= size() || static_cast<int64_t>(x) & (1ULL << 63)) {
            LOG("Out of range. base = ", base, " x = ", x, " size = ", size());
            assert(0);
        }
#endif // DEBUG
        if constexpr (sizeof(T) >= 4 && sizeof(T) <= 8) {
            return __ldg(base + x);
        }
        return base[x];
    }

    __forceinline__ __device__ T const& operator[](std::integral auto x) const
        requires(!std::is_arithmetic_v<T>)
    {
#if DEBUG
        if (x >= size() || static_cast<int64_t>(x) & (1ULL << 63)) {
            LOG("Out of range. base = ", base, " x = ", x, " size = ", size());
            assert(0);
        }
#endif // DEBUG
        return base[x];
    }

    template <typename U>
    __forceinline__ __device__ void addWarpwise(U&& x, bool const cond)
        requires(std::is_same_v<std::remove_cvref_t<U>, T>)
    {
#if DEBUG
        // __syncwarp();
        // if (__activemask() != 0xffff'ffffU) {
        //     LOG("Divergent Deadlock. base = ", base, " mask = 0xffff'ffff",
        //         " activemask = ", __activemask());
        //     assert(0);
        // }
#endif // DEBUG
        val hit = __ballot_sync(0xffff'ffffU, cond);
        if (hit == 0) return;
        val off = __popc(hit & ((1U << get_lane_id()) - 1U));
        int32_t pos = get_lane_id() == 0 ? atomicAdd(wptr, __popc(hit)) : 0;
        pos = __shfl_sync(0xffff'ffffU, pos, 0);
#if DEBUG
        if (pos + off >= capacity) {
            LOG("Overflow. base = ", base, " capacity = ", capacity);
            assert(0);
        }
#endif // DEBUG
        if (cond) base[pos + off] = std::forward<U>(x);
    }

    template <typename U>
    __forceinline__ __device__ void addWarpwise(U&& x, bool const cond, uint32_t const mask)
        requires(std::is_same_v<std::remove_cvref_t<U>, T>)
    {
#if DEBUG
        __syncwarp(mask);
        if ((mask & __activemask()) != mask) {
            LOG("Divergent Deadlock. base = ", base, " mask = ", mask,
                " activemask = ", __activemask());
            assert(0);
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
            LOG("Overflow. base = ", base, " capacity = ", capacity);
            assert(0);
        }
#endif // DEBUG
        if (cond) base[pos + off] = std::forward<U>(x);
    }

    template <typename U>
        requires requires(T t, U u) { u = t; }
    __forceinline__ __device__ int32_t drainBlockwise(U& out)
    {
        LOG("DO NOT USE THIS!");
#if DEBUG
        if (blockDim.y > 1 || blockDim.z > 1) {
            LOG("Incompatible configuration. blockDim = {", blockDim.x, ", ", blockDim.y, ", ",
                blockDim.z, "}");
            assert(0);
        }
#endif // DEBUG
        __shared__ int32_t cnt;
        __shared__ int32_t pos;
        while (threadIdx.x == 0) {
            cnt = min(blockDim.x, *wptr - *rptr);
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
