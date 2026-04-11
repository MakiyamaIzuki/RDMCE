#pragma once
#include <__clang_cuda_runtime_wrapper.h>
#ifndef GKP_QUEUE
#define GKP_QUEUE

#include "cuda.h"

#include <iostream>
#include <string>
#include <string_view>
#include <type_traits>

namespace gkp
{
#define val auto const
#define var auto
#define LOG(fmt, ...) eprint(std::format("%s(%d)", __func__, __LINE__), ##__VA_ARGS__)
#if DEBUG
#define CUDA_INVOKE(fun, ...)                                                                      \
    do                                                                                             \
    {                                                                                              \
        static_assert(                                                                             \
            std::string_view(#fun).starts_with("cuda") &&                                          \
                !std::string_view(#fun).ends_with("Async"),                                        \
            "Only for CUDA synchronized API.");                                                    \
        if (auto e = fun(__VA_ARGS__); e != cudaSuccess)                                           \
        {                                                                                          \
            printf("%s(%d): %s\n", __func__, __LINE__, cudaGetErrorString(e));                     \
            exit(__LINE__);                                                                        \
        }                                                                                          \
    } while (0)
#else
#define CUDA_INVOKE(fun, ...)                                                                      \
    do                                                                                             \
    {                                                                                              \
        static_assert(                                                                             \
            std::string_view(#fun).starts_with("cuda") &&                                          \
                !std::string_view(#fun).ends_with("Async"),                                        \
            "Only for CUDA synchronized API.");                                                    \
        fun(__VA_ARGS__);                                                                          \
    } while (0)
#endif // DEBUG

template <typename... T>
void eprint(T&&... args)
{
    ((std::cerr << args), ...);
    std::cerr << std::endl;
}

__device__ __forceinline__ auto get_lane_id()
{
    unsigned ret;
    asm volatile("mov.u32 %0, %laneid;" : "=r"(ret));
    return ret;
}

template <typename T>
struct Queue
{
    T* const base;
    ssize_t* const sz;

    // Invoker should allocate and free memory.
    Queue(T* const base, ssize_t* const size) : base(base), sz(size)
    {
        *sz = 0LL;
    }

    ~Queue()
    {
    }

    inline ssize_t size(void) const
    {
        return *sz;
    }

    __forceinline__ __device__ T& operator[](std::integral auto x)
    {
#if DEBUG
        if (x >= *sz || static_cast<int64_t>(x) & (1ULL << 63))
        {
            LOG("Out of range\tx = ", x, "\tsz = ", sz);
            exit(__LINE__);
        }
#endif // DEBUG
        return base[x];
    }

    __forceinline__ __device__ T const operator[](std::integral auto x) const
        requires(std::is_arithmetic_v<T>)
    {
#if DEBUG
        if (x >= *sz || static_cast<int64_t>(x) & (1ULL << 63))
        {
            LOG("Out of range\tx = ", x, "\tsz = ", sz);
            exit(__LINE__);
        }
#endif // DEBUG
        return base[x];
    }

    template <typename U>
    __forceinline__ __device__ void addWarpwise(
        U&& x, bool const cond, uint32_t const mask = 0xffff'ffffU)
        requires(std::is_same_v<std::remove_cvref_t<U>, T>)
    {
#if DEBUG
        if (mask & __activemask() != mask)
        {
            LOG("Divergent Deadlock\tmask = ", mask, "\tactivemask = ", __activemask());
            exit(__LINE__);
        }
#endif // DEBUG
        val hit = __ballot_sync(mask, cond);
        val off = __popc(hit & ((1U << get_lane_id()) - 1U));
        var last = get_lane_id() == 0 ? atomicAdd(sz, __popc(hit)) : 0LL;
        last = __shfl_sync(mask, last, 0);
        if (cond)
            base[last + off] = std::forward<U>(x);
    }
};

#undef CUDA_INVOKE
#undef LOG
#undef var
#undef val

} // namespace gkp

#endif // GKP_QUEUE
