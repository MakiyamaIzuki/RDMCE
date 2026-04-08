#pragma once
#ifndef GKP_QUEUE
#define GKP_QUEUE

#include "cuda.h"
#include <string>
#include <string_view>
#include <type_traits>

namespace gkp
{
#define val auto const
#define var auto
#if DEBUG
#define CUDA_INVOKE(fun, ...) do { \
    static_assert(std::string_view(#fun).starts_with("cuda") && \
    !std::string_view(#fun).ends_with("Async"), \
    "Only for CUDA synchronized API."); \
    if (auto e = fun(__VA_ARGS__); e != cudaSuccess) { \
    printf("%s(%d): %s\n", __func__, __LINE__, cudaGetErrorString(e)); \
    exit(__LINE__); } } while(0)
#else
#define CUDA_INVOKE(fun, ...) do { \
    static_assert(std::string_view(#fun).starts_with("cuda") && \
    !std::string_view(#fun).ends_with("Async"), \
    "Only for CUDA synchronized API."); fun(__VA_ARGS__); } while(0)
#endif

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
    Queue(T* const base, ssize_t* const size): base(base), sz(size) {
        *sz = 0LL;
    }

    ~Queue() {}

    inline ssize_t size(void) const { return *sz; }

    __forceinline__ __device__ void addWarpwise(T x, bool const cond) requires (std::is_arithmetic_v<T>) {
        val hit = __ballot_sync(0xffff'ffffU, cond); // TODO : Check if all of 32 have arrived here.
        val off = __popc(hit & ((1U << get_lane_id()) - 1U));
        var last = get_lane_id() == 0 ? atomicAdd(sz, __popc(hit)) : 0;
        last = __shfl_sync(0xffff'ffffU, last, 0);
        if (cond) base[last + off] = x;
    }

    template <typename U>
    __forceinline__ __device__ void addWarpwise(U&& x, bool const cond)
        requires (!std::is_arithmetic_v<T> && std::is_same_v<std::remove_cvref_t<U>, T>)
    {
        val hit = __ballot_sync(0xffff'ffffU, cond); // TODO : Check if all of 32 have arrived here.
        val off = __popc(hit & ((1U << get_lane_id()) - 1U));
        var last = get_lane_id() == 0 ? atomicAdd(sz, __popc(hit)) : 0;
        last = __shfl_sync(0xffff'ffffU, last, 0);
        if (cond) base[last + off] = std::forward<U>(x);
    }
};

#undef CUDA_INVOKE
#undef var
#undef val

} // namespace gkp

#endif // GKP_QUEUE

