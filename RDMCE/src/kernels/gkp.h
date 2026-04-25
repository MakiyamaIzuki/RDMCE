#pragma once
#ifndef GKP_H
#define GKP_H

#include "cuda.h"
#include "cuda_runtime_api.h"

#include <cassert>
#include <cstdint>
#include <type_traits>

#ifndef __CUDA_ARCH__
#include <iostream>
#include <string>
#include <string_view>
#endif

namespace gkp
{

constexpr uint32_t INVALID_VID = 0xffff'ffffU;

#define PRINT(...)                                                                                   \
    gkp::heprint(__func__, '(', gkp::get_filename(__FILE__), ':', __LINE__, "): ", ##__VA_ARGS__)

#ifndef DEBUG
#define LOG(...) ((void)(0))
#elif defined(__CUDA_ARCH__)
#define LOG(...)                                                                                   \
    gkp::deprint(                                                                                  \
        __func__, '(', gkp::get_filename(__FILE__), ':', __LINE__, ") @ [", blockIdx.x, '/',       \
        gridDim.x, ", ", threadIdx.x, '/', blockDim.x, "]: ", ##__VA_ARGS__)
#else
#define LOG(...)                                                                                   \
    gkp::heprint(__func__, '(', gkp::get_filename(__FILE__), ':', __LINE__, "): ", ##__VA_ARGS__)
#endif

#if DEBUG
#define CUDA_INVOKE(fun, ...)                                                                      \
    do {                                                                                           \
        static_assert(                                                                             \
            std::string_view(#fun).starts_with("cuda") &&                                          \
                !std::string_view(#fun).ends_with("Async"),                                        \
            "Only for CUDA synchronized API.");                                                    \
        if (auto e = fun(__VA_ARGS__); e != cudaSuccess) {                                         \
            LOG(cudaGetErrorString(e));                                                            \
            assert(0);                                                                             \
        }                                                                                          \
    } while (0)
#else
#define CUDA_INVOKE(fun, ...)                                                                      \
    do {                                                                                           \
        static_assert(                                                                             \
            std::string_view(#fun).starts_with("cuda") &&                                          \
                !std::string_view(#fun).ends_with("Async"),                                        \
            "Only for CUDA synchronized API.");                                                    \
        fun(__VA_ARGS__);                                                                          \
    } while (0)
#endif // DEBUG

template <typename T>
constexpr T align32(T x)
    requires (std::is_integral_v<T>)
{
    return (x + 31) & ~static_cast<T>(31);
}

consteval char const* get_filename(char const* path)
{
    auto file = path;
    while (*path) {
        if (*path == '/' || *path == '\\') {
            file = path + 1;
        }
        ++path;
    }
    return file;
}

struct Logger
{
    int writing = 0;
    __device__ void acquire()
    {
        while (atomicCAS(&this->writing, 0, 1) == 1)
            ;
    }
    __device__ void release()
    {
        atomicExch(&this->writing, 0);
    }
};

__device__ static Logger logger;

template <typename T>
__device__ void deprint1(T x)
{
    using U = std::remove_cvref_t<T>;
    if constexpr (std::is_same_v<char, U>) {
        printf("%c", x);
    }
    else if constexpr (std::is_same_v<U, char*> || std::is_same_v<U, char const*>) {
        printf("%s", x);
    }
    else if constexpr (std::is_pointer_v<U>) {
        printf("%p", x);
    }
    else if constexpr (std::is_integral_v<U> && std::is_signed_v<U>) {
        printf("%lld", static_cast<int64_t>(x));
    }
    else if constexpr (std::is_integral_v<U> && std::is_unsigned_v<U>) {
        printf("%llu", static_cast<uint64_t>(x));
    }
}

template <typename... T>
__device__ void deprint(T const&... args)
{
    logger.acquire();
    (deprint1(args), ...);
    deprint1("\n");
    logger.release();
}
template <typename... T>
void heprint(T const&... args)
{
    ((std::cerr << args), ...);
    std::cerr << std::endl;
}

__device__ __forceinline__ auto get_lane_id()
{
    unsigned ret;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(ret));
    return ret;
}

// The number of threads per block should be not greater than 1024, and be a multiple of 32.
// Only threads with threadIdx.x < 32 can obtain the summation.
template <typename Addable>
    requires((std::integral<Addable> || std::floating_point<Addable>) && sizeof(Addable) >= 4)
__forceinline__ __device__ Addable sumBlockwise(Addable a)
{
#if DEBUG
    if (blockDim.x > 1024 || blockDim.y > 1 || blockDim.z > 1 || (blockDim.x & 0x1f)) {
        LOG("Incompatible configuration blockDim = {", blockDim.x, ", ", blockDim.y, ", ",
            blockDim.z, "}");
        assert(0);
    }
    __syncwarp();
    if (__activemask() != 0xffff'ffffU) {
        LOG("Divergent deadlock. activemask = ", __activemask());
        assert(0);
    }
#endif // DEBUG
    a += __shfl_xor_sync(0xffff'ffffU, a, 1);
    a += __shfl_xor_sync(0xffff'ffffU, a, 2);
    a += __shfl_xor_sync(0xffff'ffffU, a, 4);
    a += __shfl_xor_sync(0xffff'ffffU, a, 8);
    a += __shfl_xor_sync(0xffff'ffffU, a, 16);
    if (blockDim.x == 32) return a;
    __shared__ Addable t[32];
    auto const wid = threadIdx.x >> 5;
    if ((threadIdx.x & 0x1f) == 0) t[wid] = a;
    __syncthreads();
    if (wid == 0) {
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

} // namespace gkp

#endif // GKP_H
