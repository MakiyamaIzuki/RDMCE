#pragma once
#ifndef GKP_H
#define GKP_H

#include "cuda.h"
#include "cuda_runtime.h"

#include <cstdint>
#include <format>
#include <iostream>
#include <string>
#include <string_view>

namespace gkp
{

constexpr uint32_t INVALID_VID = 0xffff'ffffU;

#define LOG(...) eprint(std::format("%s(%d)", __func__, __LINE__), ##__VA_ARGS__)
#if DEBUG
#define CUDA_INVOKE(fun, ...)                                                                      \
    do {                                                                                           \
        static_assert(                                                                             \
            std::string_view(#fun).starts_with("cuda") &&                                          \
                !std::string_view(#fun).ends_with("Async"),                                        \
            "Only for CUDA synchronized API.");                                                    \
        if (auto e = fun(__VA_ARGS__); e != cudaSuccess) {                                         \
            printf("%s(%d): %s\n", __func__, __LINE__, cudaGetErrorString(e));                     \
            exit(__LINE__);                                                                        \
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

template <typename... T>
void eprint(T&&... args)
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
        exit(__LINE__);
    }
    if (__activemask() != 0xffff'ffffU) {
        LOG("Divergent deadlock activemast = ", std::format("%x", __activemask()));
        exit(__LINE__);
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
