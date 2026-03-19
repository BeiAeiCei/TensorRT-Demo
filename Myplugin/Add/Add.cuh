#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename T>
__global__ void addKernel(const T* input1, const T* input2, T* output, int totalElements);

void launchAddKernelFloat(
    const float* in1,
    const float* in2,
    float* out,
    int totalElements,
    cudaStream_t stream
);

void launchAddKernelHalf(
    const __half* in1,
    const __half* in2,
    __half* out,
    int totalElements,
    cudaStream_t stream
);