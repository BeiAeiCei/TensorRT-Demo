#include "Add.cuh"
#include <cuda_fp16.h>

template <typename T>
__global__ void addKernel(const T* input1, const T* input2, T* output, int totalElements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalElements)
    {
        output[idx] = input1[idx] + input2[idx];
    }
}

template <>
__global__ void addKernel<__half>(const __half* input1, const __half* input2, __half* output, int totalElements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalElements)
    {
        output[idx] = __hadd(input1[idx], input2[idx]);
    }
}

void launchAddKernelFloat(
    const float* in1,
    const float* in2,
    float* out,
    int totalElements,
    cudaStream_t stream
)
{
    int block = 256;
    int grid = (totalElements + block - 1) / block;
    addKernel<float><<<grid, block, 0, stream>>>(in1, in2, out, totalElements);
}

void launchAddKernelHalf(
    const __half* in1,
    const __half* in2,
    __half* out,
    int totalElements,
    cudaStream_t stream
)
{
    int block = 256;
    int grid = (totalElements + block - 1) / block;
    addKernel<__half><<<grid, block, 0, stream>>>(in1, in2, out, totalElements);
}