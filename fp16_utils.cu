/**
 * @file fp16_utils.cu
 * @brief FP16 <-> FP32 conversion kernels
 */

#include <cuda_fp16.h>

__global__ void fp32ToFp16Kernel(const float* src, __half* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(src[idx]);
    }
}

__global__ void fp16ToFp32Kernel(const __half* src, float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __half2float(src[idx]);
    }
}

void fp32ToFp16(const float* src, void* dst, size_t elem_count, cudaStream_t stream) {
    if (elem_count == 0) return;
    int blockSize = 256;
    int numBlocks = (elem_count + blockSize - 1) / blockSize;
    fp32ToFp16Kernel<<<numBlocks, blockSize, 0, stream>>>
        (src, static_cast<__half*>(dst), elem_count);
}

void fp16ToFp32(const void* src, float* dst, size_t elem_count, cudaStream_t stream) {
    if (elem_count == 0) return;
    int blockSize = 256;
    int numBlocks = (elem_count + blockSize - 1) / blockSize;
    fp16ToFp32Kernel<<<numBlocks, blockSize, 0, stream>>>
        (static_cast<const __half*>(src), dst, elem_count);
}

