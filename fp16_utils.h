/**
 * @file fp16_utils.h
 * @brief FP16 <-> FP32 conversion utilities
 *
 * GPU kernels for FP32 <-> FP16 conversion.
 */

#ifndef FP16_UTILS_H
#define FP16_UTILS_H

#include <cuda_runtime.h>
#include <cstddef>

/**
 * @brief Convert FP32 device array to FP16 device array
 * @param src_device   Source FP32 array (device)
 * @param dst_device   Destination FP16 array (device)
 * @param n            Number of elements
 * @param stream       CUDA stream
 */
void fp32ToFp16(const float* src_device, void* dst_device, size_t n, cudaStream_t stream);

/**
 * @brief Convert FP16 device array to FP32 device array
 * @param src_device   Source FP16 array (device)
 * @param dst_device   Destination FP32 array (device)
 * @param n            Number of elements
 * @param stream       CUDA stream
 */
void fp16ToFp32(const void* src_device, float* dst_device, size_t n, cudaStream_t stream);

#endif // FP16_UTILS_H"
