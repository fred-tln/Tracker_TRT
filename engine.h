/**
 * @file engine.h
 * @brief TensorRT engine wrapper for OSTrack inference
 */

#ifndef OSTRACK_ENGINE_H
#define OSTRACK_ENGINE_H

#include <string>
#include <vector>
#include <memory>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>
#include "fp16_utils.h"

/**
 * @brief Lightweight TensorRT inference engine
 *
 * Loads a serialized engine, manages GPU buffers, and runs inference.
 */
class OSTrackEngine {
public:
    OSTrackEngine();
    ~OSTrackEngine();

    /**
     * @brief Load a serialized TensorRT engine from disk
     * @param engine_path Path to the .engine file
     * @return true on success
     */
    bool load(const std::string& engine_path);

    /**
     * @brief Run inference
     *
     * Inputs (host, FP32, CHW layout):
     *   template_img : [1, 3, template_size, template_size]
     *   search_img   : [1, 3, search_size, search_size]
     *
     * Outputs (host, FP32):
     *   score_map : [feat_sz, feat_sz]
     *   size_map  : [2, feat_sz, feat_sz]
     *   offset_map: [2, feat_sz, feat_sz]
     *
     * @return true on success
     */
    bool infer(
        const float* template_img,
        const float* search_img,
        float* score_map,
        float* size_map,
        float* offset_map
    );

    int getFeatSz()       const { return feat_sz_; }
    int getSearchSize()   const { return search_size_; }
    int getTemplateSize() const { return template_size_; }
    bool isFP16()         const { return is_fp16_; }

    // Non-copyable
    OSTrackEngine(const OSTrackEngine&) = delete;
    OSTrackEngine& operator=(const OSTrackEngine&) = delete;

private:
    // Engine & context
    nvinfer1::IRuntime*   runtime_   = nullptr;
    nvinfer1::ICudaEngine* engine_   = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    cudaStream_t stream_ = nullptr;

    // Tensor info
    int template_size_ = 192;   // default: matches engine tensor 'z' (1,3,192,192)
    int search_size_   = 384;
    int feat_sz_       = 24;
    bool is_fp16_      = false;

    // GPU buffers
    void* d_template_   = nullptr;
    void* d_search_     = nullptr;
    void* d_score_map_  = nullptr;
    void* d_size_map_   = nullptr;
    void* d_offset_map_ = nullptr;

    // Buffer sizes (bytes)
    size_t template_bytes_   = 0;
    size_t search_bytes_     = 0;
    size_t score_map_bytes_  = 0;
    size_t size_map_bytes_   = 0;
    size_t offset_map_bytes_ = 0;

    std::vector<char> engine_data_;

    // Helper: compute element count from Dims
    static size_t dimsCount(const nvinfer1::Dims& d);
};

#endif // OSTRACK_ENGINE_H
