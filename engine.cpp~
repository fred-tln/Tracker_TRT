/**
 * @file engine.cpp
 * @brief TensorRT engine wrapper implementation
 */

#include "engine.h"
#include "fp16_utils.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>

namespace {

class TRTLogger : public nvinfer1::ILogger {
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        switch (severity) {
            case nvinfer1::ILogger::Severity::kERROR:
                std::cerr << "[TRT ERROR] " << msg << std::endl;
                break;
            case nvinfer1::ILogger::Severity::kWARNING:
                std::cerr << "[TRT WARN] " << msg << std::endl;
                break;
            default:
                break;
        }
    }
} gTRTLogger;

}

// =============================================================================
// OSTrackEngine
// =============================================================================

OSTrackEngine::OSTrackEngine() {
    runtime_ = nvinfer1::createInferRuntime(gTRTLogger);
    cudaStreamCreate(&stream_);
}

OSTrackEngine::~OSTrackEngine() {
    if (d_template_)   cudaFree(d_template_);
    if (d_search_)     cudaFree(d_search_);
    if (d_score_map_)  cudaFree(d_score_map_);
    if (d_size_map_)   cudaFree(d_size_map_);
    if (d_offset_map_) cudaFree(d_offset_map_);
    if (stream_)       cudaStreamDestroy(stream_);
    if (context_)      context_->destroy();
    if (engine_)       engine_->destroy();
    if (runtime_)      runtime_->destroy();
}

size_t OSTrackEngine::dimsCount(const nvinfer1::Dims& d) {
    size_t count = 1;
    for (int i = 0; i < d.nbDims; i++) count *= d.d[i];
    return count;
}

bool OSTrackEngine::load(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[OSTrackEngine] Cannot open: " << engine_path << std::endl;
        return false;
    }
    engine_data_.assign(std::istreambuf_iterator<char>(file), {});
    file.close();

    if (engine_data_.empty()) {
        std::cerr << "[OSTrackEngine] Empty engine file" << std::endl;
        return false;
    }

    engine_ = runtime_->deserializeCudaEngine(engine_data_.data(), engine_data_.size());
    if (!engine_) {
        std::cerr << "[OSTrackEngine] Failed to deserialize engine" << std::endl;
        return false;
    }

    context_ = engine_->createExecutionContext();
    if (!context_) {
        std::cerr << "[OSTrackEngine] Failed to create execution context" << std::endl;
        return false;
    }

    // Discover tensors - two passes:
    // Pass 1: collect inputs to identify template (smaller spatial) vs search (larger)
    // Pass 2: allocate and bind all tensors
    int num_io = engine_->getNbIOTensors();
    std::cout << "[OSTrackEngine] IO tensors: " << num_io << std::endl;

    // --- Pass 1: identify input roles by spatial size ---
    struct InputInfo { int idx; int spatial; }; // spatial = d[2] (assumed square)
    std::vector<InputInfo> inputs;
    for (int i = 0; i < num_io; i++) {
        const char* name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            nvinfer1::Dims d = engine_->getTensorShape(name);
            if (d.nbDims == 4) {
                inputs.push_back({i, d.d[2]});
            }
        }
    }
      
    // Determine which input is template (smaller) and which is search (larger)
    int template_input_idx = -1;
    int search_input_idx   = -1;
    if (inputs.size() == 1) {
        search_input_idx = inputs[0].idx;
    } else if (inputs.size() >= 2) {
        if (inputs[0].spatial <= inputs[1].spatial) {
            template_input_idx = inputs[0].idx;
            search_input_idx   = inputs[1].idx;
        } else {
            template_input_idx = inputs[1].idx;
            search_input_idx   = inputs[0].idx;
        }
    }

    // --- Pass 2: print info, allocate GPU memory, and bind ---
    for (int i = 0; i < num_io; i++) {
        const char* name = engine_->getIOTensorName(i);
        nvinfer1::Dims dims = engine_->getTensorShape(name);
        nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(name);
        nvinfer1::DataType dtype = engine_->getTensorDataType(name);
        size_t elems = dimsCount(dims);
        size_t elem_bytes = (dtype == nvinfer1::DataType::kHALF) ? 2 : 4;
        size_t total = elems * elem_bytes;

        std::string mode_str = (mode == nvinfer1::TensorIOMode::kINPUT) ? "IN  " : "OUT ";
        std::string dtype_str = (dtype == nvinfer1::DataType::kHALF) ? "FP16" : "FP32";

        std::cout << "  " << mode_str << "[" << i << "] " << name
                  << "  shape=(";
        for (int d = 0; d < dims.nbDims; d++) {
            if (d > 0) std::cout << ",";
            std::cout << dims.d[d];
        }
        std::cout << ")  " << dtype_str << "  " << total << " bytes" << std::endl;

        // Identify and allocate
        std::string n(name);
        bool is_input = (mode == nvinfer1::TensorIOMode::kINPUT);

        if (is_input) {
            // Template: the smaller spatial input (detected in pass 1)
            if (i == template_input_idx) {
                template_size_ = dims.d[2];
                template_bytes_ = total;
                is_fp16_ = (dtype == nvinfer1::DataType::kHALF);
                cudaMalloc(&d_template_, total);
                context_->setTensorAddress(name, d_template_);
                std::cout << "    -> bound to template  (size=" << template_size_ << ")" << std::endl;
            }
            // Search: the larger spatial input (detected in pass 1)
            else if (i == search_input_idx) {
                search_size_ = dims.d[2];
                search_bytes_ = total;
                cudaMalloc(&d_search_, total);
                context_->setTensorAddress(name, d_search_);
                feat_sz_ = search_size_ / 16;  // stride=16
                std::cout << "    -> bound to search  (feat_sz=" << feat_sz_ << ")" << std::endl;
            }
        } else {
            // Output tensors
            if (n.find("score") != std::string::npos || n.find("Score") != std::string::npos) {
                score_map_bytes_ = total;
                cudaMalloc(&d_score_map_, total);
                context_->setTensorAddress(name, d_score_map_);
                if (dims.nbDims == 4) feat_sz_ = dims.d[2];
                std::cout << "    -> bound to score_map" << std::endl;
            }
            else if (n.find("size") != std::string::npos || n.find("Size") != std::string::npos) {
                size_map_bytes_ = total;
                cudaMalloc(&d_size_map_, total);
                context_->setTensorAddress(name, d_size_map_);
                std::cout << "    -> bound to size_map" << std::endl;
            }
            else if (n.find("offset") != std::string::npos || n.find("Offset") != std::string::npos) {
                offset_map_bytes_ = total;
                cudaMalloc(&d_offset_map_, total);
                context_->setTensorAddress(name, d_offset_map_);
                std::cout << "    -> bound to offset_map" << std::endl;
            }
            // Fallback: by shape
            else if (dims.nbDims == 4 && dims.d[1] == 1 && dims.d[2] <= 32 && dims.d[3] <= 32) {
                if (!d_score_map_) {
                    score_map_bytes_ = total;
                    cudaMalloc(&d_score_map_, total);
                    context_->setTensorAddress(name, d_score_map_);
                    feat_sz_ = dims.d[2];
                    std::cout << "    -> bound to score_map (by shape)" << std::endl;
                }
            }
            else if (dims.nbDims == 4 && dims.d[1] == 2 && dims.d[2] <= 32 && dims.d[3] <= 32) {
                if (!d_size_map_) {
                    size_map_bytes_ = total;
                    cudaMalloc(&d_size_map_, total);
                    context_->setTensorAddress(name, d_size_map_);
                    std::cout << "    -> bound to size_map (by shape)" << std::endl;
                } else if (!d_offset_map_) {
                    offset_map_bytes_ = total;
                    cudaMalloc(&d_offset_map_, total);
                    context_->setTensorAddress(name, d_offset_map_);
                    std::cout << "    -> bound to offset_map (by shape)" << std::endl;
                }
            }
        }
    }
    
/*  ADD */
 
std::cout << "[DEBUG] score_map bound: " << (d_score_map_ ? "YES" : "NO") << std::endl;
std::cout << "[DEBUG] size_map  bound: " << (d_size_map_  ? "YES" : "NO") << std::endl;
std::cout << "[DEBUG] offset_map bound:" << (d_offset_map_? "YES" : "NO") << std::endl;
    
/*  END ADD */

/*  SUPR
    if (!d_template_ || !d_search_ || !d_score_map_) {
        std::cerr << "[OSTrackEngine] Missing required buffers!" << std::endl;
        return false;
    }   
END  SUPR */
/*   ADD */
    if (!d_template_ || !d_search_ || !d_score_map_ || !d_size_map_ || !d_offset_map_) {
        std::cerr << "[OSTrackEngine] Missing required buffers! "
              << "score=" << (d_score_map_?1:0)
              << " size=" << (d_size_map_?1:0)
              << " offset=" << (d_offset_map_?1:0) << std::endl;
        return false;
    }
/*  END  ADD */

    std::cout << "[OSTrackEngine] Loaded OK  feat_sz=" << feat_sz_
              << "  search=" << search_size_
              << "  template=" << template_size_
              << "  fp16=" << (is_fp16_ ? "yes" : "no") << std::endl;
    return true;
}

bool OSTrackEngine::infer(
    const float* template_img,
    const float* search_img,
    float* score_map,
    float* size_map,
    float* offset_map
) {
    if (is_fp16_) {
        // FP16 engine: convert FP32 -> FP16 on GPU, then run inference
        size_t template_elems = template_size_ * template_size_ * 3;
        size_t search_elems = search_size_ * search_size_ * 3;

        // Allocate temporary FP32 GPU buffers
        float* d_tmp_template = nullptr;
        float* d_tmp_search = nullptr;
        size_t template_fp32_bytes = template_elems * sizeof(float);
        size_t search_fp32_bytes = search_elems * sizeof(float);

        cudaMalloc(&d_tmp_template, template_fp32_bytes);
        cudaMalloc(&d_tmp_search, search_fp32_bytes);

        // Upload FP32 from host
        cudaMemcpyAsync(d_tmp_template, template_img, template_fp32_bytes, cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(d_tmp_search, search_img, search_fp32_bytes, cudaMemcpyHostToDevice, stream_);

        // Convert FP32 -> FP16
        fp32ToFp16(d_tmp_template, d_template_, template_elems, stream_);
        fp32ToFp16(d_tmp_search, d_search_, search_elems, stream_);

/*   ADD */
        cudaStreamSynchronize(stream_);
/*  END  ADD */
        cudaFree(d_tmp_template);
        cudaFree(d_tmp_search);
    } else {
        // FP32 engine: direct copy
        cudaMemcpyAsync(d_template_, template_img, template_bytes_, cudaMemcpyHostToDevice, stream_);
        cudaMemcpyAsync(d_search_, search_img, search_bytes_, cudaMemcpyHostToDevice, stream_);
    }

    // Execute
    bool ok = context_->enqueueV3(stream_);
    if (!ok) {
        std::cerr << "[OSTrackEngine] executeV2 failed" << std::endl;
        return false;
    }

    // Copy outputs back
    size_t sm_elems = feat_sz_ * feat_sz_;
    size_t sz_elems = 2 * feat_sz_ * feat_sz_;
    size_t of_elems = 2 * feat_sz_ * feat_sz_;

    if (is_fp16_) {
        // FP16 engine: convert FP16 -> FP32 on GPU
        float* d_tmp_score = nullptr;
        float* d_tmp_size = nullptr;
        float* d_tmp_offset = nullptr;

        cudaMalloc(&d_tmp_score, sm_elems * sizeof(float));
        cudaMalloc(&d_tmp_size, sz_elems * sizeof(float));
        cudaMalloc(&d_tmp_offset, of_elems * sizeof(float));

        fp16ToFp32(d_score_map_, d_tmp_score, sm_elems, stream_);
        fp16ToFp32(d_size_map_, d_tmp_size, sz_elems, stream_);
        fp16ToFp32(d_offset_map_, d_tmp_offset, of_elems, stream_);

        cudaMemcpyAsync(score_map, d_tmp_score, sm_elems * sizeof(float), cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(size_map, d_tmp_size, sz_elems * sizeof(float), cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(offset_map, d_tmp_offset, of_elems * sizeof(float), cudaMemcpyDeviceToHost, stream_);

/*   ADD */
        cudaStreamSynchronize(stream_);
/*  END  ADD */
        cudaFree(d_tmp_score);
        cudaFree(d_tmp_size);
        cudaFree(d_tmp_offset);
    } else {
        // FP32 engine: direct copy
        cudaMemcpyAsync(score_map, d_score_map_, sm_elems * sizeof(float), cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(size_map, d_size_map_, sz_elems * sizeof(float), cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(offset_map, d_offset_map_, of_elems * sizeof(float), cudaMemcpyDeviceToHost, stream_);
    }

    cudaStreamSynchronize(stream_);
    return true;
}
