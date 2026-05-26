/**
 * @file ostrack_trt.h
 * @brief OSTrack visual object tracker using NVIDIA TensorRT
 *
 * This module provides a C++ implementation of the OSTrack tracker
 * using TensorRT for accelerated inference on the ostrack_384_fp16.engine model.
 */

#ifndef OSTRACK_TRT_H
#define OSTRACK_TRT_H

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Bounding box: x, y, width, height (in pixels)
struct BBox {
    float x = 0, y = 0, w = 0, h = 0;
};

// Tracker configuration matching OSTrack Python defaults
struct OSTrackConfig {
    // Template crop parameters
    float template_factor = 2.0f;
    int   template_size   = 192;  // matches engine tensor 'z' shape (1,3,192,192)

    // Search crop parameters (384 for the 384 model)
    float search_factor = 5.0f;
    int   search_size   = 384;   // matches engine tensor 'x' shape (1,3,384,384)

    // Backbone stride (ViT patch size)
    int backbone_stride = 16;

    // ImageNet normalization
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std_[3] = {0.229f, 0.224f, 0.225f};
};

// Forward declarations
class OSTrackEngine;

/**
 * @brief High-level OSTrack tracker wrapper
 */
class OSTrackTracker {
public:
    OSTrackTracker();
    ~OSTrackTracker();

    /**
     * @brief Load the TensorRT engine and initialize the tracker
     * @param engine_path  Path to *.engine file
     * @param config       Tracker configuration
     * @return true on success
     */
    bool load(const std::string& engine_path, const OSTrackConfig& config = OSTrackConfig());

    /**
     * @brief Initialize tracking: extract template from the first frame
     * @param frame      First frame (BGR, 8UC3)
     * @param init_bbox  Initial bounding box [x, y, w, h]
     * @return true on success
     */
    bool initialize(const cv::Mat& frame, const BBox& init_bbox);

    /**
     * @brief Track the object in a new frame
     * @param frame  Current frame (BGR, 8UC3)
     * @return Predicted bounding box
     */
    BBox track(const cv::Mat& frame);

    /** Get the last predicted bounding box */
    BBox getState() const { return state_; }

    /** Reset the tracker (release template) */
    void reset();

private:
    OSTrackConfig config_;
    std::unique_ptr<OSTrackEngine> engine_;

    BBox state_;
    cv::Mat z_patch_;          // stored template crop (BGR, matching Python Preprocessor input)
    float resize_factor_z_ = 1.0f;

    int feat_sz_ = 24;         // search_size / backbone_stride
    std::vector<float> hann_window_;  // 2D Hann window [feat_sz*feat_sz]

    // ---- internal helpers ----
    void extractCrop(const cv::Mat& img, const BBox& bbox,
                     float factor, int out_sz,
                     cv::Mat& crop, float& resize_factor);

    void normalizeToCHW(const cv::Mat& crop_bgr, std::vector<float>& out);

    void calBBox(const float* score_map, const float* size_map,
                 const float* offset_map,
                 float& cx, float& cy, float& w, float& h);

    BBox mapBoxBack(float cx, float cy, float w, float h,
                    float resize_factor, int search_size);

    BBox clipBox(const BBox& box, int H, int W, int margin = 10);

    std::vector<float> genHann2D(int sz);
};

/**
 * @brief Interactive demo: open video -> select ROI -> track
 *
 * Controls:
 *   Mouse  - draw ROI rectangle, press Enter to confirm
 *   q      - quit
 *   r      - reset (pick new ROI)
 *   Space  - pause / resume
 *   s      - toggle saving output video
 *
 * @return 0 on success, -1 on error
 */
int runTrackingDemo(const std::string& video_path,
                    const std::string& engine_path,
                    const OSTrackConfig& config = OSTrackConfig());

#endif // OSTRACK_TRT_H
