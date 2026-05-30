/**
 * @file tracker.cpp
 * @brief OSTrackTracker implementation
 *
 * Implements the tracking logic:
 *   - extractCrop   : sample_target equivalent
 *   - normalizeToCHW: image normalization (BGR input, CHW output)
 *   - calBBox       : bounding box regression from score/size/offset maps
 *   - mapBoxBack    : map predicted box back to image coordinates
 *   - clipBox       : clamp box to image boundaries
 *   - genHann2D     : 2D Hann window for post-processing
 */

#include "ostrack_trt.h"
#include "engine.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>

OSTrackTracker::OSTrackTracker() {}

OSTrackTracker::~OSTrackTracker() {}

bool OSTrackTracker::load(const std::string& engine_path, const OSTrackConfig& config) {
    config_ = config;
    engine_ = std::make_unique<OSTrackEngine>();

    if (!engine_->load(engine_path)) {
        std::cerr << "[OSTrackTracker] Failed to load engine" << std::endl;
        return false;
    }

    // Override config with actual engine sizes to ensure consistency
    config_.template_size = engine_->getTemplateSize();
    config_.search_size   = engine_->getSearchSize();
    feat_sz_ = engine_->getFeatSz();
    hann_window_ = genHann2D(feat_sz_);

    std::cout << "[OSTrackTracker] Initialized  feat_sz=" << feat_sz_
              << "  template=" << config_.template_size
              << "  search=" << config_.search_size << std::endl;
    return true;
}

void OSTrackTracker::reset() {
    z_patch_.release();
}

// =============================================================================
// extractCrop  (equivalent of sample_target in processing_utils.py)
// =============================================================================
void OSTrackTracker::extractCrop(const cv::Mat& img, const BBox& bbox,
                                  float factor, int out_sz,
                                  cv::Mat& crop, float& resize_factor) {
    // bbox: x, y, w, h
    float x = bbox.x, y = bbox.y, w = bbox.w, h = bbox.h;

    // crop_sz = ceil(sqrt(w*h) * factor)
    int crop_sz = std::ceil(std::sqrt(w * h) * factor);
    if (crop_sz < 1) crop_sz = 1;

    // Center of the crop
    int x1 = std::round(x + 0.5f * w - 0.5f * crop_sz);
    int y1 = std::round(y + 0.5f * h - 0.5f * crop_sz);
    int x2 = x1 + crop_sz;
    int y2 = y1 + crop_sz;

    // Padding needed
    int x1_pad = std::max(0, -x1);
    int x2_pad = std::max(0, x2 - img.cols);
    int y1_pad = std::max(0, -y1);
    int y2_pad = std::max(0, y2 - img.rows);

    // Clipped region
    int ax1 = std::max(0, x1);
    int ay1 = std::max(0, y1);
    int ax2 = std::min(img.cols, x2);
    int ay2 = std::min(img.rows, y2);

    cv::Mat roi = img( cv::Rect(ax1, ay1, ax2 - ax1, ay2 - ay1) );

    // Pad with constant (black)
    cv::copyMakeBorder(roi, crop, y1_pad, y2_pad, x1_pad, x2_pad, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    // Resize to output size
    if (crop.size() != cv::Size(out_sz, out_sz)) {
        cv::resize(crop, crop, cv::Size(out_sz, out_sz), 0, 0, cv::INTER_LINEAR);
    }

    resize_factor = static_cast<float>(out_sz) / crop_sz;
}

// =============================================================================
// normalizeToCHW  (equivalent of Preprocessor.process)
// Input: BGR image (8UC3), Output: CHW normalized float array
// Channel order is intentionally kept as BGR to match the Python Preprocessor
// which applies mean/std to channels 0,1,2 without any BGR->RGB conversion.
// =============================================================================
void OSTrackTracker::normalizeToCHW(const cv::Mat& crop_bgr, std::vector<float>& out) {
    int sz = crop_bgr.cols;  // square image
    out.resize(3 * sz * sz);

    const float* mean = config_.mean;
    const float* std_ = config_.std_;

    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < sz; y++) {
            for (int x = 0; x < sz; x++) {
                int i = y * sz + x;  // row-major index
                float val = static_cast<float>(crop_bgr.ptr<uchar>(y)[x * 3 + c]) / 255.0f;
                out[c * sz * sz + i] = (val - mean[c]) / std_[c];
            }
        }
    }
}

// =============================================================================
// genHann2D  (equivalent of hann2d in hann.py)
// Matches torch.hann_window(n, periodic=False):
//   w[i] = 0.5 * (1 - cos(2*pi*i / (n-1)))   for i = 0 .. n-1
// The window is 0 at both endpoints and peaks at the centre, which fully
// suppresses the borders of the score map (important when the model outputs
// spurious activations at feature-map edges).
// =============================================================================
std::vector<float> OSTrackTracker::genHann2D(int sz) {
    std::vector<float> hann1d(sz);
    for (int i = 0; i < sz; i++) {
        // Use sz-1 as denominator so the window reaches exactly 0 at both ends.
        // Guard against sz==1 to avoid division by zero.
        hann1d[i] = (sz > 1)
            ? 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (sz - 1)))
            : 1.0f;
    }

    std::vector<float> hann2d(sz * sz);
    for (int y = 0; y < sz; y++) {
        for (int x = 0; x < sz; x++) {
            hann2d[y * sz + x] = hann1d[y] * hann1d[x];
        }
    }
    return hann2d;
}

// =============================================================================
// calBBox  (equivalent of CenterPredictor.cal_bbox in head.py)
// =============================================================================
void OSTrackTracker::calBBox(const float* score_map, const float* size_map,
                              const float* offset_map,
                              float& cx, float& cy, float& w, float& h) {
    // Find argmax of score_map
    int idx = 0;
    float max_score = score_map[0];
    for (int i = 1; i < feat_sz_ * feat_sz_; i++) {
        if (score_map[i] > max_score) {
            max_score = score_map[i];
            idx = i;
        }
    }

    int idx_y = idx / feat_sz_;
    int idx_x = idx % feat_sz_;

    // Gather size and offset at argmax position
    // size_map: [2, feat_sz, feat_sz] -> size_map[channel * feat_sz * feat_sz + pos]
    int pos = idx;  // flat position in feature map
    float size_0 = size_map[0 * feat_sz_ * feat_sz_ + pos];  // width
    float size_1 = size_map[1 * feat_sz_ * feat_sz_ + pos];  // height
    float offset_0 = offset_map[0 * feat_sz_ * feat_sz_ + pos];  // dx
    float offset_1 = offset_map[1 * feat_sz_ * feat_sz_ + pos];  // dy

    // bbox: cx, cy, w, h (normalized to [0,1] in feature map)
    cx = (static_cast<float>(idx_x) + offset_0) / feat_sz_;
    cy = (static_cast<float>(idx_y) + offset_1) / feat_sz_;
    w = size_0;
    h = size_1;
}

// =============================================================================
// mapBoxBack  (equivalent of OSTrack.map_box_back in ostrack.py)
// =============================================================================
BBox OSTrackTracker::mapBoxBack(float cx, float cy, float w, float h,
                                 float resize_factor, int search_size) {
    // Previous state center
    float cx_prev = state_.x + 0.5f * state_.w;
    float cy_prev = state_.y + 0.5f * state_.h;

    // half_side in original image coordinates
    float half_side = 0.5f * search_size / resize_factor;

    // Map back to original image
    float cx_real = cx * search_size / resize_factor + (cx_prev - half_side);
    float cy_real = cy * search_size / resize_factor + (cy_prev - half_side);
    float w_real = w * search_size / resize_factor;
    float h_real = h * search_size / resize_factor;

    return BBox{cx_real - 0.5f * w_real, cy_real - 0.5f * h_real, w_real, h_real};
}

// =============================================================================
// clipBox  (equivalent of clip_box in box_ops.py)
// =============================================================================
BBox OSTrackTracker::clipBox(const BBox& box, int H, int W, int margin) {
    BBox b = box;
    float x2 = std::max(static_cast<float>(margin), std::min(static_cast<float>(W - margin), box.x + box.w));
    float y2 = std::max(static_cast<float>(margin), std::min(static_cast<float>(H - margin), box.y + box.h));
    b.x = std::max(0.0f, std::min(static_cast<float>(W - margin), box.x));
    b.y = std::max(0.0f, std::min(static_cast<float>(H - margin), box.y));
    b.w = std::max(static_cast<float>(margin), x2 - b.x);
    b.h = std::max(static_cast<float>(margin), y2 - b.y);
    return b;
}

// =============================================================================
// initialize  (equivalent of OSTrack.initialize in ostrack.py)
// =============================================================================
bool OSTrackTracker::initialize(const cv::Mat& frame, const BBox& init_bbox) {
    if (!engine_) {
        std::cerr << "[OSTrackTracker] Engine not loaded" << std::endl;
        return false;
    }

    state_ = init_bbox;

    // Extract template crop (BGR from frame) and store as-is.
    // The Python Preprocessor.process does NOT convert BGR->RGB before
    // building the CHW tensor, so the TRT engine expects BGR channel order.
    extractCrop(frame, init_bbox, config_.template_factor,
                config_.template_size, z_patch_, resize_factor_z_);

    std::cout << "[OSTrackTracker] Initialized with bbox: "
              << init_bbox.x << "," << init_bbox.y << ","
              << init_bbox.w << "," << init_bbox.h << std::endl;
    return true;
}

// =============================================================================
// track  (equivalent of OSTrack.track in ostrack.py)
// =============================================================================
BBox OSTrackTracker::track(const cv::Mat& frame) {
    if (!engine_ || z_patch_.empty()) {
        std::cerr << "[OSTrackTracker] Not initialized" << std::endl;
        return state_;
    }

    // Extract search crop (BGR from frame).
    // Keep BGR channel order to match the Python Preprocessor.process behaviour.
    cv::Mat x_patch_bgr;
    float resize_factor_x;
    extractCrop(frame, state_, config_.search_factor,
                config_.search_size, x_patch_bgr, resize_factor_x);

    // Normalize both crops (BGR -> CHW normalized)
    std::vector<float> template_chw;
    std::vector<float> search_chw;
    normalizeToCHW(z_patch_, template_chw);
    normalizeToCHW(x_patch_bgr, search_chw);

    // Run inference
    std::vector<float> score_map(feat_sz_ * feat_sz_);
    std::vector<float> size_map(2 * feat_sz_ * feat_sz_);
    std::vector<float> offset_map(2 * feat_sz_ * feat_sz_);

    if (!engine_->infer(template_chw.data(), search_chw.data(),
                        score_map.data(), size_map.data(), offset_map.data())) {
        std::cerr << "[OSTrackTracker] Inference failed" << std::endl;
        return state_;
    }

    // Apply Hann window to score map
    for (int i = 0; i < feat_sz_ * feat_sz_; i++) {
        score_map[i] *= hann_window_[i];
    }

    // Calculate bounding box
    float cx, cy, w, h;
    calBBox(score_map.data(), size_map.data(), offset_map.data(),
            cx, cy, w, h);

    // Map back to original coordinates
    BBox pred = mapBoxBack(cx, cy, w, h, resize_factor_x, config_.search_size);

    // Clip to image boundaries
    pred = clipBox(pred, frame.rows, frame.cols, 10);

    state_ = pred;
    return state_;
}
