/**
 * @file demo.cpp
 * @brief Interactive tracking demo
 *
 * Equivalent of demo_tracking.py in Python.
 *
 * Usage:
 *   ./ostrack_demo <video_path> <engine_path>
 *
 * Controls:
 *   Mouse  - draw ROI rectangle, press Enter to confirm
 *   q      - quit
 *   r      - reset (pick new ROI)
 *   Space  - pause / resume
 *   s      - toggle saving output video
 */

#include "ostrack_trt.h"
#include <iostream>
#include <string>
#include <cstdio>

// Global state for mouse callback
struct ROISelector {
    bool selecting = false;
    bool finished = false;
    int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    cv::Point origin;
};

static ROISelector g_selector;

static void onMouse(int event, int x, int y, int, void*) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        g_selector.selecting = true;
        g_selector.finished = false;
        g_selector.origin = cv::Point(x, y);
        g_selector.x1 = x;
        g_selector.y1 = y;
    } else if (event == cv::EVENT_MOUSEMOVE && g_selector.selecting) {
        g_selector.x2 = x;
        g_selector.y2 = y;
    } else if (event == cv::EVENT_LBUTTONUP && g_selector.selecting) {
        g_selector.selecting = false;
        g_selector.x2 = x;
        g_selector.y2 = y;
        g_selector.finished = true;
    }
}

static BBox selectROI(const std::string& win_name, cv::Mat& frame) {
    g_selector.selecting = false;
    g_selector.finished = false;

    cv::setMouseCallback(win_name, onMouse, nullptr);

    while (!g_selector.finished) {
        cv::Mat display = frame.clone();

        // Draw instructions
        cv::putText(display, "Select target with mouse, then press Enter",
                    cv::Point(20, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.0,
                    cv::Scalar(0, 255, 0), 2);
        cv::putText(display, "Click and drag to draw the bounding box",
                    cv::Point(20, 60), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8,
                    cv::Scalar(200, 200, 200), 1);

        // Draw rectangle while selecting
        if (g_selector.selecting || g_selector.finished) {
            int x = std::min(g_selector.x1, g_selector.x2);
            int y = std::min(g_selector.y1, g_selector.y2);
            int w = std::abs(g_selector.x2 - g_selector.x1);
            int h = std::abs(g_selector.y2 - g_selector.y1);
            cv::rectangle(display, cv::Rect(x, y, w, h),
                          cv::Scalar(0, 255, 0), 2);
        }

        cv::imshow(win_name, display);

        int key = cv::waitKey(30) & 0xFF;
        if (key == 13) { // Enter
            if (g_selector.finished) break;
        }
    }

    cv::setMouseCallback(win_name, nullptr, nullptr);

    int x = std::min(g_selector.x1, g_selector.x2);
    int y = std::min(g_selector.y1, g_selector.y2);
    int w = std::abs(g_selector.x2 - g_selector.x1);
    int h = std::abs(g_selector.y2 - g_selector.y1);

    return BBox{static_cast<float>(x), static_cast<float>(y),
                static_cast<float>(w), static_cast<float>(h)};
}

int runTrackingDemo(const std::string& video_path,
                    const std::string& engine_path,
                    const OSTrackConfig& config) {
    std::cout << "========================================" << std::endl;
    std::cout << "  OSTrack - Visual Object Tracking Demo" << std::endl;
    std::cout << "========================================" << std::endl;

    // Open video
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "[Demo] Cannot open video: " << video_path << std::endl;
        return -1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    std::cout << "\nVideo info:" << std::endl;
    std::cout << "  Resolution: " << width << "x" << height << std::endl;
    std::cout << "  FPS: " << fps << std::endl;
    std::cout << "  Frames: " << total_frames << std::endl;
    std::cout << "\nControls:" << std::endl;
    std::cout << "  Mouse: Select target ROI" << std::endl;
    std::cout << "  q: Quit" << std::endl;
    std::cout << "  r: Reset (new target)" << std::endl;
    std::cout << "  Space: Pause/Resume" << std::endl;
    std::cout << "  s: Save tracked video" << std::endl;
    std::cout << std::endl;

    // Load tracker
    OSTrackTracker tracker;
    if (!tracker.load(engine_path, config)) {
        std::cerr << "[Demo] Failed to load tracker" << std::endl;
        return -1;
    }

    std::string win_name = "OSTrack Demo";
    cv::namedWindow(win_name, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::resizeWindow(win_name, std::min(960, width), std::min(720, height));

    bool paused = false;
    bool initialized = false;
    bool saving = false;
    cv::VideoWriter writer;
    std::string output_path;
    int frame_idx = 0;

    auto genOutputPath = [&]() {
        std::string dir = video_path.substr(0, video_path.find_last_of("/\\"));
        std::string base = video_path.substr(video_path.find_last_of("/\\") + 1);
        std::string name = base.substr(0, base.find_last_of('.'));
        output_path = dir + "/" + name + "_tracked.mp4";
    };

    while (true) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) {
            std::cout << "\nEnd of video reached." << std::endl;
            break;
        }

        // Pause mode
        if (paused) {
            cv::Mat disp = frame.clone();
            cv::putText(disp, "PAUSED (Space to continue)",
                        cv::Point(20, 30), cv::FONT_HERSHEY_COMPLEX_SMALL,
                        1.0, cv::Scalar(0, 255, 255), 2);
            cv::imshow(win_name, disp);

            int key = cv::waitKey(0) & 0xFF;
            if (key == ' ') paused = false;
            else if (key == 'q') goto cleanup;
            else if (key == 'r') {
                cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                initialized = false;
                tracker.reset();
            }
            continue;
        }

        // First frame: initialize
        if (!initialized) {
            std::cout << "Select the target to track..." << std::endl;
            BBox init_bbox = selectROI(win_name, frame);

            if (init_bbox.w < 5 || init_bbox.h < 5) {
                std::cout << "ROI too small, retrying..." << std::endl;
                continue;
            }

            std::cout << "Target: x=" << init_bbox.x << " y=" << init_bbox.y
                      << " w=" << init_bbox.w << " h=" << init_bbox.h << std::endl;

            if (!tracker.initialize(frame, init_bbox)) {
                std::cerr << "[Demo] Initialization failed" << std::endl;
                break;
            }

            initialized = true;
            frame_idx = 0;

            // Display initial frame with bbox
            cv::Mat disp = frame.clone();
            cv::rectangle(disp, cv::Rect(init_bbox.x, init_bbox.y,
                                          init_bbox.w, init_bbox.h),
                          cv::Scalar(0, 255, 0), 3);
            cv::imshow(win_name, disp);
            cv::waitKey(1);
            continue;
        }

        // Track
        double t0 = cv::getTickCount();
        BBox result = tracker.track(frame);
        double t1 = cv::getTickCount();
        double elapsed_ms = (t1 - t0) / cv::getTickFrequency() * 1000.0;

        // Draw result
        cv::Mat disp = frame.clone();
        cv::rectangle(disp, cv::Rect(result.x, result.y, result.w, result.h),
                      cv::Scalar(0, 255, 0), 3);

        std::string label = "x:" + std::to_string(static_cast<int>(result.x))
                          + " y:" + std::to_string(static_cast<int>(result.y))
                          + " w:" + std::to_string(static_cast<int>(result.w))
                          + " h:" + std::to_string(static_cast<int>(result.h));
        cv::putText(disp, label, cv::Point(result.x, result.y - 10),
                    cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, cv::Scalar(0, 255, 0), 2);

        cv::putText(disp, "Frame: " + std::to_string(frame_idx) + "/" + std::to_string(total_frames),
                    cv::Point(20, 30), cv::FONT_HERSHEY_COMPLEX_SMALL,
                    0.9, cv::Scalar(0, 255, 255), 2);
        cv::putText(disp, "Time: " + std::to_string(static_cast<int>(elapsed_ms)) + "ms"
                    + " (" + std::to_string(static_cast<int>(1000.0 / elapsed_ms)) + " FPS)",
                    cv::Point(20, 55), cv::FONT_HERSHEY_COMPLEX_SMALL,
                    0.9, cv::Scalar(0, 255, 255), 2);

        cv::imshow(win_name, disp);

        // Save if enabled
        if (saving) {
            if (!writer.isOpened()) {
                genOutputPath();
                int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
                writer.open(output_path, fourcc, fps, cv::Size(width, height));
                std::cout << "\nSaving to: " << output_path << std::endl;
            }
            if (writer.isOpened()) writer.write(disp);
        }

        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q') break;
        else if (key == 'r') {
            std::cout << "\nResetting..." << std::endl;
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            initialized = false;
            tracker.reset();
            if (writer.isOpened()) writer.release();
            saving = false;
        } else if (key == ' ') {
            paused = !paused;
        } else if (key == 's') {
            saving = !saving;
            std::cout << "Saving: " << (saving ? "ON" : "OFF") << std::endl;
        }

        frame_idx++;
    }

cleanup:
    cap.release();
    if (writer.isOpened()) writer.release();
    cv::destroyWindow(win_name);

    std::cout << "\nDone!" << std::endl;
    return 0;
}
