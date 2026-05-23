/**
 * @file main.cpp
 * @brief Entry point for OSTrack TensorRT demo
 *
 * Usage:
 *   ./ostrack_demo <video_path> [engine_path]
 *
 * If engine_path is not specified, it defaults to ostrack_384_fp16.engine
 * in the project root directory.
 */

#include "ostrack_trt.h"
#include <iostream>
#include <string>
#include <cstring>

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " <video_path> [engine_path]" << std::endl;
    std::cout << "\nArguments:" << std::endl;
    std::cout << "  video_path   Path to input video file" << std::endl;
    std::cout << "  engine_path  Path to TensorRT engine (default: ostrack_384_fp16.engine)" << std::endl;
    std::cout << "\nControls:" << std::endl;
    std::cout << "  Mouse  - Select target ROI on first frame" << std::endl;
    std::cout << "  q      - Quit" << std::endl;
    std::cout << "  r      - Reset (select new target)" << std::endl;
    std::cout << "  Space  - Pause / Resume" << std::endl;
    std::cout << "  s      - Toggle saving output video" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printUsage(argv[0]);
        return -1;
    }

    std::string video_path = argv[1];
    std::string engine_path = (argc >= 3) ? argv[2] : "ostrack_384_fp16.engine";

    std::cout << "OSTrack TensorRT Demo" << std::endl;
    std::cout << "=====================" << std::endl;
    std::cout << "Video:  " << video_path << std::endl;
    std::cout << "Engine: " << engine_path << std::endl;
    std::cout << std::endl;

    // Default configuration for ostrack_384 model
    OSTrackConfig config;
    config.template_factor = 2.0f;
    config.template_size = 128;
    config.search_factor = 5.0f;
    config.search_size = 384;
    config.backbone_stride = 16;

    return runTrackingDemo(video_path, engine_path, config);
}
