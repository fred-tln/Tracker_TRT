cmake_minimum_required(VERSION 3.18)
project(ostrack_demo)# LANGUAGES CXX CUDA)

# =============================================================================
# C++ standard
# =============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CUDA_STANDARD 17)
#set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# =============================================================================
# TensorRT
# =============================================================================
set(TENSORRT_ROOT "/usr/local/TensorRT-8.6.1.6" CACHE PATH "TensorRT installation path")

include_directories(
    ${TENSORRT_ROOT}/include
)

link_directories(
    ${TENSORRT_ROOT}/lib
)

# =============================================================================
# CUDA
# =============================================================================
#find_package(CUDA REQUIRED)

#include_directories(${CUDA_INCLUDE_DIRS}) 

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
enable_language(CUDA)

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)


# =============================================================================
# OpenCV
# =============================================================================
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc videoio)
include_directories(${OpenCV_INCLUDE_DIRS})

# =============================================================================
# Source files
# =============================================================================
set(SOURCES
    engine.cpp
    tracker.cpp
    demo.cpp
    main.cpp
    fp16_utils.cu
)

# =============================================================================
# Executable
# =============================================================================
add_executable(ostrack_demo ${SOURCES})

target_include_directories(ostrack_demo PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(ostrack_demo
    nvinfer
    nvonnxparser
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
)

# =============================================================================
# Compile options
# =============================================================================
target_compile_options(ostrack_demo PRIVATE
    -O3
    -DNOMINMAX
)

set_target_properties(ostrack_demo PROPERTIES
    CUDA_ARCHITECTURES "70;75;80;86"
)

# =============================================================================
# Install
# =============================================================================
install(TARGETS ostrack_demo DESTINATION bin)

