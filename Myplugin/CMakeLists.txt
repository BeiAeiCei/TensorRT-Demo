cmake_minimum_required(VERSION 3.18)
project(MyPlugin LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 60 61 70 75 80 86)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


# TensorRT
set(TENSORRT_ROOT $ENV{TENSORRT_ROOT})
set(TENSORRT_INCLUDE_DIR ${TENSORRT_ROOT}/include)
set(TENSORRT_LIB_DIR ${TENSORRT_ROOT}/lib)


set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.4)

find_package(CUDA REQUIRED)
#find_package(pybind11 REQUIRED)

include_directories(
    ${TENSORRT_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

file(GLOB_RECURSE SRC_FILES
    "Add/*.cpp"
    "Add/*.cu"
    "LayerNorm/*.cpp"
    "LayerNorm/*.cu"
)


add_library(MyPlugin SHARED ${SRC_FILES})

set_target_properties(MyPlugin PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(MyPlugin
    ${TENSORRT_LIB_DIR}/libnvinfer.so
    ${TENSORRT_LIB_DIR}/libnvinfer_plugin.so
    cudart
)
target_link_options(MyPlugin PRIVATE -Wl,--no-as-needed)
# cmake .. -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc