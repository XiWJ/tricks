cmake_minimum_required(VERSION 3.4...3.18)
project(example)

# C++ 17
set(CMAKE_CXX_STANDARD 17)

# include
set(INCLUDE_H ${CMAKE_CURRENT_SOURCE_DIR}/inc)
include_directories(${INCLUDE_H})

# src
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src DIR_SRCS)

# pybind11
add_subdirectory(pybind11) # 包含子目录 pybind11 https://github.com/pybind/pybind11.git
pybind11_add_module(example ${DIR_SRCS})

# version setup
set(EXAMPLE_VERSION_INFO 1.0) # 设定版本号
target_compile_definitions(example
                          PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})