cmake_minimum_required(VERSION 3.21)
project(ChernoCpp)

# C++ 17
set(CMAKE_CXX_STANDARD 17)

# include
# set(INCLUDE_H ${PROJECT_SOURCE_DIR}/include)
# include_directories(${INCLUDE_H})

# lib
# set(LIB_PATH ${PROJECT_SOURCE_DIR}/lib)
# file(GLOB_RECURSE LIB_FILES ${LIB_PATH}/*.lib)
# link_directories(${LIB_PATH})

# message("*** LIB_PATH: ${LIB_PATH}")
# message("*** LIB_FILES: ${LIB_FILES}")

# src
aux_source_directory(${PROJECT_SOURCE_DIR}/ DIR_SRCS)

# sub dir
add_subdirectory(${PROJECT_SOURCE_DIR}/36_init_constructor)
add_subdirectory(${PROJECT_SOURCE_DIR}/43_object_life)

# exe
# add_executable(Test ${DIR_SRCS})
# target_link_libraries(Test ${LIB_FILES})