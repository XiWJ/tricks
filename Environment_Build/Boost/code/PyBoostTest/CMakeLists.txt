cmake_minimum_required(VERSION 3.21)
project(PyBoostTest)

# C++ 17
set(CMAKE_CXX_STANDARD 17)

# include
set(INCLUDE_H ${PROJECT_SOURCE_DIR}/inc)
set(INCLUDE_add "C:\\local\\include" 
                "D:\\ProgramData\\Anaconda3\\envs\\pyboost\\include")
include_directories(${INCLUDE_H} ${INCLUDE_add})

# lib
set(LIB_PATH ${PROJECT_SOURCE_DIR}/lib)
set(LIB_add "C:\\local\\lib" 
            "D:\\ProgramData\\Anaconda3\\envs\\pyboost\\libs")
file(GLOB_RECURSE LIB_FILES ${LIB_PATH}/*.lib ${LIB_add}/*.lib)
link_directories(${LIB_PATH} ${LIB_add})

message("*** LIB_PATH: ${LIB_PATH}")
message("*** LIB_FILES: ${LIB_FILES}")

# src
aux_source_directory(${PROJECT_SOURCE_DIR}/src DIR_SRCS)

# pyd
add_library(PyBoostTest ${DIR_SRCS})
target_link_libraries(PyBoostTest ${LIB_FILES})

# suffix
set (EXT ".so")
if (WIN32)
    set (EXT ".pyd")
endif()
set_target_properties(PyBoostTest PROPERTIES PREFIX "" SUFFIX ${EXT})