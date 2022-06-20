cmake_minimum_required(VERSION 3.21)
project(Test)

# C++ 17
set(CMAKE_CXX_STANDARD 17)

# include
set(INCLUDE_H ${PROJECT_SOURCE_DIR}/include)
include_directories(${INCLUDE_H})

# lib
set(LIB_PATH ${PROJECT_SOURCE_DIR}/lib)
file(GLOB_RECURSE LIB_FILES ${LIB_PATH}/*.lib)
link_directories(${LIB_PATH})

message("*** LIB_PATH: ${LIB_PATH}")
message("*** LIB_FILES: ${LIB_FILES}")

# src
aux_source_directory(${PROJECT_SOURCE_DIR}/src DIR_SRCS)

# sub dir
add_subdirectory(${PROJECT_SOURCE_DIR}/src/00_Mini_Render)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/01_Hello_Window)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/02_Hello_Triangle)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/03_Shaders)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/04_Textures)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/05_Transformations)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/06_Coordinate_Systems)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/07_Camera)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/08_Colors)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/09_Basic_Lighting)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/10_Materials)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/11_Lighting_maps)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/13_Multiple_lights)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/15_Depth_testing)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/16_discard)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/17_Blending)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/18_Face_Culling)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/19_FrameBuffer)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/99_Depth_Peeling)
add_subdirectory(${PROJECT_SOURCE_DIR}/src/20_Cubemaps)

# exe
# add_executable(Test ${DIR_SRCS})
# target_link_libraries(Test ${LIB_FILES})