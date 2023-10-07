# project name
get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${PROJECT_NAME})

# src
aux_source_directory(. SUB_DIR_SRCS)

# exe
add_executable(${PROJECT_NAME} ${SUB_DIR_SRCS})