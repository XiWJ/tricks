# project name
get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${PROJECT_NAME})

# src
aux_source_directory(. SUB_DIR_SRCS)

# glad.c
set(GLAD_C ../glad.c)

# exe
add_executable(${PROJECT_NAME} ${SUB_DIR_SRCS} ${GLAD_C})
target_link_libraries(${PROJECT_NAME} ${LIB_FILES})