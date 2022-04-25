# sub dir name
set(SUB_DIR_NAME "02_Hello_Triangle")
message("*** SUB_DIR_NAME: ${SUB_DIR_NAME}")

# src
aux_source_directory(CMAKE_CURRENT_SOURCE_DIR DIR_SRCS)

# exe
add_executable(${SUB_DIR_NAME} ${DIR_SRCS})
target_link_libraries(${SUB_DIR_NAME} ${LIB_FILES})