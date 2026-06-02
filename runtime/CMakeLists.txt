cmake_minimum_required(VERSION 3.20)
project(opacc1ty_runtime
    VERSION 0.1.0
    DESCRIPTION "Opacc1ty C runtime for 2-bit quantized inference on Apple Silicon"
    LANGUAGES C OBJC
)

set(CMAKE_C_STANDARD 17)
set(CMAKE_OBJC_STANDARD 17)

# macOS-specific settings
if(APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum macOS version")
endif()

# Find required frameworks
find_library(METAL_FRAMEWORK Metal REQUIRED)
find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)

# Source files
set(SOURCES
    src/bf2_format.c
    src/inference.c
)

set(OBJC_SOURCES
    src/metal_backend.m
)

# Library target
add_library(opacc1ty SHARED
    ${SOURCES}
    ${OBJC_SOURCES}
)

target_include_directories(opacc1ty
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(opacc1ty
    PUBLIC
        ${METAL_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
)

# Compiler flags
target_compile_options(opacc1ty PRIVATE
    -Wall -Wextra -Wpedantic
    -O3 -march=armv8.5-a
    -ffast-math
)

# Linker flags
target_link_options(opacc1ty PRIVATE
    -Wl,-dead_strip
)

# Install
install(TARGETS opacc1ty
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)

# Example
add_executable(example_simple examples/simple_inference.c)
target_link_libraries(example_simple PRIVATE opacc1ty)

# Tests (if CTest is enabled)
if(BUILD_TESTING)
    enable_testing()
    add_executable(test_bf2_format tests/test_bf2_format.c)
    target_link_libraries(test_bf2_format PRIVATE opacc1ty)
    add_test(NAME bf2_format COMMAND test_bf2_format)
endif()

# Print configuration summary
message(STATUS "Opacc1ty Runtime v${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Metal framework: ${METAL_FRAMEWORK}")
