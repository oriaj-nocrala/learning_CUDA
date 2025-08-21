cmake_minimum_required(VERSION 3.12)
project(SPH-CUDA LANGUAGES CXX CUDA C)

# Set build type to Release for performance
set(CMAKE_BUILD_TYPE Release)

# Find CUDA
find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLFW REQUIRED glfw3)

# Set CUDA architectures for Ampere (RTX 30-series)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Set include directories
include_directories(
    .
    glad
    libs
    KHR
    ${GLFW_INCLUDE_DIRS}
)

# Add the executable
add_executable(sph-cuda
    main.cu
    visualizer.cpp
    glad.c
    src/box.cpp
    src/sphere.cpp
    src/MarchingCubes/CudaGrid.cpp
    src/MarchingCubes/CudaGrid.cu
    src/MarchingCubes/skel/MarchingCubes.cpp
    src/MarchingCubes/skel/MarchingCubes.cu
    src/Renderer/AttributeBufferLayout.cpp
    src/Renderer/IndexBuffer.cpp
    src/Renderer/Renderer.cpp
    src/Renderer/VertexArrayObject.cpp
    src/Renderer/VertexBuffer.cpp
    src/Shaders/Shader.cpp
)

# Link libraries
target_link_libraries(sph-cuda
    ${CUDA_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${GLFW_LIBRARIES}
)

# Set CUDA specific options
set_target_properties(sph-cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)