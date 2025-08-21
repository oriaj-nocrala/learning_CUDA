#pragma once

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// CUDA error checking macro
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Water simulation parameters
const int N = 128;
const int WIN_SIZE = 768;
const float DT = 0.04f;
const float DIFF = 0.0005f;  // Increased diffusion for more visible spread
const float VISC = 0.0005f;  // Reduced viscosity for more fluid movement

// Helper macros and functions
#define SWAP(x0,x) do{ float* tmp=x0; x0=x; x=tmp; }while(0)
__device__ __host__ inline int IX(int x, int y) { return y * (N + 2) + x; }