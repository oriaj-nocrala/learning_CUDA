#pragma once

#include "common.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Real Physical Constants (SI Units)
const int MAX_PARTICLES = 8192;

// Simulation Scale: 1 unit = 1 meter in simulation domain
const float SIMULATION_SCALE = 1.0f;    // 1 meter simulation domain
const float PARTICLE_RADIUS = 0.02f;    // 20mm particle radius (based on reference)
const float SMOOTHING_LENGTH = 0.14f;   // h = 0.14 (based on reference - much larger interaction)

// Physically-inspired but numerically stable parameters
const float WATER_DENSITY = 1000.0f;     // kg/m³ (reference)
const float GRAVITY_MAGNITUDE = 9.81f;   // m/s² (real gravity)

// SPH Parameters (based on reference simulation)
const float PARTICLE_VOLUME = (4.0f/3.0f) * M_PI * PARTICLE_RADIUS * PARTICLE_RADIUS * PARTICLE_RADIUS;
const float PARTICLE_MASS = 0.0007f;                                   // Mass from reference
const float REST_DENSITY = 1.4f;                                       // p0 from reference
const float DYNAMIC_VISCOSITY = 0.05f;                                 // e from reference
const float GRAVITY = -9.81f;                                          // Full gravity like reference

// Numerical stability parameters
const float GAS_CONSTANT = 0.5f;         // k from reference (much lower stiffness)
const float BOUNDARY_DAMPING = 0.3f;     // boundary_damping from reference
const float TIME_STEP = 0.005f;          // timestep from reference

// SPH Kernel constants (precalculated like reference) - device compatible
#define CONST_POLY6 (315.0f / (64.0f * M_PI * powf(SMOOTHING_LENGTH, 9)))
#define CONST_SPIKY (-45.0f / (M_PI * powf(SMOOTHING_LENGTH, 6)))
#define CONST_VISC (15.0f / (2.0f * M_PI * powf(SMOOTHING_LENGTH, 3)))

// Spatial hashing
const int GRID_SIZE = 64;
const float CELL_SIZE = SMOOTHING_LENGTH;

// Particle structure
struct Particle {
    float2 position;
    float2 velocity;
    float2 force;
    float density;
    float pressure;
    
    __host__ __device__ Particle() : 
        position(make_float2(0,0)), 
        velocity(make_float2(0,0)), 
        force(make_float2(0,0)), 
        density(0), 
        pressure(0) {}
};

// Grid cell for spatial hashing
struct GridCell {
    int start;
    int count;
    
    __host__ __device__ GridCell() : start(-1), count(0) {}
};

// SPH Kernel functions
__device__ float poly6_kernel(float r, float h);
__device__ float2 spiky_kernel_gradient(float2 r, float h);
__device__ float viscosity_kernel_laplacian(float r, float h);

// Utility functions
__device__ __host__ inline int2 get_grid_pos(float2 pos) {
    return make_int2((int)(pos.x / CELL_SIZE), (int)(pos.y / CELL_SIZE));
}

__device__ __host__ inline int grid_hash(int2 grid_pos) {
    // Ensure positive hash values
    int hash = ((grid_pos.x * 73856093) ^ (grid_pos.y * 19349663));
    hash = hash & 0x7FFFFFFF; // Make positive
    return hash % (GRID_SIZE * GRID_SIZE);
}