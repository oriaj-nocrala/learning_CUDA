#pragma once

#include "ball.h"

// --- Declaraciones de los Kernels de CUDA ---
__global__ void calculate_grid_hash_kernel(Ball* balls, unsigned int* grid_keys, unsigned int* grid_values, vec3 box_min);
__global__ void find_cell_boundaries_kernel(unsigned int* sorted_keys, int* cell_starts, int* cell_ends);
__global__ void collide_kernel(Ball* balls, vec3* forces, unsigned int* sorted_values, int* cell_starts, int* cell_ends, vec3 box_center);
__global__ void integrate_kernel(Ball* balls, vec3* forces, float dt, vec3 box_accel, vec3 box_center);
__global__ void apply_impulse_kernel(Ball* balls, int ball_index, vec3 impulse);
__global__ void apply_area_force_kernel(Ball* balls, vec3 force_center, vec3 force_direction, float force_radius, float force_strength);
