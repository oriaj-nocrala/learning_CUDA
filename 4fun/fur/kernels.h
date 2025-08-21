#pragma once

#include "hair.h"

// --- Declaraciones de los Kernels de CUDA ---
__global__ void update_hair_kernel(Hair* hairs, vec3* vbo_ptr, float dt, vec3 sphere_center, vec3 sphere_velocity, vec3 interaction_sphere_center, bool interaction_active);
