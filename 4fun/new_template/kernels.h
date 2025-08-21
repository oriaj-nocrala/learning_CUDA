#pragma once

#include "ball.h"

// --- Declaraciones de los Kernels de CUDA ---
__global__ void update_particles_kernel(Ball* balls, float dt, vec3 box_center);
