#include <cuda_runtime.h>
#include "kernels.h"
#include "simulation_parameters.h"

// --- Implementaciones de los Kernels de CUDA ---

// Kernel para calcular el hash de la celda para cada pelota
__global__ void calculate_grid_hash_kernel(Ball* balls, unsigned int* grid_keys, unsigned int* grid_values, vec3 box_min) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUM_BALLS) return;

    vec3 pos = balls[i].pos;
    int grid_x = (int)((pos.x - box_min.x) / CELL_SIZE);
    int grid_y = (int)((pos.y - box_min.y) / CELL_SIZE);
    int grid_z = (int)((pos.z - box_min.z) / CELL_SIZE);

    // Clamp to grid dimensions
    grid_x = max(0, min(grid_x, GRID_DIM_X - 1));
    grid_y = max(0, min(grid_y, GRID_DIM_Y - 1));
    grid_z = max(0, min(grid_z, GRID_DIM_Z - 1));

    unsigned int hash = grid_z * (GRID_DIM_X * GRID_DIM_Y) + grid_y * GRID_DIM_X + grid_x;
    
    grid_keys[i] = hash;
    grid_values[i] = i;
}

// Kernel para encontrar los límites de cada celda en los arrays ordenados
__global__ void find_cell_boundaries_kernel(unsigned int* sorted_keys, int* cell_starts, int* cell_ends) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUM_BALLS) return;

    unsigned int current_key = sorted_keys[i];

    // Si la clave es diferente de la anterior, es el inicio de una nueva celda
    if (i == 0 || current_key != sorted_keys[i - 1]) {
        cell_starts[current_key] = i;
    }
    // Si la clave es diferente de la siguiente, es el final de una celda
    if (i == NUM_BALLS - 1 || current_key != sorted_keys[i + 1]) {
        cell_ends[current_key] = i + 1;
    }
}

// Kernel para calcular las fuerzas de colisión entre pelotas
__global__ void collide_kernel(Ball* balls, vec3* forces, unsigned int* sorted_values, int* cell_starts, int* cell_ends, vec3 box_center) {
    int i_sorted = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_sorted >= NUM_BALLS) return;

    int i_original = __ldg(&sorted_values[i_sorted]);
    vec3 pos1 = balls[i_original].pos;
    vec3 vel1 = balls[i_original].vel;

    int grid_x = (int)((pos1.x - (box_center.x - BOX_SIZE)) / CELL_SIZE);
    int grid_y = (int)((pos1.y - (box_center.y - BOX_SIZE)) / CELL_SIZE);
    int grid_z = (int)((pos1.z - (box_center.z - BOX_SIZE)) / CELL_SIZE);

    // Iterar sobre la celda actual y las 26 vecinas
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = grid_x + dx;
                int ny = grid_y + dy;
                int nz = grid_z + dz;

                if (nx < 0 || nx >= GRID_DIM_X || ny < 0 || ny >= GRID_DIM_Y || nz < 0 || nz >= GRID_DIM_Z) continue;

                unsigned int neighbor_hash = nz * (GRID_DIM_X * GRID_DIM_Y) + ny * GRID_DIM_X + nx;
                int start = __ldg(&cell_starts[neighbor_hash]);
                if (start == -1) continue; // Celda vacía
                int end = __ldg(&cell_ends[neighbor_hash]);

                // Comprobar colisiones con las pelotas de la celda vecina
                for (int j_sorted = start; j_sorted < end; ++j_sorted) {
                    int j_original = __ldg(&sorted_values[j_sorted]);
                    if (i_original >= j_original) continue; // Evitar dobles cálculos y auto-colisión

                    vec3 pos2 = balls[j_original].pos;
                    vec3 vel2 = balls[j_original].vel;

                    vec3 delta_pos = {pos1.x - pos2.x, pos1.y - pos2.y, pos1.z - pos2.z};
                    float dist_sq = dot(delta_pos, delta_pos);

                    if (dist_sq < (2 * BALL_RADIUS) * (2 * BALL_RADIUS) && dist_sq > 1e-9) {
                        float dist = sqrtf(dist_sq);
                        vec3 normal = {delta_pos.x / dist, delta_pos.y / dist, delta_pos.z / dist};
                        
                        // Fuerza de resorte (Hooke)
                        float penetration = (2 * BALL_RADIUS) - dist;
                        vec3 spring_force = {normal.x * STIFFNESS * penetration, normal.y * STIFFNESS * penetration, normal.z * STIFFNESS * penetration};

                        // Fuerza de amortiguación
                        vec3 rel_vel = {vel1.x - vel2.x, vel1.y - vel2.y, vel1.z - vel2.z};
                        float closing_vel = dot(rel_vel, normal);
                        vec3 damping_force = {normal.x * -DAMPING * closing_vel, normal.y * -DAMPING * closing_vel, normal.z * -DAMPING * closing_vel};

                        vec3 total_force = {spring_force.x + damping_force.x, spring_force.y + damping_force.y, spring_force.z + damping_force.z};

                        atomicAdd(&forces[i_original].x, total_force.x);
                        atomicAdd(&forces[i_original].y, total_force.y);
                        atomicAdd(&forces[i_original].z, total_force.z);

                        atomicAdd(&forces[j_original].x, -total_force.x);
                        atomicAdd(&forces[j_original].y, -total_force.y);
                        atomicAdd(&forces[j_original].z, -total_force.z);
                    }
                }
            }
        }
    }
}

// Kernel para integrar las fuerzas y actualizar posición y velocidad
__global__ void integrate_kernel(Ball* balls, vec3* forces, float dt, vec3 box_accel, vec3 box_center) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUM_BALLS) return;

    // Aplicar gravedad y fuerza de inercia de la caja
    vec3 total_force = forces[i];
    total_force.y -= 9.8f; // Gravedad (asumiendo masa = 1)
    total_force.x -= box_accel.x;
    total_force.y -= box_accel.y;
    total_force.z -= box_accel.z;

    // Actualizar velocidad (integración de Euler)
    balls[i].vel.x += total_force.x * dt;
    balls[i].vel.y += total_force.y * dt;
    balls[i].vel.z += total_force.z * dt;

    // Aplicar amortiguamiento lineal graduado (más fuerte para velocidades bajas)
    float speed = sqrtf(dot(balls[i].vel, balls[i].vel));
    float damping_factor = LINEAR_DAMPING;
    if (speed < 1.0f) {
        damping_factor *= (1.0f + (1.0f - speed) * 2.0f); // Más damping para velocidades bajas
    }
    balls[i].vel.x *= (1.0f - damping_factor * dt);
    balls[i].vel.y *= (1.0f - damping_factor * dt);
    balls[i].vel.z *= (1.0f - damping_factor * dt);

    // Umbral de reposo para simular fricción estática
    float speed_sq = dot(balls[i].vel, balls[i].vel);
    if (speed_sq < VELOCITY_STOP_THRESHOLD * VELOCITY_STOP_THRESHOLD) {
        balls[i].vel = {0.0f, 0.0f, 0.0f};
    }
    
    // Fricción adicional cuando las bolas están cerca del suelo
    if (balls[i].pos.y < box_center.y - BOX_SIZE + BALL_RADIUS * 1.5f) {
        balls[i].vel.x *= 0.99f;
        balls[i].vel.z *= 0.99f;
    }

    // Actualizar posición
    balls[i].pos.x += balls[i].vel.x * dt;
    balls[i].pos.y += balls[i].vel.y * dt;
    balls[i].pos.z += balls[i].vel.z * dt;

    // Colisiones con las paredes (con fricción)
    // Suelo
    if (balls[i].pos.y < box_center.y - BOX_SIZE + BALL_RADIUS) { 
        balls[i].pos.y = box_center.y - BOX_SIZE + BALL_RADIUS; 
        balls[i].vel.y *= -RESTITUTION;
        // Aplicar fricción extra en el suelo para ayudar al asentamiento
        if (abs(balls[i].vel.y) < 0.1f) {
            balls[i].vel.x *= 0.8f;
            balls[i].vel.z *= 0.8f;
        }
        balls[i].vel.x *= (1.0f - FRICTION);
        balls[i].vel.z *= (1.0f - FRICTION);
    }
    // Techo
    if (balls[i].pos.y > box_center.y + BOX_SIZE - BALL_RADIUS) { 
        balls[i].pos.y = box_center.y + BOX_SIZE - BALL_RADIUS; 
        balls[i].vel.y *= -RESTITUTION;
        balls[i].vel.x *= (1.0f - FRICTION);
        balls[i].vel.z *= (1.0f - FRICTION);
    }
    // Pared -X
    if (balls[i].pos.x < box_center.x - BOX_SIZE + BALL_RADIUS) { 
        balls[i].pos.x = box_center.x - BOX_SIZE + BALL_RADIUS; 
        balls[i].vel.x *= -RESTITUTION;
        balls[i].vel.y *= (1.0f - FRICTION);
        balls[i].vel.z *= (1.0f - FRICTION);
    }
    // Pared +X
    if (balls[i].pos.x > box_center.x + BOX_SIZE - BALL_RADIUS) { 
        balls[i].pos.x = box_center.x + BOX_SIZE - BALL_RADIUS; 
        balls[i].vel.x *= -RESTITUTION;
        balls[i].vel.y *= (1.0f - FRICTION);
        balls[i].vel.z *= (1.0f - FRICTION);
    }
    // Pared -Z
    if (balls[i].pos.z < box_center.z - BOX_SIZE + BALL_RADIUS) { 
        balls[i].pos.z = box_center.z - BOX_SIZE + BALL_RADIUS; 
        balls[i].vel.z *= -RESTITUTION;
        balls[i].vel.x *= (1.0f - FRICTION);
        balls[i].vel.y *= (1.0f - FRICTION);
    }
    // Pared +Z
    if (balls[i].pos.z > box_center.z + BOX_SIZE - BALL_RADIUS) { 
        balls[i].pos.z = box_center.z + BOX_SIZE - BALL_RADIUS; 
        balls[i].vel.z *= -RESTITUTION;
        balls[i].vel.x *= (1.0f - FRICTION);
        balls[i].vel.y *= (1.0f - FRICTION);
    }

    // Limpiar fuerzas para el siguiente paso
    forces[i] = {0.0f, 0.0f, 0.0f};
}

__global__ void apply_impulse_kernel(Ball* balls, int ball_index, vec3 impulse) {
    if (threadIdx.x == 0 && blockIdx.x == 0) { // Solo un hilo necesita hacer esto
        balls[ball_index].vel.x += impulse.x;
        balls[ball_index].vel.y += impulse.y;
        balls[ball_index].vel.z += impulse.z;
    }
}

__global__ void apply_area_force_kernel(Ball* balls, vec3 force_center, vec3 force_direction, float force_radius, float force_strength) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUM_BALLS) return;

    // Calcular distancia desde el centro de la fuerza
    vec3 diff = {balls[i].pos.x - force_center.x, balls[i].pos.y - force_center.y, balls[i].pos.z - force_center.z};
    float distance = sqrtf(dot(diff, diff));

    // Solo aplicar fuerza si está dentro del radio
    if (distance <= force_radius && distance > 0.01f) {
        // Calcular intensidad de la fuerza basada en la distancia (más fuerte cerca del centro)
        float intensity = 1.0f - (distance / force_radius);
        intensity = intensity * intensity; // Cuadrática para efecto más pronunciado
        
        // Aplicar la fuerza en la dirección especificada
        vec3 force = {force_direction.x * force_strength * intensity, 
                      force_direction.y * force_strength * intensity, 
                      force_direction.z * force_strength * intensity};
        
        balls[i].vel.x += force.x;
        balls[i].vel.y += force.y;
        balls[i].vel.z += force.z;
    }
}
