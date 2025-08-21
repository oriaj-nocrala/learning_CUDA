#include "kernels.h"
#include "hair.h"
#include "simulation_parameters.h"
#include "glm_helper.cuh"

__global__ void update_hair_kernel(Hair* hairs, vec3* vbo_ptr, float dt, vec3 sphere_center, vec3 sphere_velocity, vec3 interaction_sphere_center, bool interaction_active) {
    extern __shared__ Hair local_hairs[];

    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    int local_id = threadIdx.x;

    // Cargar datos desde memoria global a compartida
    if (global_id < NUM_HAIRS) {
        local_hairs[local_id] = hairs[global_id];
    }
    __syncthreads();

    if (global_id >= NUM_HAIRS) return;

    float segment_length = HAIR_LENGTH / (HAIR_SEGMENTS > 1 ? (HAIR_SEGMENTS - 1) : 1);
    float stiffness = 0.8f;

    // 1. Integración Verlet y gravedad (operando en memoria compartida)
    for (int j = 0; j < HAIR_SEGMENTS; ++j) {
        vec3 temp = local_hairs[local_id].pos[j];
        local_hairs[local_id].pos[j] += (local_hairs[local_id].pos[j] - local_hairs[local_id].old_pos[j]) * 0.98f + vec3(0.0f, -1.5f, 0.0f) * dt * dt;
        local_hairs[local_id].old_pos[j] = temp;
    }

    // 2. Solucionar restricciones (operando en memoria compartida)
    for (int k = 0; k < 15; ++k) {
        local_hairs[local_id].pos[0] = sphere_center + local_hairs[local_id].anchor_direction * SPHERE_RADIUS;

        for (int j = 0; j < HAIR_SEGMENTS - 1; ++j) {
            vec3& p1 = local_hairs[local_id].pos[j];
            vec3& p2 = local_hairs[local_id].pos[j + 1];
            
            vec3 delta = p2 - p1;
            float dist = length(delta);
            if (dist > 1e-6) {
                float diff = (dist - segment_length) / dist;
                p2 -= delta * 0.5f * diff;
                if (j > 0) {
                    p1 += delta * 0.5f * diff;
                }
            }

            vec3 target_dir = (j == 0) ? local_hairs[local_id].anchor_direction : normalize(p1 - local_hairs[local_id].pos[j-1]);
            vec3 current_dir = normalize(p2 - p1);
            vec3 new_dir = normalize(current_dir + target_dir * stiffness);
            local_hairs[local_id].pos[j+1] = p1 + new_dir * segment_length;
        }

        for (int j = 1; j < HAIR_SEGMENTS; ++j) {
            vec3 seg_to_center = local_hairs[local_id].pos[j] - sphere_center;
            float dist_to_center = length(seg_to_center);
            if (dist_to_center < SPHERE_RADIUS) {
                vec3 normal = seg_to_center / dist_to_center;
                local_hairs[local_id].pos[j] = sphere_center + normal * SPHERE_RADIUS;
            }
        }

        if (interaction_active) {
            for (int j = 0; j < HAIR_SEGMENTS; ++j) {
                vec3 seg_to_interaction = local_hairs[local_id].pos[j] - interaction_sphere_center;
                float dist_to_interaction = length(seg_to_interaction);
                if (dist_to_interaction < INTERACTION_SPHERE_RADIUS) {
                    vec3 normal = seg_to_interaction / dist_to_interaction;
                    local_hairs[local_id].pos[j] = interaction_sphere_center + normal * INTERACTION_SPHERE_RADIUS;
                }
            }
        }
        __syncthreads(); // Sincronizar después de cada iteración de restricciones
    }

    // 3. Actualizar velocidad y escribir en el VBO
    for (int j = 0; j < HAIR_SEGMENTS; ++j) {
        local_hairs[local_id].vel[j] = (local_hairs[local_id].pos[j] - local_hairs[local_id].old_pos[j]) / dt;
        vbo_ptr[global_id * HAIR_SEGMENTS + j] = local_hairs[local_id].pos[j];
    }
    local_hairs[local_id].vel[0] += sphere_velocity;

    // Escribir los datos de vuelta a la memoria global
    if (global_id < NUM_HAIRS) {
        hairs[global_id] = local_hairs[local_id];
    }
}
