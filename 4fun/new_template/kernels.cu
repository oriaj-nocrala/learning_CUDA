#include "kernels.h"
#include "simulation_parameters.h"

// Kernel para actualizar la posición y velocidad de las partículas
__global__ void update_particles_kernel(Ball* balls, float dt, vec3 box_center) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUM_BALLS) return;

    // Aplicar gravedad
    balls[i].vel.y -= 9.8f * dt;

    // Actualizar posición
    balls[i].pos += balls[i].vel * dt;

    // Colisiones simples con las paredes
    if (balls[i].pos.y < box_center.y - BOX_SIZE + BALL_RADIUS) {
        balls[i].pos.y = box_center.y - BOX_SIZE + BALL_RADIUS;
        balls[i].vel.y *= -RESTITUTION;
    }
    if (balls[i].pos.x < box_center.x - BOX_SIZE + BALL_RADIUS) {
        balls[i].pos.x = box_center.x - BOX_SIZE + BALL_RADIUS;
        balls[i].vel.x *= -RESTITUTION;
    }
    if (balls[i].pos.x > box_center.x + BOX_SIZE - BALL_RADIUS) {
        balls[i].pos.x = box_center.x + BOX_SIZE - BALL_RADIUS;
        balls[i].vel.x *= -RESTITUTION;
    }
    if (balls[i].pos.z < box_center.z - BOX_SIZE + BALL_RADIUS) {
        balls[i].pos.z = box_center.z - BOX_SIZE + BALL_RADIUS;
        balls[i].vel.z *= -RESTITUTION;
    }
    if (balls[i].pos.z > box_center.z + BOX_SIZE - BALL_RADIUS) {
        balls[i].pos.z = box_center.z + BOX_SIZE - BALL_RADIUS;
        balls[i].vel.z *= -RESTITUTION;
    }
}
