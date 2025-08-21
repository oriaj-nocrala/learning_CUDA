#include "../include/water_physics.h"

// Declare external CUDA kernel wrappers
extern void launch_add_source_kernel(float *x, const float *s, float dt, int size);
extern void launch_set_bnd_kernel(int b, float *x);
extern void launch_lin_solve_kernel(int b, float *x, const float *x0, float a, float c);
extern void launch_advect_kernel(int b, float *d, const float *d0, const float *u, const float *v, float dt);
extern void launch_project_step1(float *div, float *p, const float *u, const float *v);
extern void launch_project_step2(float *u, float *v, const float *p);

WaterPhysics::WaterPhysics() : d_dens(nullptr), d_dens_prev(nullptr), 
                               d_u(nullptr), d_v(nullptr), 
                               d_u_prev(nullptr), d_v_prev(nullptr) {
}

WaterPhysics::~WaterPhysics() {
    cleanup();
}

void WaterPhysics::initialize() {
    size_t bytes = (N+2) * (N+2) * sizeof(float);
    
    CHECK_CUDA(cudaMalloc(&d_dens, bytes));
    CHECK_CUDA(cudaMalloc(&d_dens_prev, bytes));
    CHECK_CUDA(cudaMalloc(&d_u, bytes));
    CHECK_CUDA(cudaMalloc(&d_u_prev, bytes));
    CHECK_CUDA(cudaMalloc(&d_v, bytes));
    CHECK_CUDA(cudaMalloc(&d_v_prev, bytes));
    
    CHECK_CUDA(cudaMemset(d_dens, 0, bytes));
    CHECK_CUDA(cudaMemset(d_dens_prev, 0, bytes));
    CHECK_CUDA(cudaMemset(d_u, 0, bytes));
    CHECK_CUDA(cudaMemset(d_u_prev, 0, bytes));
    CHECK_CUDA(cudaMemset(d_v, 0, bytes));
    CHECK_CUDA(cudaMemset(d_v_prev, 0, bytes));
    
    // Initialize with some water at the bottom
    float* h_temp = new float[(N+2)*(N+2)];
    memset(h_temp, 0, bytes);
    
    // Add more water at the bottom of the simulation
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N/3; j++) {
            h_temp[IX(i, j)] = 200.0f; // Increased initial water density
        }
    }
    CHECK_CUDA(cudaMemcpy(d_dens, h_temp, bytes, cudaMemcpyHostToDevice));
    
    delete[] h_temp;
}

void WaterPhysics::cleanup() {
    if (d_dens) { cudaFree(d_dens); d_dens = nullptr; }
    if (d_dens_prev) { cudaFree(d_dens_prev); d_dens_prev = nullptr; }
    if (d_u) { cudaFree(d_u); d_u = nullptr; }
    if (d_u_prev) { cudaFree(d_u_prev); d_u_prev = nullptr; }
    if (d_v) { cudaFree(d_v); d_v = nullptr; }
    if (d_v_prev) { cudaFree(d_v_prev); d_v_prev = nullptr; }
}

void WaterPhysics::simulate_step() {
    vel_step(d_u, d_v, d_u_prev, d_v_prev, VISC, DT);
    dens_step(d_dens, d_dens_prev, d_u, d_v, DIFF, DT);
}

void WaterPhysics::add_density_source(int x, int y, float amount) {
    if (x > 0 && x <= N && y > 0 && y <= N) {
        float h_value = amount;
        CHECK_CUDA(cudaMemcpy(&d_dens_prev[IX(x, y)], &h_value, sizeof(float), cudaMemcpyHostToDevice));
    }
}

void WaterPhysics::add_velocity_source(int x, int y, float vx, float vy) {
    if (x > 0 && x <= N && y > 0 && y <= N) {
        CHECK_CUDA(cudaMemcpy(&d_u_prev[IX(x, y)], &vx, sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(&d_v_prev[IX(x, y)], &vy, sizeof(float), cudaMemcpyHostToDevice));
    }
}

void WaterPhysics::clear_sources() {
    size_t bytes = (N+2) * (N+2) * sizeof(float);
    CHECK_CUDA(cudaMemset(d_dens_prev, 0, bytes));
    CHECK_CUDA(cudaMemset(d_u_prev, 0, bytes));
    CHECK_CUDA(cudaMemset(d_v_prev, 0, bytes));
}

void WaterPhysics::set_bnd(int b, float *x) {
    launch_set_bnd_kernel(b, x);
}

void WaterPhysics::lin_solve(int b, float *x, float *x0, float a, float c, int iter) {
    for (int k = 0; k < iter; k++) {
        launch_lin_solve_kernel(b, x, x0, a, c);
        set_bnd(b, x);
    }
}

void WaterPhysics::diffuse(int b, float *x, float *x0, float diff, float dt, int iter) {
    float a = dt * diff * N * N;
    lin_solve(b, x, x0, a, 1 + 4 * a, iter);
}

void WaterPhysics::advect(int b, float *d, float *d0, float *u, float *v, float dt) {
    launch_advect_kernel(b, d, d0, u, v, dt);
    set_bnd(b, d);
}

void WaterPhysics::project(float *u, float *v, float *p, float *div) {
    launch_project_step1(div, p, u, v);
    set_bnd(0, div); 
    set_bnd(0, p);
    lin_solve(0, p, div, 1, 4, 20);
    launch_project_step2(u, v, p);
    set_bnd(1, u); 
    set_bnd(2, v);
}

void WaterPhysics::dens_step(float *x, float *x0, float *u, float *v, float diff, float dt) {
    int size = (N + 2) * (N + 2);
    launch_add_source_kernel(x, x0, dt, size);
    SWAP(x0, x); 
    diffuse(0, x, x0, diff, dt, 20);
    SWAP(x0, x); 
    advect(0, x, x0, u, v, dt);
}

void WaterPhysics::vel_step(float *u, float *v, float *u_prev, float *v_prev, float visc, float dt) {
    int size = (N + 2) * (N + 2);
    launch_add_source_kernel(u, u_prev, dt, size);
    launch_add_source_kernel(v, v_prev, dt, size);
    SWAP(u_prev, u); 
    diffuse(1, u, u_prev, visc, dt, 20);
    SWAP(v_prev, v); 
    diffuse(2, v, v_prev, visc, dt, 20);
    project(u, v, u_prev, v_prev);
    SWAP(u_prev, u); 
    SWAP(v_prev, v);
    advect(1, u, u_prev, u_prev, v_prev, dt);
    advect(2, v, v_prev, u_prev, v_prev, dt);
    project(u, v, u_prev, v_prev);
}