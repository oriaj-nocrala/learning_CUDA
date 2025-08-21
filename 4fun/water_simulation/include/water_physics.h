#pragma once

#include "common.h"

class WaterPhysics {
public:
    // Constructor
    WaterPhysics();
    ~WaterPhysics();

    // Initialize GPU memory
    void initialize();
    void cleanup();

    // Simulation steps
    void simulate_step();
    void add_density_source(int x, int y, float amount);
    void add_velocity_source(int x, int y, float vx, float vy);
    void clear_sources();

    // Get density data for rendering
    float* get_density_data() const { return d_dens; }

private:
    // GPU memory buffers
    float *d_dens, *d_dens_prev;
    float *d_u, *d_v, *d_u_prev, *d_v_prev;

    // Core simulation functions
    void dens_step(float *x, float *x0, float *u, float *v, float diff, float dt);
    void vel_step(float *u, float *v, float *u_prev, float *v_prev, float visc, float dt);
    
    // Helper functions
    void set_bnd(int b, float *x);
    void lin_solve(int b, float *x, float *x0, float a, float c, int iter);
    void diffuse(int b, float *x, float *x0, float diff, float dt, int iter);
    void advect(int b, float *d, float *d0, float *u, float *v, float dt);
    void project(float *u, float *v, float *p, float *div);
};