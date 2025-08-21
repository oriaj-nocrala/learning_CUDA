#include "../include/sph_common.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// SPH Kernel functions implementation (using precalculated constants)
__device__ float poly6_kernel(float r, float h) {
    if (r >= h) return 0.0f;
    
    float h2 = h * h;
    float diff = h2 - r * r;
    
    return CONST_POLY6 * diff * diff * diff;
}

__device__ float2 spiky_kernel_gradient(float2 r, float h) {
    float r_len = sqrtf(r.x * r.x + r.y * r.y);
    if (r_len >= h || r_len == 0.0f) return make_float2(0.0f, 0.0f);
    
    float diff = h - r_len;
    float coeff = CONST_SPIKY * diff * diff / r_len;
    
    return make_float2(coeff * r.x, coeff * r.y);
}

__device__ float viscosity_kernel_laplacian(float r, float h) {
    if (r >= h) return 0.0f;
    
    return CONST_VISC * (h - r);
}

// Reset grid kernel
__global__ void reset_grid_kernel(GridCell* grid, int grid_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= grid_size) return;
    
    grid[idx].start = -1;
    grid[idx].count = 0;
}

// Compute grid hash for each particle
__global__ void compute_hash_kernel(Particle* particles, int* hashes, int* indices, int num_particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    
    int2 grid_pos = get_grid_pos(particles[idx].position);
    hashes[idx] = grid_hash(grid_pos);
    indices[idx] = idx;
}

// Reorder particles based on hash
__global__ void reorder_particles_kernel(
    Particle* particles_in, Particle* particles_out,
    int* sorted_indices, int num_particles) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    
    int sorted_idx = sorted_indices[idx];
    particles_out[idx] = particles_in[sorted_idx];
}

// Find start of each cell in sorted array
__global__ void find_cell_start_kernel(
    int* hashes, GridCell* grid, int num_particles) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    
    int hash = hashes[idx];
    int prev_hash = (idx == 0) ? -1 : hashes[idx - 1];
    
    if (hash != prev_hash) {
        grid[hash].start = idx;
    }
    
    int next_hash = (idx == num_particles - 1) ? -1 : hashes[idx + 1];
    if (hash != next_hash) {
        grid[hash].count = idx - grid[hash].start + 1;
    }
}

// Compute density and pressure using spatial hashing
__global__ void compute_density_pressure_kernel(
    Particle* particles, GridCell* grid, int num_particles) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    
    Particle& p = particles[idx];
    float2 pos = p.position;
    float density = 0.0f;
    
    // Get grid cell for this particle
    int2 grid_pos = get_grid_pos(pos);
    
    // Search in 3x3 neighboring cells
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            int2 neighbor_pos = make_int2(grid_pos.x + x, grid_pos.y + y);
            
            // Check bounds
            if (neighbor_pos.x < 0 || neighbor_pos.x >= GRID_SIZE ||
                neighbor_pos.y < 0 || neighbor_pos.y >= GRID_SIZE) continue;
            
            int hash = grid_hash(neighbor_pos);
            if (hash < 0 || hash >= GRID_SIZE * GRID_SIZE) continue;
            
            GridCell& cell = grid[hash];
            
            if (cell.start == -1 || cell.count <= 0) continue;
            
            // Check all particles in this cell
            for (int i = 0; i < cell.count; i++) {
                int neighbor_idx = cell.start + i;
                if (neighbor_idx < 0 || neighbor_idx >= num_particles) break;
                
                Particle& neighbor = particles[neighbor_idx];
                float2 diff = make_float2(pos.x - neighbor.position.x, 
                                        pos.y - neighbor.position.y);
                float r = sqrtf(diff.x * diff.x + diff.y * diff.y);
                
                if (r < SMOOTHING_LENGTH) {
                    density += PARTICLE_MASS * poly6_kernel(r, SMOOTHING_LENGTH);
                }
            }
        }
    }
    
    p.density = fmaxf(density, REST_DENSITY * 0.01f); // Allow some compression
    
    // Tait equation of state (same as reference) - much more stable
    // P = (k * p0 / 7) * ((ρ/ρ₀)^7 - 1)
    float density_ratio = p.density / REST_DENSITY;
    p.pressure = fmaxf((GAS_CONSTANT * REST_DENSITY / 7.0f) * (powf(density_ratio, 7.0f) - 1.0f), 0.0f);
    
    // Clamp pressure to avoid extreme values
    p.pressure = fmaxf(p.pressure, 0.0f);      // No negative pressure
    p.pressure = fminf(p.pressure, 1000.0f);   // Cap maximum pressure
}

// Compute forces using spatial hashing (pressure + viscosity + gravity)
__global__ void compute_forces_kernel(
    Particle* particles, GridCell* grid, int num_particles) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    
    Particle& p = particles[idx];
    float2 pos = p.position;
    float2 vel = p.velocity;
    float2 pressure_force = make_float2(0.0f, 0.0f);
    float2 viscosity_force = make_float2(0.0f, 0.0f);
    
    // Get grid cell for this particle
    int2 grid_pos = get_grid_pos(pos);
    
    // Search in 3x3 neighboring cells
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            int2 neighbor_pos = make_int2(grid_pos.x + x, grid_pos.y + y);
            
            // Check bounds
            if (neighbor_pos.x < 0 || neighbor_pos.x >= GRID_SIZE ||
                neighbor_pos.y < 0 || neighbor_pos.y >= GRID_SIZE) continue;
            
            int hash = grid_hash(neighbor_pos);
            if (hash < 0 || hash >= GRID_SIZE * GRID_SIZE) continue;
            
            GridCell& cell = grid[hash];
            
            if (cell.start == -1 || cell.count <= 0) continue;
            
            // Check all particles in this cell
            for (int i = 0; i < cell.count; i++) {
                int neighbor_idx = cell.start + i;
                if (neighbor_idx < 0 || neighbor_idx >= num_particles || neighbor_idx == idx) continue;
                
                Particle& neighbor = particles[neighbor_idx];
                float2 diff = make_float2(pos.x - neighbor.position.x, 
                                        pos.y - neighbor.position.y);
                float r = sqrtf(diff.x * diff.x + diff.y * diff.y);
                
                if (r < SMOOTHING_LENGTH && r > 0.0f) {
                    // Pressure force (same formula as reference)
                    float2 r_norm = make_float2(diff.x / r, diff.y / r);
                    float w_press = CONST_SPIKY * powf(SMOOTHING_LENGTH - r, 2.0f);
                    float pressure_coeff = PARTICLE_MASS * (p.pressure / (p.density * p.density) + 
                                                          neighbor.pressure / (neighbor.density * neighbor.density));
                    pressure_force.x -= pressure_coeff * w_press * r_norm.x;
                    pressure_force.y -= pressure_coeff * w_press * r_norm.y;
                    
                    // Viscosity force (same as reference)
                    float2 vel_diff = make_float2(neighbor.velocity.x - vel.x, 
                                                neighbor.velocity.y - vel.y);
                    float w_visc = CONST_SPIKY * (SMOOTHING_LENGTH - r);
                    float visc_coeff = PARTICLE_MASS * (vel_diff.x / neighbor.density) * w_visc;
                    viscosity_force.x += DYNAMIC_VISCOSITY * visc_coeff;
                    visc_coeff = PARTICLE_MASS * (vel_diff.y / neighbor.density) * w_visc;
                    viscosity_force.y += DYNAMIC_VISCOSITY * visc_coeff;
                }
            }
        }
    }
    
    // Total force = (pressure + viscosity) / density + gravity (same as reference)
    p.force.x = (pressure_force.x + viscosity_force.x) / p.density;
    p.force.y = (pressure_force.y + viscosity_force.y) / p.density + GRAVITY;
}

// Integrate particles (Leapfrog integration)
__global__ void integrate_kernel(Particle* particles, int num_particles, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    
    Particle& p = particles[idx];
    
    // Update velocity (F = ma, so a = F/m)
    float2 acceleration = make_float2(p.force.x / PARTICLE_MASS, p.force.y / PARTICLE_MASS);
    p.velocity.x += acceleration.x * dt;
    p.velocity.y += acceleration.y * dt;
    
    // Apply velocity damping to prevent instabilities
    const float DAMPING = 0.99f;
    p.velocity.x *= DAMPING;
    p.velocity.y *= DAMPING;
    
    // Clamp velocity to reasonable values
    const float MAX_VELOCITY = 2.0f; // 2 m/s max
    float vel_mag = sqrtf(p.velocity.x * p.velocity.x + p.velocity.y * p.velocity.y);
    if (vel_mag > MAX_VELOCITY) {
        p.velocity.x = (p.velocity.x / vel_mag) * MAX_VELOCITY;
        p.velocity.y = (p.velocity.y / vel_mag) * MAX_VELOCITY;
    }
    
    // Update position
    p.position.x += p.velocity.x * dt;
    p.position.y += p.velocity.y * dt;
}

// Handle boundary conditions
__global__ void boundary_kernel(Particle* particles, int num_particles) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_particles) return;
    
    Particle& p = particles[idx];
    
    // Boundaries (assuming [0,1] x [0,1] domain)
    float margin = PARTICLE_RADIUS;
    
    if (p.position.x < margin) {
        p.position.x = margin;
        p.velocity.x *= -BOUNDARY_DAMPING;
    }
    if (p.position.x > 1.0f - margin) {
        p.position.x = 1.0f - margin;
        p.velocity.x *= -BOUNDARY_DAMPING;
    }
    if (p.position.y < margin) {
        p.position.y = margin;
        p.velocity.y *= -BOUNDARY_DAMPING;
    }
    if (p.position.y > 1.0f - margin) {
        p.position.y = 1.0f - margin;
        p.velocity.y *= -BOUNDARY_DAMPING;
    }
}

// Wrapper functions for kernel launches
void launch_reset_grid(GridCell* grid, int grid_size) {
    int block_size = 256;
    int grid_blocks = (grid_size + block_size - 1) / block_size;
    reset_grid_kernel<<<grid_blocks, block_size>>>(grid, grid_size);
}

void launch_compute_hash(Particle* particles, int* hashes, int* indices, int num_particles) {
    int block_size = 256;
    int grid_blocks = (num_particles + block_size - 1) / block_size;
    compute_hash_kernel<<<grid_blocks, block_size>>>(particles, hashes, indices, num_particles);
}

void launch_reorder_particles(Particle* particles_in, Particle* particles_out, int* sorted_indices, int num_particles) {
    int block_size = 256;
    int grid_blocks = (num_particles + block_size - 1) / block_size;
    reorder_particles_kernel<<<grid_blocks, block_size>>>(particles_in, particles_out, sorted_indices, num_particles);
}

void launch_find_cell_start(int* hashes, GridCell* grid, int num_particles) {
    int block_size = 256;
    int grid_blocks = (num_particles + block_size - 1) / block_size;
    find_cell_start_kernel<<<grid_blocks, block_size>>>(hashes, grid, num_particles);
}

void launch_compute_density_pressure(Particle* particles, GridCell* grid, int num_particles) {
    int block_size = 256;
    int grid_blocks = (num_particles + block_size - 1) / block_size;
    compute_density_pressure_kernel<<<grid_blocks, block_size>>>(particles, grid, num_particles);
}

void launch_compute_forces(Particle* particles, GridCell* grid, int num_particles) {
    int block_size = 256;
    int grid_blocks = (num_particles + block_size - 1) / block_size;
    compute_forces_kernel<<<grid_blocks, block_size>>>(particles, grid, num_particles);
}

void launch_integrate(Particle* particles, int num_particles, float dt) {
    int block_size = 256;
    int grid_blocks = (num_particles + block_size - 1) / block_size;
    integrate_kernel<<<grid_blocks, block_size>>>(particles, num_particles, dt);
}

void launch_boundary(Particle* particles, int num_particles) {
    int block_size = 256;
    int grid_blocks = (num_particles + block_size - 1) / block_size;
    boundary_kernel<<<grid_blocks, block_size>>>(particles, num_particles);
}