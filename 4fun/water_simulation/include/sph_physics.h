#pragma once

#include "sph_common.h"

class SPHPhysics {
public:
    SPHPhysics();
    ~SPHPhysics();

    // Initialize the SPH system
    void initialize();
    void cleanup();

    // Simulation step
    void simulate_step();
    
    // Particle management
    void add_particles_at(float2 position, int count);
    void add_particle_interaction(float2 position, float2 velocity);
    
    // Getters for rendering
    int get_particle_count() const { return num_particles; }
    Particle* get_particles() const { return d_particles; }

private:
    // Particle data
    Particle* d_particles;
    Particle* d_particles_sorted;
    int num_particles;
    
    // Spatial hashing data
    int* d_particle_indices;
    int* d_particle_indices_sorted;
    int* d_grid_hashes;
    int* d_grid_hashes_sorted;
    GridCell* d_grid;
    
    // Neighbor data
    int* d_neighbors;
    int* d_neighbor_counts;
    
    // SPH computation steps
    void update_spatial_hash();
    void find_neighbors();
    void compute_density_pressure();
    void compute_forces();
    void integrate();
    void handle_boundaries();
    
    // Helper functions
    void sort_particles();
    void reset_grid();
};