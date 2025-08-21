#include "../include/sph_physics.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <random>

// Declare external CUDA kernel wrappers
extern void launch_reset_grid(GridCell* grid, int grid_size);
extern void launch_compute_hash(Particle* particles, int* hashes, int* indices, int num_particles);
extern void launch_reorder_particles(Particle* particles_in, Particle* particles_out, int* sorted_indices, int num_particles);
extern void launch_find_cell_start(int* hashes, GridCell* grid, int num_particles);
extern void launch_compute_density_pressure(Particle* particles, GridCell* grid, int num_particles);
extern void launch_compute_forces(Particle* particles, GridCell* grid, int num_particles);
extern void launch_integrate(Particle* particles, int num_particles, float dt);
extern void launch_boundary(Particle* particles, int num_particles);

SPHPhysics::SPHPhysics() : 
    d_particles(nullptr), d_particles_sorted(nullptr), num_particles(0),
    d_particle_indices(nullptr), d_particle_indices_sorted(nullptr),
    d_grid_hashes(nullptr), d_grid_hashes_sorted(nullptr), d_grid(nullptr),
    d_neighbors(nullptr), d_neighbor_counts(nullptr) {
}

SPHPhysics::~SPHPhysics() {
    cleanup();
}

void SPHPhysics::initialize() {
    // Allocate memory for particles
    CHECK_CUDA(cudaMalloc(&d_particles, MAX_PARTICLES * sizeof(Particle)));
    CHECK_CUDA(cudaMalloc(&d_particles_sorted, MAX_PARTICLES * sizeof(Particle)));
    
    // Allocate memory for spatial hashing
    CHECK_CUDA(cudaMalloc(&d_particle_indices, MAX_PARTICLES * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_particle_indices_sorted, MAX_PARTICLES * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_grid_hashes, MAX_PARTICLES * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_grid_hashes_sorted, MAX_PARTICLES * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_grid, GRID_SIZE * GRID_SIZE * sizeof(GridCell)));
    
    // Initialize with some particles
    std::vector<Particle> h_particles;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> pos_x(0.1f, 0.9f);
    std::uniform_real_distribution<float> pos_y(0.1f, 0.4f);
    
    // Create initial water pool - use spacing from reference simulation
    float spacing = 0.07f;  // spawn_dist from reference (much larger spacing)
    int particles_per_row = (int)(0.6f / spacing);  // 60cm width of water
    int rows = (int)(0.2f / spacing);               // 20cm height of water pool
    num_particles = 0;
    
    // Start water pool at bottom center (like reference)
    float start_x = 0.2f;  // 20cm from left edge
    float start_y = 0.1f;  // 10cm from bottom (higher start like reference)
    
    for (int row = 0; row < rows && num_particles < MAX_PARTICLES; row++) {
        for (int col = 0; col < particles_per_row && num_particles < MAX_PARTICLES; col++) {
            Particle p;
            // Hexagonal close packing for more realistic initial configuration
            float offset_x = (row % 2) * spacing * 0.5f;
            p.position.x = start_x + col * spacing + offset_x;
            p.position.y = start_y + row * spacing;
            
            // Add small random perturbation to break perfect symmetry
            p.position.x += (pos_x(gen) - 0.5f) * spacing * 0.1f;
            p.position.y += (pos_y(gen) - 0.5f) * spacing * 0.1f;
            
            p.velocity = make_float2(0.0f, 0.0f);
            p.force = make_float2(0.0f, 0.0f);
            p.density = REST_DENSITY;
            p.pressure = 0.0f;
            
            h_particles.push_back(p);
            num_particles++;
        }
    }
    
    // Copy to GPU
    CHECK_CUDA(cudaMemcpy(d_particles, h_particles.data(), 
                         num_particles * sizeof(Particle), cudaMemcpyHostToDevice));
    
    std::cout << "SPH initialized with " << num_particles << " particles" << std::endl;
}

void SPHPhysics::cleanup() {
    if (d_particles) { cudaFree(d_particles); d_particles = nullptr; }
    if (d_particles_sorted) { cudaFree(d_particles_sorted); d_particles_sorted = nullptr; }
    if (d_particle_indices) { cudaFree(d_particle_indices); d_particle_indices = nullptr; }
    if (d_particle_indices_sorted) { cudaFree(d_particle_indices_sorted); d_particle_indices_sorted = nullptr; }
    if (d_grid_hashes) { cudaFree(d_grid_hashes); d_grid_hashes = nullptr; }
    if (d_grid_hashes_sorted) { cudaFree(d_grid_hashes_sorted); d_grid_hashes_sorted = nullptr; }
    if (d_grid) { cudaFree(d_grid); d_grid = nullptr; }
    if (d_neighbors) { cudaFree(d_neighbors); d_neighbors = nullptr; }
    if (d_neighbor_counts) { cudaFree(d_neighbor_counts); d_neighbor_counts = nullptr; }
}

void SPHPhysics::simulate_step() {
    if (num_particles == 0) return;
    
    // Full SPH with spatial hashing
    // 1. Update spatial hash
    update_spatial_hash();
    
    // 2. Compute density and pressure
    compute_density_pressure();
    
    // 3. Compute forces
    compute_forces();
    
    // 4. Integrate
    integrate();
    
    // 5. Handle boundaries
    handle_boundaries();
}

void SPHPhysics::update_spatial_hash() {
    // Reset grid
    launch_reset_grid(d_grid, GRID_SIZE * GRID_SIZE);
    
    // Compute hash for each particle
    launch_compute_hash(d_particles, d_grid_hashes, d_particle_indices, num_particles);
    
    // Sort by hash using thrust
    try {
        thrust::device_ptr<int> hash_ptr(d_grid_hashes);
        thrust::device_ptr<int> index_ptr(d_particle_indices);
        thrust::sort_by_key(hash_ptr, hash_ptr + num_particles, index_ptr);
        
        // Copy sorted data
        CHECK_CUDA(cudaMemcpy(d_grid_hashes_sorted, d_grid_hashes, 
                             num_particles * sizeof(int), cudaMemcpyDeviceToDevice));
        CHECK_CUDA(cudaMemcpy(d_particle_indices_sorted, d_particle_indices, 
                             num_particles * sizeof(int), cudaMemcpyDeviceToDevice));
        
        // Reorder particles
        launch_reorder_particles(d_particles, d_particles_sorted, d_particle_indices_sorted, num_particles);
        
        // Copy sorted particles back
        CHECK_CUDA(cudaMemcpy(d_particles, d_particles_sorted, 
                             num_particles * sizeof(Particle), cudaMemcpyDeviceToDevice));
        
        // Find start of each cell
        launch_find_cell_start(d_grid_hashes_sorted, d_grid, num_particles);
        
    } catch (const std::exception& e) {
        std::cerr << "Thrust error in spatial hashing: " << e.what() << std::endl;
        // Fall back to simple hash without sorting
        launch_reset_grid(d_grid, GRID_SIZE * GRID_SIZE);
    }
}

void SPHPhysics::compute_density_pressure() {
    launch_compute_density_pressure(d_particles, d_grid, num_particles);
}

void SPHPhysics::compute_forces() {
    launch_compute_forces(d_particles, d_grid, num_particles);
}

void SPHPhysics::integrate() {
    launch_integrate(d_particles, num_particles, TIME_STEP);
}

void SPHPhysics::handle_boundaries() {
    launch_boundary(d_particles, num_particles);
}

void SPHPhysics::add_particles_at(float2 position, int count) {
    if (num_particles + count > MAX_PARTICLES) {
        count = MAX_PARTICLES - num_particles;
    }
    
    if (count <= 0) return;
    
    std::vector<Particle> new_particles(count);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> offset(-0.02f, 0.02f);
    
    for (int i = 0; i < count; i++) {
        Particle& p = new_particles[i];
        p.position.x = position.x + offset(gen);
        p.position.y = position.y + offset(gen);
        p.velocity = make_float2(0.0f, 0.0f);
        p.force = make_float2(0.0f, 0.0f);
        p.density = REST_DENSITY;
        p.pressure = 0.0f;
    }
    
    CHECK_CUDA(cudaMemcpy(&d_particles[num_particles], new_particles.data(),
                         count * sizeof(Particle), cudaMemcpyHostToDevice));
    
    num_particles += count;
}

void SPHPhysics::add_particle_interaction(float2 position, float2 velocity) {
    // Find particles near the interaction point and apply velocity
    std::vector<Particle> h_particles(num_particles);
    CHECK_CUDA(cudaMemcpy(h_particles.data(), d_particles, 
                         num_particles * sizeof(Particle), cudaMemcpyDeviceToHost));
    
    float interaction_radius = 0.08f;  // 8cm interaction radius
    float force_strength = 1.0f;      // Much gentler force
    
    for (int i = 0; i < num_particles; i++) {
        float2 diff = make_float2(h_particles[i].position.x - position.x,
                                h_particles[i].position.y - position.y);
        float dist = sqrtf(diff.x * diff.x + diff.y * diff.y);
        
        if (dist < interaction_radius && dist > 0.001f) {
            float influence = 1.0f - (dist / interaction_radius);
            
            // Apply gentle force impulse
            float impulse = force_strength * influence * TIME_STEP / PARTICLE_MASS;
            h_particles[i].velocity.x += velocity.x * impulse * 0.1f; // Much gentler
            h_particles[i].velocity.y += velocity.y * impulse * 0.1f;
        }
    }
    
    CHECK_CUDA(cudaMemcpy(d_particles, h_particles.data(),
                         num_particles * sizeof(Particle), cudaMemcpyHostToDevice));
}