#pragma once

#include "sph_common.h"

class SPHRenderer {
public:
    SPHRenderer();
    ~SPHRenderer();

    // Initialize OpenGL resources for particle rendering
    bool initialize();
    void cleanup();

    // Render particles
    void render(const Particle* particles, int particle_count);

private:
    // OpenGL objects
    GLuint VAO, VBO;
    GLuint shaderProgram;
    
    // Particle rendering data
    float* d_render_positions;
    float* h_render_positions;
    
    // Shader compilation
    GLuint compile_shader(const char* source, GLenum type);
    GLuint create_shader_program();
    void setup_particle_rendering();
    
    // Convert particles to render format
    void update_render_data(const Particle* particles, int particle_count);
};