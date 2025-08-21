#pragma once

#include "common.h"

class Renderer {
public:
    Renderer();
    ~Renderer();

    // Initialize OpenGL resources
    bool initialize();
    void cleanup();

    // Render the water simulation
    void render(const float* density_data);

    // Get CUDA graphics resource for interop
    struct cudaGraphicsResource* get_cuda_resource() const { return cuda_gl_resource; }

private:
    // OpenGL objects
    GLuint VAO, VBO, EBO;
    GLuint shaderProgram;
    GLuint texture_gl;
    struct cudaGraphicsResource *cuda_gl_resource;

    // Shader compilation
    GLuint compile_shader(const char* source, GLenum type);
    GLuint create_shader_program();
    void setup_geometry();
    void setup_texture();
};