#include "../include/sph_renderer.h"

SPHRenderer::SPHRenderer() : VAO(0), VBO(0), shaderProgram(0), 
                             d_render_positions(nullptr), h_render_positions(nullptr) {
}

SPHRenderer::~SPHRenderer() {
    cleanup();
}

bool SPHRenderer::initialize() {
    if (!create_shader_program()) {
        return false;
    }
    
    setup_particle_rendering();
    
    // Allocate memory for render data
    CHECK_CUDA(cudaMalloc(&d_render_positions, MAX_PARTICLES * 2 * sizeof(float)));
    h_render_positions = new float[MAX_PARTICLES * 2];
    
    return true;
}

void SPHRenderer::cleanup() {
    if (VAO) { glDeleteVertexArrays(1, &VAO); VAO = 0; }
    if (VBO) { glDeleteBuffers(1, &VBO); VBO = 0; }
    if (shaderProgram) { glDeleteProgram(shaderProgram); shaderProgram = 0; }
    if (d_render_positions) { cudaFree(d_render_positions); d_render_positions = nullptr; }
    if (h_render_positions) { delete[] h_render_positions; h_render_positions = nullptr; }
}

GLuint SPHRenderer::compile_shader(const char* source, GLenum type) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, NULL, infoLog);
        std::cerr << "Shader compilation failed: " << infoLog << std::endl;
        return 0;
    }
    
    return shader;
}

GLuint SPHRenderer::create_shader_program() {
    // Vertex shader for particle rendering
    const char* vertex_shader_source = R"glsl(
        #version 330 core
        layout (location = 0) in vec2 aPos;
        
        uniform vec2 u_resolution;
        out vec2 worldPos;
        
        void main() {
            // Convert from [0,1] world space to NDC [-1,1]
            vec2 ndc = aPos * 2.0 - 1.0;
            // Don't flip Y - let Y=0 be bottom, Y=1 be top (standard OpenGL)
            
            gl_Position = vec4(ndc, 0.0, 1.0);
            gl_PointSize = 8.0; // Size of particle points
            worldPos = aPos;
        }
    )glsl";
    
    // Fragment shader with water-like particle rendering
    const char* fragment_shader_source = R"glsl(
        #version 330 core
        out vec4 FragColor;
        in vec2 worldPos;
        
        void main() {
            // Create circular particles
            vec2 coord = gl_PointCoord - vec2(0.5);
            float dist = length(coord);
            
            if (dist > 0.5) {
                discard; // Make particles circular
            }
            
            // Water-like coloring based on depth and density
            float alpha = 1.0 - dist * 2.0; // Fade from center
            alpha = smoothstep(0.0, 1.0, alpha);
            
            // Water color gradient
            vec3 centerColor = vec3(0.1, 0.4, 0.9);  // Blue center
            vec3 edgeColor = vec3(0.0, 0.2, 0.6);    // Darker blue edge
            vec3 color = mix(edgeColor, centerColor, alpha);
            
            FragColor = vec4(color, alpha * 0.8);
        }
    )glsl";
    
    GLuint vs = compile_shader(vertex_shader_source, GL_VERTEX_SHADER);
    GLuint fs = compile_shader(fragment_shader_source, GL_FRAGMENT_SHADER);
    
    if (vs == 0 || fs == 0) {
        return false;
    }
    
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vs);
    glAttachShader(shaderProgram, fs);
    glLinkProgram(shaderProgram);
    
    GLint success;
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cerr << "Shader program linking failed: " << infoLog << std::endl;
        shaderProgram = 0;
    }
    
    glDeleteShader(vs);
    glDeleteShader(fs);
    
    return shaderProgram != 0;
}

void SPHRenderer::setup_particle_rendering() {
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    
    // Allocate buffer for maximum particles
    glBufferData(GL_ARRAY_BUFFER, MAX_PARTICLES * 2 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
    
    // Position attribute
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    glBindVertexArray(0);
}

void SPHRenderer::update_render_data(const Particle* particles, int particle_count) {
    if (particle_count == 0 || particle_count > MAX_PARTICLES) return;
    
    // Synchronize CUDA before memory copy
    cudaDeviceSynchronize();
    
    // Copy particle positions from GPU to CPU safely
    std::vector<Particle> h_particles(particle_count);
    cudaError_t err = cudaMemcpy(h_particles.data(), particles, 
                                particle_count * sizeof(Particle), cudaMemcpyDeviceToHost);
    
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy error: " << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // Extract positions for rendering
    for (int i = 0; i < particle_count; i++) {
        h_render_positions[i * 2] = h_particles[i].position.x;
        h_render_positions[i * 2 + 1] = h_particles[i].position.y;
    }
}

void SPHRenderer::render(const Particle* particles, int particle_count) {
    if (particle_count == 0) return;
    
    // Update render data
    update_render_data(particles, particle_count);
    
    // Enable blending for transparent particles
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_PROGRAM_POINT_SIZE);
    
    // Clear background with dark blue
    glClearColor(0.05f, 0.1f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    
    // Use shader program
    glUseProgram(shaderProgram);
    
    // Set resolution uniform
    GLint resolutionLoc = glGetUniformLocation(shaderProgram, "u_resolution");
    glUniform2f(resolutionLoc, (float)WIN_SIZE, (float)WIN_SIZE);
    
    // Update VBO with particle positions
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferSubData(GL_ARRAY_BUFFER, 0, particle_count * 2 * sizeof(float), h_render_positions);
    
    // Render particles as points
    glDrawArrays(GL_POINTS, 0, particle_count);
    
    glBindVertexArray(0);
    glDisable(GL_PROGRAM_POINT_SIZE);
    glDisable(GL_BLEND);
}