#include "../include/renderer.h"
#include <fstream>
#include <sstream>

// Forward declare CUDA function wrapper
void launch_density_to_rgba_kernel(cudaSurfaceObject_t surface, const float* density, int width, int height);

Renderer::Renderer() : VAO(0), VBO(0), EBO(0), shaderProgram(0), texture_gl(0), cuda_gl_resource(nullptr) {
}

Renderer::~Renderer() {
    cleanup();
}

bool Renderer::initialize() {
    if (!create_shader_program()) {
        return false;
    }
    
    setup_geometry();
    setup_texture();
    
    return true;
}

void Renderer::cleanup() {
    if (cuda_gl_resource) {
        cudaGraphicsUnregisterResource(cuda_gl_resource);
        cuda_gl_resource = nullptr;
    }
    
    if (VAO) { glDeleteVertexArrays(1, &VAO); VAO = 0; }
    if (VBO) { glDeleteBuffers(1, &VBO); VBO = 0; }
    if (EBO) { glDeleteBuffers(1, &EBO); EBO = 0; }
    if (shaderProgram) { glDeleteProgram(shaderProgram); shaderProgram = 0; }
    if (texture_gl) { glDeleteTextures(1, &texture_gl); texture_gl = 0; }
}

GLuint Renderer::compile_shader(const char* source, GLenum type) {
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

GLuint Renderer::create_shader_program() {
    // Inline shader sources for simplicity
    const char* vertex_shader_source = R"glsl(
        #version 330 core
        layout (location = 0) in vec2 aPos;
        layout (location = 1) in vec2 aTexCoord;
        out vec2 TexCoord;
        void main() {
            gl_Position = vec4(aPos, 0.0, 1.0);
            TexCoord = aTexCoord;
        }
    )glsl";
    
    const char* fragment_shader_source = R"glsl(
        #version 330 core
        out vec4 FragColor;
        in vec2 TexCoord;
        uniform sampler2D waterTexture;
        void main() {
            vec4 tex = texture(waterTexture, TexCoord);
            float density = tex.a; // Use alpha channel for density
            
            // Enhanced water visualization
            vec3 deepWater = vec3(0.0, 0.3, 0.8);     // Dark blue for deep water
            vec3 shallowWater = vec3(0.2, 0.7, 1.0);   // Light blue for shallow water
            vec3 foam = vec3(0.9, 0.95, 1.0);          // White foam
            
            // More dramatic color mixing
            vec3 waterColor = mix(deepWater, shallowWater, density);
            vec3 finalColor = mix(waterColor, foam, density * density);
            
            float alpha = clamp(density * 2.0, 0.0, 1.0);
            FragColor = vec4(finalColor, alpha);
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

void Renderer::setup_geometry() {
    float vertices[] = {
        1.0f,  1.0f, 1.0f, 1.0f,  // top right
        1.0f, -1.0f, 1.0f, 0.0f,  // bottom right
       -1.0f, -1.0f, 0.0f, 0.0f,  // bottom left
       -1.0f,  1.0f, 0.0f, 1.0f   // top left
    };
    
    unsigned int indices[] = {
        0, 1, 3,
        1, 2, 3
    };
    
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    
    glBindVertexArray(VAO);
    
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
}

void Renderer::setup_texture() {
    glGenTextures(1, &texture_gl);
    glBindTexture(GL_TEXTURE_2D, texture_gl);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, N+2, N+2, 0, GL_RGBA, GL_FLOAT, NULL);
    
    CHECK_CUDA(cudaGraphicsGLRegisterImage(&cuda_gl_resource, texture_gl, 
                                           GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
}

void Renderer::render(const float* density_data) {
    // Map OpenGL texture to CUDA
    CHECK_CUDA(cudaGraphicsMapResources(1, &cuda_gl_resource, 0));
    
    cudaArray_t cuda_array;
    CHECK_CUDA(cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_gl_resource, 0, 0));
    
    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = cuda_array;
    
    cudaSurfaceObject_t surface_obj = 0;
    CHECK_CUDA(cudaCreateSurfaceObject(&surface_obj, &res_desc));
    
    // Launch CUDA kernel to convert density to RGBA
    launch_density_to_rgba_kernel(surface_obj, density_data, N+2, N+2);
    
    CHECK_CUDA(cudaDestroySurfaceObject(surface_obj));
    CHECK_CUDA(cudaGraphicsUnmapResources(1, &cuda_gl_resource, 0));
    
    // Render with OpenGL
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    glUseProgram(shaderProgram);
    glBindTexture(GL_TEXTURE_2D, texture_gl);
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
}