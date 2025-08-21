#include <iostream>
#include <vector>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>

#include "glm_helper.h"
#include "camera.h"
#include "sphere.h"
#include "input_handler.h"
#include "ball.h"
#include "simulation_parameters.h"
#include "kernels.h"

// --- MACRO PARA VERIFICAR ERRORES DE CUDA ---
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// --- Estado de la cámara y la caja ---
Camera camera(vec3(0.0f, 0.0f, 20.0f));
vec3 box_center(0.0f, 0.0f, 0.0f);

// ==================================================================
// SHADERS DE OPENGL
// ==================================================================
const char* vertex_shader_source = R"glsl(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aOffset;
    layout (location = 2) in vec3 aColor;

    uniform mat4 view;
    uniform mat4 projection;

    out vec3 FragColor;

    void main() {
        gl_Position = projection * view * vec4(aPos + aOffset, 1.0);
        FragColor = aColor;
    }
)glsl";

const char* fragment_shader_source = R"glsl(
    #version 330 core
    in vec3 FragColor;
    out vec4 OutColor;
    void main() { OutColor = vec4(FragColor, 1.0); }
)glsl";

const char* line_vertex_shader_source = R"glsl(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    void main() { gl_Position = projection * view * model * vec4(aPos, 1.0); }
)glsl";

const char* line_fragment_shader_source = R"glsl(
    #version 330 core
    out vec4 FragColor;
    void main() { FragColor = vec4(1.0, 1.0, 1.0, 1.0); }
)glsl";

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

// ==================================================================
// FUNCIÓN MAIN
// ==================================================================
int main() {
    // 1. Inicialización de GLFW y GLAD
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(WIN_SIZE, WIN_SIZE, "CUDA 3D Template", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    glfwSwapInterval(1);

    InputHandler input_handler(camera, box_center);
    input_handler.setup_callbacks(window);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // 2. Inicializar CUDA
    CHECK_CUDA(cudaSetDevice(0));

    // 3. Compilar Shaders de OpenGL
    GLuint vs = glCreateShader(GL_VERTEX_SHADER); glShaderSource(vs, 1, &vertex_shader_source, NULL); glCompileShader(vs);
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER); glShaderSource(fs, 1, &fragment_shader_source, NULL); glCompileShader(fs);
    GLuint shaderProgram = glCreateProgram(); glAttachShader(shaderProgram, vs); glAttachShader(shaderProgram, fs); glLinkProgram(shaderProgram); glDeleteShader(vs); glDeleteShader(fs);

    GLuint line_vs = glCreateShader(GL_VERTEX_SHADER); glShaderSource(line_vs, 1, &line_vertex_shader_source, NULL); glCompileShader(line_vs);
    GLuint line_fs = glCreateShader(GL_FRAGMENT_SHADER); glShaderSource(line_fs, 1, &line_fragment_shader_source, NULL); glCompileShader(line_fs);
    GLuint lineShaderProgram = glCreateProgram(); glAttachShader(lineShaderProgram, line_vs); glAttachShader(lineShaderProgram, line_fs); glLinkProgram(lineShaderProgram); glDeleteShader(line_vs); glDeleteShader(line_fs);

    // 4. Crear la malla de la esfera y la caja
    std::vector<float> sphere_vertices; std::vector<unsigned int> sphere_indices;
    create_sphere(sphere_vertices, sphere_indices, BALL_RADIUS, 12, 12);
    GLuint sphereVAO, sphereVBO, sphereEBO, instanceVBO;
    glGenVertexArrays(1, &sphereVAO); glGenBuffers(1, &sphereVBO); glGenBuffers(1, &sphereEBO); glGenBuffers(1, &instanceVBO);
    
    glBindVertexArray(sphereVAO);
    glBindBuffer(GL_ARRAY_BUFFER, sphereVBO);
    glBufferData(GL_ARRAY_BUFFER, sphere_vertices.size() * sizeof(float), sphere_vertices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphereEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphere_indices.size() * sizeof(unsigned int), sphere_indices.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Ball) * NUM_BALLS, NULL, GL_DYNAMIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Ball), (void*)offsetof(Ball, pos));
    glVertexAttribDivisor(1, 1);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Ball), (void*)offsetof(Ball, color));
    glVertexAttribDivisor(2, 1);
    glBindVertexArray(0);

    float box_vertices[] = {
        -BOX_SIZE,-BOX_SIZE,-BOX_SIZE, BOX_SIZE,-BOX_SIZE,-BOX_SIZE, BOX_SIZE,-BOX_SIZE,-BOX_SIZE, BOX_SIZE, BOX_SIZE,-BOX_SIZE,
        BOX_SIZE, BOX_SIZE,-BOX_SIZE, -BOX_SIZE, BOX_SIZE,-BOX_SIZE, -BOX_SIZE, BOX_SIZE,-BOX_SIZE, -BOX_SIZE,-BOX_SIZE,-BOX_SIZE,
        -BOX_SIZE,-BOX_SIZE, BOX_SIZE, BOX_SIZE,-BOX_SIZE, BOX_SIZE, BOX_SIZE,-BOX_SIZE, BOX_SIZE, BOX_SIZE, BOX_SIZE, BOX_SIZE,
        BOX_SIZE, BOX_SIZE, BOX_SIZE, -BOX_SIZE, BOX_SIZE, BOX_SIZE, -BOX_SIZE, BOX_SIZE, BOX_SIZE, -BOX_SIZE,-BOX_SIZE, BOX_SIZE,
        -BOX_SIZE,-BOX_SIZE,-BOX_SIZE, -BOX_SIZE,-BOX_SIZE, BOX_SIZE, BOX_SIZE,-BOX_SIZE,-BOX_SIZE, BOX_SIZE,-BOX_SIZE, BOX_SIZE,
        BOX_SIZE, BOX_SIZE,-BOX_SIZE, BOX_SIZE, BOX_SIZE, BOX_SIZE, -BOX_SIZE, BOX_SIZE,-BOX_SIZE, -BOX_SIZE, BOX_SIZE, BOX_SIZE
    };
    GLuint boxVAO, boxVBO;
    glGenVertexArrays(1, &boxVAO); glGenBuffers(1, &boxVBO);
    glBindVertexArray(boxVAO); glBindBuffer(GL_ARRAY_BUFFER, boxVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(box_vertices), box_vertices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0); glEnableVertexAttribArray(0);

    // 5. Crear las partículas y reservar memoria en GPU
    std::vector<Ball> h_balls(NUM_BALLS);
    for (int i = 0; i < NUM_BALLS; ++i) {
        h_balls[i].pos = vec3(
            ((float)rand() / RAND_MAX) * (BOX_SIZE*1.8f) - (BOX_SIZE*0.9f),
            ((float)rand() / RAND_MAX) * (BOX_SIZE*1.8f) - (BOX_SIZE*0.9f),
            ((float)rand() / RAND_MAX) * (BOX_SIZE*1.8f) - (BOX_SIZE*0.9f)
        );
        h_balls[i].vel = vec3(0.0f, 0.0f, 0.0f);
        h_balls[i].color = vec3((float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX);
    }
    Ball* d_balls; CHECK_CUDA(cudaMalloc(&d_balls, NUM_BALLS * sizeof(Ball)));
    CHECK_CUDA(cudaMemcpy(d_balls, h_balls.data(), NUM_BALLS * sizeof(Ball), cudaMemcpyHostToDevice));

    glEnable(GL_DEPTH_TEST);
    float lastTime = 0.0f;

    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_BALLS + threadsPerBlock - 1) / threadsPerBlock;

    // 6. Bucle principal
    while (!glfwWindowShouldClose(window)) {
        float currentTime = glfwGetTime(); float dt = currentTime - lastTime; lastTime = currentTime;
        if (dt > 0.016f) dt = 0.016f; // Limitar dt para estabilidad

        glfwPollEvents();

        // --- Actualización de la física con CUDA ---
        update_particles_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_balls, dt, box_center);
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        // --- Copiar datos de vuelta a la CPU para renderizar ---
        CHECK_CUDA(cudaMemcpy(h_balls.data(), d_balls, NUM_BALLS * sizeof(Ball), cudaMemcpyDeviceToHost));

        // --- Renderizado con OpenGL ---
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);

        mat4 projection = perspective(45.0f, (float)display_w / (float)display_h, 0.1f, 100.0f);
        mat4 view = camera.getViewMatrix();

        // Dibujar las partículas (con instancing)
        glUseProgram(shaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, &projection.m[0]);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, &view.m[0]);
        
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, NUM_BALLS * sizeof(Ball), h_balls.data());

        glBindVertexArray(sphereVAO);
        glDrawElementsInstanced(GL_TRIANGLES, sphere_indices.size(), GL_UNSIGNED_INT, 0, NUM_BALLS);
        glBindVertexArray(0);

        // Dibujar la caja
        glUseProgram(lineShaderProgram);
        mat4 box_model; box_model.m[12] = box_center.x; box_model.m[13] = box_center.y; box_model.m[14] = box_center.z;
        glUniformMatrix4fv(glGetUniformLocation(lineShaderProgram, "projection"), 1, GL_FALSE, &projection.m[0]);
        glUniformMatrix4fv(glGetUniformLocation(lineShaderProgram, "view"), 1, GL_FALSE, &view.m[0]);
        glUniformMatrix4fv(glGetUniformLocation(lineShaderProgram, "model"), 1, GL_FALSE, &box_model.m[0]);
        glBindVertexArray(boxVAO);
        glDrawArrays(GL_LINES, 0, 24);

        glfwSwapBuffers(window);
    }

    // 7. Limpieza
    CHECK_CUDA(cudaFree(d_balls));
    CHECK_CUDA(cudaDeviceReset());

    glDeleteVertexArrays(1, &sphereVAO); 
    glDeleteBuffers(1, &sphereVBO); 
    glDeleteBuffers(1, &sphereEBO);
    glDeleteBuffers(1, &instanceVBO);
    glDeleteVertexArrays(1, &boxVAO); 
    glDeleteBuffers(1, &boxVBO);
    glDeleteProgram(shaderProgram); 
    glDeleteProgram(lineShaderProgram);
    
    glfwDestroyWindow(window); 
    glfwTerminate();

    return 0;
}