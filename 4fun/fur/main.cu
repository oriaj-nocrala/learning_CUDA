#include <iostream>
#include <vector>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "glm_helper.cuh"
#include "camera.h"
#include "sphere.h"
#include "input_handler.h"
#include "hair.h"
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

// --- Estado de la cámara y las esferas ---
Camera camera(vec3(0.0f, 0.0f, 10.0f));
vec3 sphere_center(0.0f, 0.0f, 0.0f);
vec3 interaction_sphere_center(2.0f, 0.0f, 0.0f);

// ==================================================================
// SHADERS DE OPENGL
// ==================================================================
const char* sphere_vertex_shader_source = R"glsl(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    void main() { gl_Position = projection * view * model * vec4(aPos, 1.0); }
)glsl";

const char* sphere_fragment_shader_source = R"glsl(
    #version 330 core
    out vec4 FragColor;
    uniform vec4 uColor;
    void main() { FragColor = uColor; }
)glsl";

const char* hair_vertex_shader_source = R"glsl(
    #version 330 core
    layout (location = 0) in vec3 aPos; // Posición del vértice del pelo (offset)
    layout (location = 1) in vec3 aColor; // Color por instancia
    layout (location = 2) in vec3 aOffset; // Posición base de la instancia

    uniform mat4 view;
    uniform mat4 projection;

    out vec3 vColor;

    void main() {
        gl_Position = projection * view * vec4(aPos + aOffset, 1.0);
        vColor = aColor;
    }
)glsl";

const char* hair_fragment_shader_source = R"glsl(
    #version 330 core
    in vec3 vColor;
    out vec4 FragColor;
    void main() { FragColor = vec4(vColor, 1.0); }
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
    GLFWwindow* window = glfwCreateWindow(WIN_SIZE, WIN_SIZE, "Fur Simulation", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    glfwSwapInterval(0); // Desactivar V-Sync para medir el rendimiento máximo

    InputHandler input_handler(camera, sphere_center, interaction_sphere_center);
    input_handler.setup_callbacks(window);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // 2. Inicializar CUDA y la interoperabilidad con OpenGL
    CHECK_CUDA(cudaGLSetGLDevice(0));

    // 3. Compilar Shaders de OpenGL
    GLuint sphere_vs = glCreateShader(GL_VERTEX_SHADER); glShaderSource(sphere_vs, 1, &sphere_vertex_shader_source, NULL); glCompileShader(sphere_vs);
    GLuint sphere_fs = glCreateShader(GL_FRAGMENT_SHADER); glShaderSource(sphere_fs, 1, &sphere_fragment_shader_source, NULL); glCompileShader(sphere_fs);
    GLuint sphereShaderProgram = glCreateProgram(); glAttachShader(sphereShaderProgram, sphere_vs); glAttachShader(sphereShaderProgram, sphere_fs); glLinkProgram(sphereShaderProgram); glDeleteShader(sphere_vs); glDeleteShader(sphere_fs);

    GLuint hair_vs = glCreateShader(GL_VERTEX_SHADER); glShaderSource(hair_vs, 1, &hair_vertex_shader_source, NULL); glCompileShader(hair_vs);
    GLuint hair_fs = glCreateShader(GL_FRAGMENT_SHADER); glShaderSource(hair_fs, 1, &hair_fragment_shader_source, NULL); glCompileShader(hair_fs);
    GLuint hairShaderProgram = glCreateProgram(); glAttachShader(hairShaderProgram, hair_vs); glAttachShader(hairShaderProgram, hair_fs); glLinkProgram(hairShaderProgram); glDeleteShader(hair_vs); glDeleteShader(hair_fs);

    // 4. Crear la malla de las esferas
    std::vector<float> sphere_vertices; std::vector<unsigned int> sphere_indices;
    create_sphere(sphere_vertices, sphere_indices, 1.0f, 32, 32);
    GLuint sphereVAO, sphereVBO, sphereEBO;
    glGenVertexArrays(1, &sphereVAO); glGenBuffers(1, &sphereVBO); glGenBuffers(1, &sphereEBO);
    glBindVertexArray(sphereVAO);
    glBindBuffer(GL_ARRAY_BUFFER, sphereVBO);
    glBufferData(GL_ARRAY_BUFFER, sphere_vertices.size() * sizeof(float), sphere_vertices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphereEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphere_indices.size() * sizeof(unsigned int), sphere_indices.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glBindVertexArray(0);

    // 5. Crear los pelos y reservar memoria en GPU
    std::vector<Hair> h_hairs(NUM_HAIRS);
    for (int i = 0; i < NUM_HAIRS; ++i) {
        vec3 rand_point = normalize(vec3( (float)rand() / RAND_MAX * 2.0f - 1.0f, (float)rand() / RAND_MAX * 2.0f - 1.0f, (float)rand() / RAND_MAX * 2.0f - 1.0f));
        h_hairs[i].anchor_direction = rand_point;
        h_hairs[i].color = vec3(0.6f, 0.4f, 0.2f);
        for (int j = 0; j < HAIR_SEGMENTS; ++j) {
            h_hairs[i].pos[j] = rand_point * (SPHERE_RADIUS + (float)j / (HAIR_SEGMENTS - 1) * HAIR_LENGTH);
            h_hairs[i].old_pos[j] = h_hairs[i].pos[j];
        }
    }
    Hair* d_hairs; CHECK_CUDA(cudaMalloc(&d_hairs, NUM_HAIRS * sizeof(Hair)));
    CHECK_CUDA(cudaMemcpy(d_hairs, h_hairs.data(), NUM_HAIRS * sizeof(Hair), cudaMemcpyHostToDevice));

    // 6. Crear VBOs para el pelo y registrarlo con CUDA
    GLuint hairPosVBO, hairColorVBO;
    glGenBuffers(1, &hairPosVBO);
    glBindBuffer(GL_ARRAY_BUFFER, hairPosVBO);
    glBufferData(GL_ARRAY_BUFFER, NUM_HAIRS * HAIR_SEGMENTS * sizeof(vec3), NULL, GL_DYNAMIC_DRAW);
    struct cudaGraphicsResource* cuda_hair_vbo_resource;
    CHECK_CUDA(cudaGraphicsGLRegisterBuffer(&cuda_hair_vbo_resource, hairPosVBO, cudaGraphicsMapFlagsNone));

    std::vector<vec3> hair_colors;
    for (int i = 0; i < NUM_HAIRS; ++i) {
        for (int j = 0; j < HAIR_SEGMENTS; ++j) {
            hair_colors.push_back(h_hairs[i].color);
        }
    }
    glGenBuffers(1, &hairColorVBO);
    glBindBuffer(GL_ARRAY_BUFFER, hairColorVBO);
    glBufferData(GL_ARRAY_BUFFER, hair_colors.size() * sizeof(vec3), hair_colors.data(), GL_STATIC_DRAW);

    // 7. Crear VAO para el pelo y pre-calcular los arrays para glMultiDrawArrays
    GLuint hairVAO;
    glGenVertexArrays(1, &hairVAO);
    glBindVertexArray(hairVAO);
    
    glBindBuffer(GL_ARRAY_BUFFER, hairPosVBO);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, hairColorVBO);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);

    std::vector<GLint> first(NUM_HAIRS);
    std::vector<GLsizei> count(NUM_HAIRS, HAIR_SEGMENTS);
    for (int i = 0; i < NUM_HAIRS; ++i) {
        first[i] = i * HAIR_SEGMENTS;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    double lastTime = 0.0;
    int frameCount = 0;
    double lastFPSTime = 0.0;
    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_HAIRS + threadsPerBlock - 1) / threadsPerBlock;

    // 8. Bucle principal
    while (!glfwWindowShouldClose(window)) {
        double currentTime = glfwGetTime();
        float dt = (float)(currentTime - lastTime);
        lastTime = currentTime;
        if (dt > 0.02f) dt = 0.02f;

        frameCount++;
        if (currentTime - lastFPSTime >= 1.0) {
            char title[256];
            sprintf(title, "Fur Simulation | %.2f FPS", (float)frameCount / (currentTime - lastFPSTime));
            glfwSetWindowTitle(window, title);
            frameCount = 0;
            lastFPSTime = currentTime;
        }

        vec3 old_sphere_center = sphere_center;
        input_handler.handle_input(window, dt);
        vec3 sphere_velocity = (sphere_center - old_sphere_center) / dt;

        // --- Mapear VBO para que CUDA lo use ---
        vec3* d_vbo_ptr;
        size_t num_bytes;
        CHECK_CUDA(cudaGraphicsMapResources(1, &cuda_hair_vbo_resource, 0));
        CHECK_CUDA(cudaGraphicsResourceGetMappedPointer((void**)&d_vbo_ptr, &num_bytes, cuda_hair_vbo_resource));

        // --- Actualización de la física con CUDA ---
        size_t shared_mem_size = threadsPerBlock * sizeof(Hair);
        update_hair_kernel<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(d_hairs, d_vbo_ptr, dt, sphere_center, sphere_velocity, interaction_sphere_center, input_handler.interaction_active);
        CHECK_CUDA(cudaPeekAtLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        // --- Desmapear VBO ---
        CHECK_CUDA(cudaGraphicsUnmapResources(1, &cuda_hair_vbo_resource, 0));

        // --- Renderizado con OpenGL ---
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);

        mat4 projection = perspective(45.0f, (float)display_w / (float)display_h, 0.1f, 100.0f);
        mat4 view = camera.getViewMatrix();

        glUseProgram(sphereShaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(sphereShaderProgram, "view"), 1, GL_FALSE, &view.m[0]);
        glUniformMatrix4fv(glGetUniformLocation(sphereShaderProgram, "projection"), 1, GL_FALSE, &projection.m[0]);

        // Dibujar la esfera principal
        mat4 model; model.m[0] = SPHERE_RADIUS; model.m[5] = SPHERE_RADIUS; model.m[10] = SPHERE_RADIUS;
        model.m[12] = sphere_center.x; model.m[13] = sphere_center.y; model.m[14] = sphere_center.z;
        glUniformMatrix4fv(glGetUniformLocation(sphereShaderProgram, "model"), 1, GL_FALSE, &model.m[0]);
        glUniform4f(glGetUniformLocation(sphereShaderProgram, "uColor"), 0.8f, 0.7f, 0.6f, 1.0f);
        glBindVertexArray(sphereVAO);
        glDrawElements(GL_TRIANGLES, sphere_indices.size(), GL_UNSIGNED_INT, 0);

        // Dibujar la esfera de interacción
        if (input_handler.interaction_active) {
            mat4 interaction_model;
            interaction_model.m[0] = INTERACTION_SPHERE_RADIUS; interaction_model.m[5] = INTERACTION_SPHERE_RADIUS; interaction_model.m[10] = INTERACTION_SPHERE_RADIUS;
            interaction_model.m[12] = interaction_sphere_center.x; interaction_model.m[13] = interaction_sphere_center.y; interaction_model.m[14] = interaction_sphere_center.z;
            glUniformMatrix4fv(glGetUniformLocation(sphereShaderProgram, "model"), 1, GL_FALSE, &interaction_model.m[0]);
            glUniform4f(glGetUniformLocation(sphereShaderProgram, "uColor"), 0.2f, 0.8f, 0.2f, 0.5f);
            glDrawElements(GL_TRIANGLES, sphere_indices.size(), GL_UNSIGNED_INT, 0);
        }

        // Dibujar los pelos con una sola llamada a glMultiDrawArrays
        glUseProgram(hairShaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(hairShaderProgram, "view"), 1, GL_FALSE, &view.m[0]);
        glUniformMatrix4fv(glGetUniformLocation(hairShaderProgram, "projection"), 1, GL_FALSE, &projection.m[0]);

        glBindVertexArray(hairVAO);
        glMultiDrawArrays(GL_LINE_STRIP, first.data(), count.data(), NUM_HAIRS);
        glBindVertexArray(0);

        glfwSwapBuffers(window);
    }

    // 9. Limpieza
    CHECK_CUDA(cudaFree(d_hairs));
    CHECK_CUDA(cudaGraphicsUnregisterResource(cuda_hair_vbo_resource));
    glDeleteVertexArrays(1, &sphereVAO); 
    glDeleteBuffers(1, &sphereVBO); 
    glDeleteBuffers(1, &sphereEBO);
    glDeleteVertexArrays(1, &hairVAO);
    glDeleteBuffers(1, &hairPosVBO);
    glDeleteBuffers(1, &hairColorVBO);
    glDeleteProgram(sphereShaderProgram); 
    glDeleteProgram(hairShaderProgram);
    
    glfwDestroyWindow(window); 
    glfwTerminate();

    return 0;
}
