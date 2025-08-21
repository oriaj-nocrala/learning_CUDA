#include <iostream>
#include <vector>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>

#include "glm_helper.h"
#include "camera.h"
#include "sphere.h"
#include "input_handler.h"

// --- MACRO PARA VERIFICAR ERRORES DE CUDA ---
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// --- Parámetros Físicos ---
const int WIN_SIZE = 768;
const int NUM_BALLS = 1000;
const float BALL_RADIUS = 0.1f;
const float BOX_SIZE = 5.0f;
const float FRICTION = 0.1f;       // Coeficiente de fricción

// --- Estructura para las bolas ---
struct Ball {
    vec3 pos;
    vec3 vel;
};

// --- Estado del ratón y la caja ---
Camera camera(vec3(0.0f, 0.0f, 20.0f));
vec3 box_center(0.0f, 0.0f, 0.0f);




// ==================================================================
// KERNEL DE CUDA
// ==================================================================
__global__ void update_balls_kernel(Ball* balls, float dt, vec3 box_center, vec3 box_accel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NUM_BALLS) return;

    // 1. Aplicar fuerzas (Gravedad y Fuerza de Inercia)
    balls[i].vel.y -= 9.8f * dt;
    balls[i].vel.x -= box_accel.x * dt;
    balls[i].vel.y -= box_accel.y * dt;
    balls[i].vel.z -= box_accel.z * dt;

    // 2. Colisiones entre bolas
    for (int j = i + 1; j < NUM_BALLS; ++j) {
        vec3& pos1 = balls[i].pos;
        vec3& vel1 = balls[i].vel;
        vec3& pos2 = balls[j].pos;
        vec3& vel2 = balls[j].vel;

        float dist_sq = (pos1.x - pos2.x) * (pos1.x - pos2.x) +
                        (pos1.y - pos2.y) * (pos1.y - pos2.y) +
                        (pos1.z - pos2.z) * (pos1.z - pos2.z);

        if (dist_sq < (2 * BALL_RADIUS) * (2 * BALL_RADIUS) && dist_sq > 1e-9) {
            vec3 normal = {pos1.x - pos2.x, pos1.y - pos2.y, pos1.z - pos2.z};
            float len = sqrtf(dist_sq);
            normal.x /= len; normal.y /= len; normal.z /= len;

            float k = (vel1.x - vel2.x) * normal.x + (vel1.y - vel2.y) * normal.y + (vel1.z - vel2.z) * normal.z;
            if (k < 0) {
                vec3 delta_v = {k * normal.x, k * normal.y, k * normal.z};
                vel1.x -= delta_v.x; vel1.y -= delta_v.y; vel1.z -= delta_v.z;
                vel2.x += delta_v.x; vel2.y += delta_v.y; vel2.z += delta_v.z;
            }
        }
    }

    // 3. Actualizar posición
    balls[i].pos.x += balls[i].vel.x * dt;
    balls[i].pos.y += balls[i].vel.y * dt;
    balls[i].pos.z += balls[i].vel.z * dt;

    // 4. Colisiones con las paredes (con fricción)
    float restitution = -0.7f;
    // Suelo
    if (balls[i].pos.y < box_center.y - BOX_SIZE + BALL_RADIUS) { 
        balls[i].pos.y = box_center.y - BOX_SIZE + BALL_RADIUS; 
        balls[i].vel.y *= restitution;
        balls[i].vel.x *= (1.0f - FRICTION);
        balls[i].vel.z *= (1.0f - FRICTION);
    }
    // Techo
    if (balls[i].pos.y > box_center.y + BOX_SIZE - BALL_RADIUS) { 
        balls[i].pos.y = box_center.y + BOX_SIZE - BALL_RADIUS; 
        balls[i].vel.y *= restitution;
        balls[i].vel.x *= (1.0f - FRICTION);
        balls[i].vel.z *= (1.0f - FRICTION);
    }
    // Pared -X
    if (balls[i].pos.x < box_center.x - BOX_SIZE + BALL_RADIUS) { 
        balls[i].pos.x = box_center.x - BOX_SIZE + BALL_RADIUS; 
        balls[i].vel.x *= restitution;
        balls[i].vel.y *= (1.0f - FRICTION);
        balls[i].vel.z *= (1.0f - FRICTION);
    }
    // Pared +X
    if (balls[i].pos.x > box_center.x + BOX_SIZE - BALL_RADIUS) { 
        balls[i].pos.x = box_center.x + BOX_SIZE - BALL_RADIUS; 
        balls[i].vel.x *= restitution;
        balls[i].vel.y *= (1.0f - FRICTION);
        balls[i].vel.z *= (1.0f - FRICTION);
    }
    // Pared -Z
    if (balls[i].pos.z < box_center.z - BOX_SIZE + BALL_RADIUS) { 
        balls[i].pos.z = box_center.z - BOX_SIZE + BALL_RADIUS; 
        balls[i].vel.z *= restitution;
        balls[i].vel.x *= (1.0f - FRICTION);
        balls[i].vel.y *= (1.0f - FRICTION);
    }
    // Pared +Z
    if (balls[i].pos.z > box_center.z + BOX_SIZE - BALL_RADIUS) { 
        balls[i].pos.z = box_center.z + BOX_SIZE - BALL_RADIUS; 
        balls[i].vel.z *= restitution;
        balls[i].vel.x *= (1.0f - FRICTION);
        balls[i].vel.y *= (1.0f - FRICTION);
    }
}

// ==================================================================
// SHADERS DE OPENGL
// ==================================================================
const char* vertex_shader_source = R"glsl(
    #version 330 core
    layout (location = 0) in vec3 aPos;
    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;
    void main() { gl_Position = projection * view * model * vec4(aPos, 1.0); }
)glsl";

const char* fragment_shader_source = R"glsl(
    #version 330 core
    out vec4 FragColor;
    void main() { FragColor = vec4(1.0, 0.5, 0.2, 1.0); }
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

// ==================================================================
// FUNCIÓN MAIN
// ==================================================================
int main() {
    // 1. Inicialización de GLFW y GLAD
    glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11); // Forzar X11 para evitar problemas con Wayland
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(WIN_SIZE, WIN_SIZE, "Bolas Cayendo con CUDA", NULL, NULL);
    glfwMakeContextCurrent(window);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    // --- Configurar el manejador de entrada ---
    InputHandler input_handler(camera, box_center);
    input_handler.setup_callbacks(window);

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSwapInterval(1);

    // 2. Inicializar CUDA
    CHECK_CUDA(cudaSetDevice(0));

    // 3. Compilar Shaders de OpenGL
    GLuint vs = glCreateShader(GL_VERTEX_SHADER); glShaderSource(vs, 1, &vertex_shader_source, NULL); glCompileShader(vs);
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER); glShaderSource(fs, 1, &fragment_shader_source, NULL); glCompileShader(fs);
    GLuint shaderProgram = glCreateProgram(); glAttachShader(shaderProgram, vs); glAttachShader(shaderProgram, fs); glLinkProgram(shaderProgram); glDeleteShader(vs); glDeleteShader(fs);

    GLuint line_vs = glCreateShader(GL_VERTEX_SHADER); glShaderSource(line_vs, 1, &line_vertex_shader_source, NULL); glCompileShader(line_vs);
    GLuint line_fs = glCreateShader(GL_FRAGMENT_SHADER); glShaderSource(line_fs, 1, &line_fragment_shader_source, NULL); glCompileShader(line_fs);
    GLuint lineShaderProgram = glCreateProgram(); glAttachShader(lineShaderProgram, line_vs); glAttachShader(lineShaderProgram, line_fs); glLinkProgram(lineShaderProgram); glDeleteShader(line_vs); glDeleteShader(line_fs);

    // 4. Crear la malla de la esfera
    std::vector<float> sphere_vertices; std::vector<unsigned int> sphere_indices;
    create_sphere(sphere_vertices, sphere_indices, BALL_RADIUS, 20, 20);
    GLuint sphereVAO, sphereVBO, sphereEBO;
    glGenVertexArrays(1, &sphereVAO); glGenBuffers(1, &sphereVBO); glGenBuffers(1, &sphereEBO);
    glBindVertexArray(sphereVAO);
    glBindBuffer(GL_ARRAY_BUFFER, sphereVBO); glBufferData(GL_ARRAY_BUFFER, sphere_vertices.size() * sizeof(float), sphere_vertices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphereEBO); glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphere_indices.size() * sizeof(unsigned int), sphere_indices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0); glEnableVertexAttribArray(0);

    // 5. Crear la malla de la caja (alambre)
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

    // 6. Crear las bolas en Host y Device
    std::vector<Ball> h_balls(NUM_BALLS);
    for (int i = 0; i < NUM_BALLS; ++i) {
        h_balls[i].pos = vec3(
            ((float)rand() / RAND_MAX) * (BOX_SIZE*2-BALL_RADIUS*2) - (BOX_SIZE-BALL_RADIUS),
            ((float)rand() / RAND_MAX) * (BOX_SIZE*2-BALL_RADIUS*2) - (BOX_SIZE-BALL_RADIUS),
            ((float)rand() / RAND_MAX) * (BOX_SIZE*2-BALL_RADIUS*2) - (BOX_SIZE-BALL_RADIUS)
        );
        h_balls[i].vel = vec3(0.0f, 0.0f, 0.0f);
    }
    Ball* d_balls; CHECK_CUDA(cudaMalloc(&d_balls, NUM_BALLS * sizeof(Ball)));
    CHECK_CUDA(cudaMemcpy(d_balls, h_balls.data(), NUM_BALLS * sizeof(Ball), cudaMemcpyHostToDevice));

    glEnable(GL_DEPTH_TEST);
    float lastTime = 0.0f;
    vec3 box_center_prev = box_center;
    vec3 box_velocity(0.0f, 0.0f, 0.0f);
    vec3 box_accel_smoothed(0.0f, 0.0f, 0.0f);
    float accel_smoothing_factor = 0.95f;

    // 7. Bucle principal
    while (!glfwWindowShouldClose(window)) {
        float currentTime = glfwGetTime(); float dt = currentTime - lastTime; lastTime = currentTime;
        if (dt > 0.02f) dt = 0.02f; // Limitar dt para evitar inestabilidad

        glfwPollEvents();

        // Calcular velocidad y aceleración de la caja (suavizada)
        vec3 box_accel_raw(0.0f, 0.0f, 0.0f);
        if (dt > 1e-6) {
            vec3 current_velocity = {
                (box_center.x - box_center_prev.x) / dt,
                (box_center.y - box_center_prev.y) / dt,
                (box_center.z - box_center_prev.z) / dt
            };
            box_accel_raw = {
                (current_velocity.x - box_velocity.x) / dt,
                (current_velocity.y - box_velocity.y) / dt,
                (current_velocity.z - box_velocity.z) / dt
            };
            box_velocity = current_velocity;
        }
        box_center_prev = box_center;

        box_accel_smoothed.x = box_accel_smoothed.x * accel_smoothing_factor + box_accel_raw.x * (1.0f - accel_smoothing_factor);
        box_accel_smoothed.y = box_accel_smoothed.y * accel_smoothing_factor + box_accel_raw.y * (1.0f - accel_smoothing_factor);
        box_accel_smoothed.z = box_accel_smoothed.z * accel_smoothing_factor + box_accel_raw.z * (1.0f - accel_smoothing_factor);

        // --- Actualización de la física con CUDA ---
        int threadsPerBlock = 256;
        int blocksPerGrid = (NUM_BALLS + threadsPerBlock - 1) / threadsPerBlock;
        update_balls_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_balls, dt, box_center, box_accel_smoothed);
        CHECK_CUDA(cudaPeekAtLastError()); CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_balls.data(), d_balls, NUM_BALLS * sizeof(Ball), cudaMemcpyDeviceToHost));

        // --- Renderizado con OpenGL ---
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        mat4 projection = perspective(45.0f, (float)WIN_SIZE / (float)WIN_SIZE, 0.1f, 100.0f);
        mat4 view = camera.getViewMatrix();

        // Dibujar las bolas
        glUseProgram(shaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, &projection.m[0]);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, &view.m[0]);
        glBindVertexArray(sphereVAO);
        for (int i = 0; i < NUM_BALLS; ++i) {
            mat4 model; model.m[12] = h_balls[i].pos.x; model.m[13] = h_balls[i].pos.y; model.m[14] = h_balls[i].pos.z;
            glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, &model.m[0]);
            glDrawElements(GL_TRIANGLES, sphere_indices.size(), GL_UNSIGNED_INT, 0);
        }

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

    // 8. Limpieza
    glfwSetMouseButtonCallback(window, nullptr);
    glfwSetCursorPosCallback(window, nullptr);
    glfwSetKeyCallback(window, nullptr);

    CHECK_CUDA(cudaFree(d_balls));
    CHECK_CUDA(cudaDeviceReset()); // Resetea el dispositivo CUDA

    glDeleteVertexArrays(1, &sphereVAO); 
    glDeleteBuffers(1, &sphereVBO); 
    glDeleteBuffers(1, &sphereEBO);
    glDeleteVertexArrays(1, &boxVAO); 
    glDeleteBuffers(1, &boxVBO);
    glDeleteProgram(shaderProgram); 
    glDeleteProgram(lineShaderProgram);
    
    glfwDestroyWindow(window); 
    glfwTerminate();

    return 0;
}
