#include <iostream>
#include <vector>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

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
// SHADERS DE OPENGL (Sin cambios)
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

const char* cursor_vertex_shader_source = R"glsl(
    #version 330 core
    uniform vec2 u_cursorPos;
    void main() {
        gl_Position = vec4(u_cursorPos, 0.0, 1.0);
        gl_PointSize = 10.0;
    }
)glsl";

const char* cursor_fragment_shader_source = R"glsl(
    #version 330 core
    out vec4 FragColor;
    void main() {
        vec2 center = gl_PointCoord - 0.5;
        if (length(center) > 0.5) discard;
        FragColor = vec4(1.0, 1.0, 1.0, 1.0);
    }
)glsl";

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

// Función para detectar intersecciones rayo-caja
bool ray_box_intersection(vec3 ray_origin, vec3 ray_direction, vec3 box_min, vec3 box_max, float& t) {
    vec3 inv_dir = {1.0f / ray_direction.x, 1.0f / ray_direction.y, 1.0f / ray_direction.z};
    vec3 tmin_vec = {(box_min.x - ray_origin.x) * inv_dir.x, (box_min.y - ray_origin.y) * inv_dir.y, (box_min.z - ray_origin.z) * inv_dir.z};
    vec3 tmax_vec = {(box_max.x - ray_origin.x) * inv_dir.x, (box_max.y - ray_origin.y) * inv_dir.y, (box_max.z - ray_origin.z) * inv_dir.z};
    
    float tmin = fmax(fmax(fmin(tmin_vec.x, tmax_vec.x), fmin(tmin_vec.y, tmax_vec.y)), fmin(tmin_vec.z, tmax_vec.z));
    float tmax = fmin(fmin(fmax(tmin_vec.x, tmax_vec.x), fmax(tmin_vec.y, tmax_vec.y)), fmax(tmin_vec.z, tmax_vec.z));
    
    if (tmax < 0 || tmin > tmax) {
        return false;
    }
    
    t = (tmin > 0) ? tmin : tmax;
    return t > 0;
}

// ==================================================================
// FUNCIÓN MAIN
// ==================================================================
int main() {
    // 1. Inicialización de GLFW y GLAD
    glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(WIN_SIZE, WIN_SIZE, "Bolas Cayendo con CUDA (Spatial Grid)", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

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

    GLuint cursor_vs = glCreateShader(GL_VERTEX_SHADER); glShaderSource(cursor_vs, 1, &cursor_vertex_shader_source, NULL); glCompileShader(cursor_vs);
    GLuint cursor_fs = glCreateShader(GL_FRAGMENT_SHADER); glShaderSource(cursor_fs, 1, &cursor_fragment_shader_source, NULL); glCompileShader(cursor_fs);
    GLuint cursorShaderProgram = glCreateProgram(); glAttachShader(cursorShaderProgram, cursor_vs); glAttachShader(cursorShaderProgram, cursor_fs); glLinkProgram(cursorShaderProgram); glDeleteShader(cursor_vs); glDeleteShader(cursor_fs);

    // 4. Crear la malla de la esfera y la caja
    std::vector<float> sphere_vertices; std::vector<unsigned int> sphere_indices;
    create_sphere(sphere_vertices, sphere_indices, BALL_RADIUS, 12, 12); // Menos detalle para más rendimiento
    GLuint sphereVAO, sphereVBO, sphereEBO;
    glGenVertexArrays(1, &sphereVAO); glGenBuffers(1, &sphereVBO); glGenBuffers(1, &sphereEBO);
    
    GLuint instanceVBO;
    glGenBuffers(1, &instanceVBO);

    glBindVertexArray(sphereVAO);

    // VBO para la malla de la esfera
    glBindBuffer(GL_ARRAY_BUFFER, sphereVBO);
    glBufferData(GL_ARRAY_BUFFER, sphere_vertices.size() * sizeof(float), sphere_vertices.data(), GL_STATIC_DRAW);
    
    // EBO para los índices de la esfera
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphereEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sphere_indices.size() * sizeof(unsigned int), sphere_indices.data(), GL_STATIC_DRAW);

    // Atributo de vértice (aPos)
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    // VBO para las instancias (datos por pelota)
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Ball) * NUM_BALLS, NULL, GL_DYNAMIC_DRAW);

    // Atributo de instancia de posición (aOffset)
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Ball), (void*)offsetof(Ball, pos));
    glVertexAttribDivisor(1, 1); // Actualizar por instancia

    // Atributo de instancia de color (aColor)
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Ball), (void*)offsetof(Ball, color));
    glVertexAttribDivisor(2, 1); // Actualizar por instancia

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

    // Malla para el cursor (un solo punto)
    GLuint cursorVAO, cursorVBO;
    float cursor_pos[] = { 0.0f, 0.0f };
    glGenVertexArrays(1, &cursorVAO);
    glGenBuffers(1, &cursorVBO);
    glBindVertexArray(cursorVAO);
    glBindBuffer(GL_ARRAY_BUFFER, cursorVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(cursor_pos), cursor_pos, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);

    // 5. Crear las bolas y reservar memoria en GPU
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

    vec3* d_forces; CHECK_CUDA(cudaMalloc(&d_forces, NUM_BALLS * sizeof(vec3)));
    CHECK_CUDA(cudaMemset(d_forces, 0, NUM_BALLS * sizeof(vec3)));

    unsigned int* d_grid_keys; CHECK_CUDA(cudaMalloc(&d_grid_keys, NUM_BALLS * sizeof(unsigned int)));
    unsigned int* d_grid_values; CHECK_CUDA(cudaMalloc(&d_grid_values, NUM_BALLS * sizeof(unsigned int)));
    
    int* d_cell_starts; CHECK_CUDA(cudaMalloc(&d_cell_starts, NUM_CELLS * sizeof(int)));
    int* d_cell_ends; CHECK_CUDA(cudaMalloc(&d_cell_ends, NUM_CELLS * sizeof(int)));

    glEnable(GL_DEPTH_TEST);
    float lastTime = 0.0f;
    vec3 box_center_prev = box_center;
    vec3 box_velocity(0.0f, 0.0f, 0.0f);
    vec3 box_accel_smoothed(0.0f, 0.0f, 0.0f);
    float accel_smoothing_factor = 0.95f;

    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_BALLS + threadsPerBlock - 1) / threadsPerBlock;

    // Variables para el contador de FPS
    double lastFPSTime = 0.0;
    int frameCount = 0;

    // 6. Bucle principal
    while (!glfwWindowShouldClose(window)) {
        float currentTime = glfwGetTime(); float dt = currentTime - lastTime; lastTime = currentTime;
        if (dt > 0.016f) dt = 0.016f; // Limitar dt para estabilidad

        // Calcular y mostrar FPS
        frameCount++;
        if (currentTime - lastFPSTime >= 1.0) {
            char title[256];
            sprintf(title, "Bolas Cayendo con CUDA (Spatial Grid) - %.2f FPS", (double)frameCount / (currentTime - lastFPSTime));
            glfwSetWindowTitle(window, title);
            frameCount = 0;
            lastFPSTime = currentTime;
        }

        glfwPollEvents();

        // Calcular aceleración de la caja
        vec3 box_accel_raw(0.0f, 0.0f, 0.0f);
        if (dt > 1e-6) {
            vec3 current_velocity = {(box_center.x - box_center_prev.x) / dt, (box_center.y - box_center_prev.y) / dt, (box_center.z - box_center_prev.z) / dt};
            box_accel_raw = {(current_velocity.x - box_velocity.x) / dt, (current_velocity.y - box_velocity.y) / dt, (current_velocity.z - box_velocity.z) / dt};
            box_velocity = current_velocity;
        }
        box_center_prev = box_center;
        box_accel_smoothed.x = box_accel_smoothed.x * accel_smoothing_factor + box_accel_raw.x * (1.0f - accel_smoothing_factor);
        box_accel_smoothed.y = box_accel_smoothed.y * accel_smoothing_factor + box_accel_raw.y * (1.0f - accel_smoothing_factor);
        box_accel_smoothed.z = box_accel_smoothed.z * accel_smoothing_factor + box_accel_raw.z * (1.0f - accel_smoothing_factor);

        // --- Actualización de la física con CUDA ---
        
        // 1. Calcular hash de la cuadrícula para cada pelota
        vec3 box_min = {box_center.x - BOX_SIZE, box_center.y - BOX_SIZE, box_center.z - BOX_SIZE};
        calculate_grid_hash_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_balls, d_grid_keys, d_grid_values, box_min);
        CHECK_CUDA(cudaPeekAtLastError());

        // 2. Ordenar las pelotas por su hash de celda
        thrust::sort_by_key(thrust::device, d_grid_keys, d_grid_keys + NUM_BALLS, d_grid_values);
        CHECK_CUDA(cudaPeekAtLastError());

        // 3. Encontrar los límites de las celdas
        CHECK_CUDA(cudaMemset(d_cell_starts, -1, NUM_CELLS * sizeof(int))); // Inicializar a -1
        find_cell_boundaries_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_grid_keys, d_cell_starts, d_cell_ends);
        CHECK_CUDA(cudaPeekAtLastError());

        // 4. Calcular fuerzas de colisión
        collide_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_balls, d_forces, d_grid_values, d_cell_starts, d_cell_ends, box_center);
        CHECK_CUDA(cudaPeekAtLastError());

        // 5. Integrar y manejar colisiones con paredes
        integrate_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_balls, d_forces, dt, box_accel_smoothed, box_center);
        CHECK_CUDA(cudaPeekAtLastError());
        
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(h_balls.data(), d_balls, NUM_BALLS * sizeof(Ball), cudaMemcpyDeviceToHost));

        // --- Renderizado con OpenGL ---
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);

        mat4 projection = perspective(45.0f, (float)display_w / (float)display_h, 0.1f, 100.0f);
        mat4 view = camera.getViewMatrix();

        // Interacción con el ratón
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        vec3 ray_origin = camera.position;
        vec3 ray_direction = normalize(unproject(vec3(xpos, ypos, 1.0f), view, projection, display_w, display_h) - ray_origin);

        // Arrastre de la caja
        if (input_handler.box_grab_requested) {
            vec3 box_min = {box_center.x - BOX_SIZE, box_center.y - BOX_SIZE, box_center.z - BOX_SIZE};
            vec3 box_max = {box_center.x + BOX_SIZE, box_center.y + BOX_SIZE, box_center.z + BOX_SIZE};
            float t;
            if (ray_box_intersection(ray_origin, ray_direction, box_min, box_max, t)) {
                input_handler.is_dragging_box = true;
                vec3 hit_point = {ray_origin.x + ray_direction.x * t, ray_origin.y + ray_direction.y * t, ray_origin.z + ray_direction.z * t};
                input_handler.grab_offset = {hit_point.x - box_center.x, hit_point.y - box_center.y, hit_point.z - box_center.z};
            }
            input_handler.box_grab_requested = false;
        }

        if (input_handler.is_dragging_box) {
            float dist = dot(input_handler.grab_offset - ray_origin, camera.front);
            vec3 target_pos = {ray_origin.x + ray_direction.x * dist, ray_origin.y + ray_direction.y * dist, ray_origin.z + ray_direction.z * dist};
            box_center = {target_pos.x - input_handler.grab_offset.x, target_pos.y - input_handler.grab_offset.y, target_pos.z - input_handler.grab_offset.z};
        }

        // Disparar pelotas
        if (input_handler.interaction_requested) {
            float closest_dist = 1e10f;
            int hit_ball_index = -1;
            for (int i = 0; i < NUM_BALLS; ++i) {
                vec3 oc = {h_balls[i].pos.x - ray_origin.x, h_balls[i].pos.y - ray_origin.y, h_balls[i].pos.z - ray_origin.z};
                float b = dot(oc, ray_direction);
                float c = dot(oc, oc) - BALL_RADIUS * BALL_RADIUS;
                float discriminant = b * b - c;
                if (discriminant > 0) {
                    float dist = b - sqrtf(discriminant);
                    if (dist > 0 && dist < closest_dist) {
                        closest_dist = dist;
                        hit_ball_index = i;
                    }
                }
            }

            if (hit_ball_index != -1) {
                float impulse_strength = 20.0f;
                vec3 impulse = {ray_direction.x * impulse_strength, ray_direction.y * impulse_strength, ray_direction.z * impulse_strength};
                apply_impulse_kernel<<<1, 1>>>(d_balls, hit_ball_index, impulse);
                CHECK_CUDA(cudaPeekAtLastError());
            }
            input_handler.interaction_requested = false;
        }

        // Dibujar las bolas (con instancing)
        glUseProgram(shaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, &projection.m[0]);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, &view.m[0]);
        
        // Actualizar el VBO de instancias con las nuevas posiciones y colores
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, NUM_BALLS * sizeof(Ball), h_balls.data());

        // Dibujar todas las pelotas con una sola llamada
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

        // Dibujar el cursor personalizado
        double cursor_x, cursor_y;
        glfwGetCursorPos(window, &cursor_x, &cursor_y);
        float ndc_x = (2.0f * cursor_x) / display_w - 1.0f;
        float ndc_y = 1.0f - (2.0f * cursor_y) / display_h;

        glUseProgram(cursorShaderProgram);
        glUniform2f(glGetUniformLocation(cursorShaderProgram, "u_cursorPos"), ndc_x, ndc_y);
        glEnable(GL_PROGRAM_POINT_SIZE);
        glBindVertexArray(cursorVAO);
        glDrawArrays(GL_POINTS, 0, 1);
        glDisable(GL_PROGRAM_POINT_SIZE);

        glfwSwapBuffers(window);
    }

    // 7. Limpieza
    glfwSetMouseButtonCallback(window, nullptr);
    glfwSetCursorPosCallback(window, nullptr);
    glfwSetKeyCallback(window, nullptr);

    CHECK_CUDA(cudaFree(d_balls));
    CHECK_CUDA(cudaFree(d_forces));
    CHECK_CUDA(cudaFree(d_grid_keys));
    CHECK_CUDA(cudaFree(d_grid_values));
    CHECK_CUDA(cudaFree(d_cell_starts));
    CHECK_CUDA(cudaFree(d_cell_ends));
    CHECK_CUDA(cudaDeviceReset());

    glDeleteVertexArrays(1, &sphereVAO); 
    glDeleteBuffers(1, &sphereVBO); 
    glDeleteBuffers(1, &sphereEBO);
    glDeleteBuffers(1, &instanceVBO);
    glDeleteVertexArrays(1, &boxVAO); 
    glDeleteBuffers(1, &boxVBO);
    glDeleteVertexArrays(1, &cursorVAO);
    glDeleteBuffers(1, &cursorVBO);
    glDeleteProgram(shaderProgram); 
    glDeleteProgram(lineShaderProgram);
    glDeleteProgram(cursorShaderProgram);
    
    glfwDestroyWindow(window); 
    glfwTerminate();

    return 0;
}