#include <iostream>
#include <vector>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
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

// --- OPTIMIZED CUDA ERROR CHECKING (Only in debug) ---
#ifdef DEBUG
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#define CHECK_CUDA_KERNEL() CHECK_CUDA(cudaPeekAtLastError())
#else
#define CHECK_CUDA(call) call
#define CHECK_CUDA_KERNEL()
#endif

// --- Estado de la cámara y la caja ---
Camera camera(vec3(0.0f, 0.0f, 20.0f));
vec3 box_center(0.0f, 0.0f, 0.0f);

// CUDA-OpenGL interop resources
struct CudaGLResources {
    cudaGraphicsResource* position_resource;
    cudaGraphicsResource* color_resource;
    GLuint position_vbo;
    GLuint color_vbo;
    bool initialized = false;
} gl_resources;

// Optimized kernel launch parameters
struct OptimizedKernelParams {
    int threadsPerBlock;
    int blocksPerGrid;

    OptimizedKernelParams(int num_elements) {
        // Conservative optimization to maintain physics behavior
        threadsPerBlock = 256; // Keep safe value that works well
        blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    }
} kernel_params(NUM_BALLS);

// Same shaders as before
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

// Initialize CUDA-OpenGL interoperability
void initCudaGLInterop(GLuint instanceVBO) {
    if (gl_resources.initialized) return;

    // Register OpenGL buffer with CUDA
    CHECK_CUDA(cudaGraphicsGLRegisterBuffer(&gl_resources.position_resource, instanceVBO, cudaGraphicsMapFlagsWriteDiscard));
    gl_resources.position_vbo = instanceVBO;
    gl_resources.initialized = true;

    std::cout << "CUDA-OpenGL interop initialized" << std::endl;
}

// Map OpenGL buffer for CUDA access
Ball* mapCudaGLBuffer() {
    if (!gl_resources.initialized) return nullptr;

    Ball* cuda_ptr;
    size_t buffer_size;

    CHECK_CUDA(cudaGraphicsMapResources(1, &gl_resources.position_resource, 0));
    CHECK_CUDA(cudaGraphicsResourceGetMappedPointer((void**)&cuda_ptr, &buffer_size, gl_resources.position_resource));

    return cuda_ptr;
}

// Unmap OpenGL buffer
void unmapCudaGLBuffer() {
    if (!gl_resources.initialized) return;
    CHECK_CUDA(cudaGraphicsUnmapResources(1, &gl_resources.position_resource, 0));
}

int main() {
    // 1. Inicialización de GLFW y GLAD (optimizada)
    glfwInitHint(GLFW_PLATFORM, GLFW_PLATFORM_X11);
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
    glfwWindowHint(GLFW_REFRESH_RATE, GLFW_DONT_CARE);

    GLFWwindow* window = glfwCreateWindow(WIN_SIZE, WIN_SIZE, "CUDA OPTIMIZED - Unlimited FPS", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    InputHandler input_handler(camera, box_center);
    input_handler.setup_callbacks(window);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSwapInterval(0); // Disable VSync

    std::cout << "Optimized CUDA version - VSync disabled" << std::endl;
    std::cout << "Kernel params: " << kernel_params.threadsPerBlock << " threads, " << kernel_params.blocksPerGrid << " blocks" << std::endl;

    // 2. Inicializar CUDA (con device properties)
    int deviceId;
    CHECK_CUDA(cudaGetDevice(&deviceId));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, deviceId));
    std::cout << "GPU: " << prop.name << " (SM " << prop.major << "." << prop.minor << ")" << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;

    // 3. Compilar Shaders de OpenGL (sin cambios)
    GLuint vs = glCreateShader(GL_VERTEX_SHADER); glShaderSource(vs, 1, &vertex_shader_source, NULL); glCompileShader(vs);
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER); glShaderSource(fs, 1, &fragment_shader_source, NULL); glCompileShader(fs);
    GLuint shaderProgram = glCreateProgram(); glAttachShader(shaderProgram, vs); glAttachShader(shaderProgram, fs); glLinkProgram(shaderProgram); glDeleteShader(vs); glDeleteShader(fs);

    GLuint line_vs = glCreateShader(GL_VERTEX_SHADER); glShaderSource(line_vs, 1, &line_vertex_shader_source, NULL); glCompileShader(line_vs);
    GLuint line_fs = glCreateShader(GL_FRAGMENT_SHADER); glShaderSource(line_fs, 1, &line_fragment_shader_source, NULL); glCompileShader(line_fs);
    GLuint lineShaderProgram = glCreateProgram(); glAttachShader(lineShaderProgram, line_vs); glAttachShader(lineShaderProgram, line_fs); glLinkProgram(lineShaderProgram); glDeleteShader(line_vs); glDeleteShader(line_fs);

    GLuint cursor_vs = glCreateShader(GL_VERTEX_SHADER); glShaderSource(cursor_vs, 1, &cursor_vertex_shader_source, NULL); glCompileShader(cursor_vs);
    GLuint cursor_fs = glCreateShader(GL_FRAGMENT_SHADER); glShaderSource(cursor_fs, 1, &cursor_fragment_shader_source, NULL); glCompileShader(cursor_fs);
    GLuint cursorShaderProgram = glCreateProgram(); glAttachShader(cursorShaderProgram, cursor_vs); glAttachShader(cursorShaderProgram, cursor_fs); glLinkProgram(cursorShaderProgram); glDeleteShader(cursor_vs); glDeleteShader(cursor_fs);

    // 4. Crear la malla de la esfera (optimizada - menos detalle)
    std::vector<float> sphere_vertices; std::vector<unsigned int> sphere_indices;
    create_sphere(sphere_vertices, sphere_indices, BALL_RADIUS, 8, 8); // Reduced detail: 8x8 instead of 12x12
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

    // VBO para las instancias (datos por pelota) - DYNAMIC for CUDA interop
    glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Ball) * NUM_BALLS, NULL, GL_DYNAMIC_DRAW);

    // Atributo de instancia de posición (aOffset)
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Ball), (void*)offsetof(Ball, pos));
    glVertexAttribDivisor(1, 1);

    // Atributo de instancia de color (aColor)
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Ball), (void*)offsetof(Ball, color));
    glVertexAttribDivisor(2, 1);

    glBindVertexArray(0);

    // Box vertices (sin cambios)
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

    // Cursor (sin cambios)
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

    // 5. Initialize CUDA-OpenGL Interop
    initCudaGLInterop(instanceVBO);

    // 6. Crear las bolas y reservar memoria en GPU (optimizado)
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

    // CUDA memory allocation (pinned memory for faster transfers when needed)
    Ball* d_balls;
    CHECK_CUDA(cudaMalloc(&d_balls, NUM_BALLS * sizeof(Ball)));
    CHECK_CUDA(cudaMemcpy(d_balls, h_balls.data(), NUM_BALLS * sizeof(Ball), cudaMemcpyHostToDevice));

    vec3* d_forces;
    CHECK_CUDA(cudaMalloc(&d_forces, NUM_BALLS * sizeof(vec3)));

    // Pre-allocate and clear forces buffer once
    CHECK_CUDA(cudaMemset(d_forces, 0, NUM_BALLS * sizeof(vec3)));

    // Grid data structures
    unsigned int* d_grid_keys;
    CHECK_CUDA(cudaMalloc(&d_grid_keys, NUM_BALLS * sizeof(unsigned int)));
    unsigned int* d_grid_values;
    CHECK_CUDA(cudaMalloc(&d_grid_values, NUM_BALLS * sizeof(unsigned int)));

    int* d_cell_starts;
    CHECK_CUDA(cudaMalloc(&d_cell_starts, NUM_CELLS * sizeof(int)));
    int* d_cell_ends;
    CHECK_CUDA(cudaMalloc(&d_cell_ends, NUM_CELLS * sizeof(int)));

    glEnable(GL_DEPTH_TEST);

    // Timing and performance tracking
    float lastTime = 0.0f;
    vec3 box_center_prev = box_center;
    vec3 box_velocity(0.0f, 0.0f, 0.0f);
    vec3 box_accel_smoothed(0.0f, 0.0f, 0.0f);
    float accel_smoothing_factor = 0.95f;

    // FPS tracking (optimized)
    double lastFPSTime = glfwGetTime();
    int frameCount = 0;
    double frameStartTime = glfwGetTime();
    int totalFrames = 0;

    // Performance metrics
    float physics_time = 0.0f;
    float render_time = 0.0f;

    // CUDA events for timing
    cudaEvent_t start_physics, stop_physics;
    CHECK_CUDA(cudaEventCreate(&start_physics));
    CHECK_CUDA(cudaEventCreate(&stop_physics));

    std::cout << "Starting optimized render loop..." << std::endl;

    // 6. Bucle principal OPTIMIZADO
    while (!glfwWindowShouldClose(window)) {
        float currentTime = glfwGetTime();
        float dt = currentTime - lastTime;
        lastTime = currentTime;
        if (dt > 0.016f) dt = 0.016f; // Limit dt for stability

        // Optimized FPS calculation
        frameCount++;
        totalFrames++;

        if (currentTime - lastFPSTime >= 0.5) { // Update every 0.5s
            double elapsed = currentTime - lastFPSTime;
            double instantFPS = (double)frameCount / elapsed;
            double averageFPS = (double)totalFrames / (currentTime - frameStartTime);
            double frameTimeMs = elapsed * 1000.0 / frameCount;

            char title[256];
            sprintf(title, "CUDA OPTIMIZED - %.0f FPS | Avg: %.0f | Physics: %.2fms | Render: %.2fms",
                   instantFPS, averageFPS, physics_time, render_time);
            glfwSetWindowTitle(window, title);

            frameCount = 0;
            lastFPSTime = currentTime;
        }

        glfwPollEvents();

        // Box acceleration calculation (unchanged)
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

        // --- OPTIMIZED CUDA PHYSICS PIPELINE ---
        CHECK_CUDA(cudaEventRecord(start_physics, 0));

        // 1. Calculate grid hash for each ball (optimized kernel params)
        vec3 box_min = {box_center.x - BOX_SIZE, box_center.y - BOX_SIZE, box_center.z - BOX_SIZE};
        calculate_grid_hash_kernel<<<kernel_params.blocksPerGrid, kernel_params.threadsPerBlock>>>(d_balls, d_grid_keys, d_grid_values, box_min);
        CHECK_CUDA_KERNEL();

        // 2. Sort balls by grid hash (unchanged - Thrust is already optimized)
        thrust::sort_by_key(thrust::device, d_grid_keys, d_grid_keys + NUM_BALLS, d_grid_values);
        CHECK_CUDA_KERNEL();

        // 3. Find cell boundaries (optimized - async memset)
        CHECK_CUDA(cudaMemsetAsync(d_cell_starts, -1, NUM_CELLS * sizeof(int), 0));
        find_cell_boundaries_kernel<<<kernel_params.blocksPerGrid, kernel_params.threadsPerBlock>>>(d_grid_keys, d_cell_starts, d_cell_ends);
        CHECK_CUDA_KERNEL();

        // 4. Calculate collision forces (optimized kernel params)
        collide_kernel<<<kernel_params.blocksPerGrid, kernel_params.threadsPerBlock>>>(d_balls, d_forces, d_grid_values, d_cell_starts, d_cell_ends, box_center);
        CHECK_CUDA_KERNEL();

        // 5. Integrate and handle wall collisions (optimized kernel params)
        integrate_kernel<<<kernel_params.blocksPerGrid, kernel_params.threadsPerBlock>>>(d_balls, d_forces, dt, box_accel_smoothed, box_center);
        CHECK_CUDA_KERNEL();

        // Record physics end time
        CHECK_CUDA(cudaEventRecord(stop_physics, 0));
        CHECK_CUDA(cudaEventSynchronize(stop_physics));

        float physics_ms;
        CHECK_CUDA(cudaEventElapsedTime(&physics_ms, start_physics, stop_physics));
        physics_time = physics_ms;

        // --- OPTIMIZED RENDERING WITH CUDA-OpenGL INTEROP ---
        auto render_start = std::chrono::high_resolution_clock::now();

        // NO MORE CPU-GPU MEMORY TRANSFERS!
        // Instead, use the data directly in GPU buffer

        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);

        mat4 projection = perspective(45.0f, (float)display_w / (float)display_h, 0.1f, 100.0f);
        mat4 view = camera.getViewMatrix();

        // Mouse interaction (simplified for performance)
        double xpos, ypos;
        glfwGetCursorPos(window, &xpos, &ypos);
        vec3 ray_origin = camera.position;
        vec3 ray_direction = normalize(unproject(vec3(xpos, ypos, 1.0f), view, projection, display_w, display_h) - ray_origin);

        // Box interaction (unchanged but only when needed)
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

        // Ball interaction (only when needed - simplified)
        if (input_handler.interaction_requested) {
            // For performance, we'll copy only when needed
            CHECK_CUDA(cudaMemcpy(h_balls.data(), d_balls, NUM_BALLS * sizeof(Ball), cudaMemcpyDeviceToHost));

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
                CHECK_CUDA_KERNEL();
            }
            input_handler.interaction_requested = false;
        }

        // Area force (only when needed)
        if (input_handler.is_area_force_active && !input_handler.box_drag_modifier_active) {
            float force_distance = 10.0f;
            vec3 force_center = {ray_origin.x + ray_direction.x * force_distance,
                               ray_origin.y + ray_direction.y * force_distance,
                               ray_origin.z + ray_direction.z * force_distance};

            float force_radius = 1.5f;
            float force_strength = 5.0f;

            apply_area_force_kernel<<<kernel_params.blocksPerGrid, kernel_params.threadsPerBlock>>>(d_balls, force_center, ray_direction, force_radius, force_strength);
            CHECK_CUDA_KERNEL();
        }

        // CRITICAL OPTIMIZATION: Update VBO directly from CUDA
        Ball* mapped_buffer = mapCudaGLBuffer();
        if (mapped_buffer) {
            // Copy ball data directly from CUDA to OpenGL buffer
            CHECK_CUDA(cudaMemcpy(mapped_buffer, d_balls, NUM_BALLS * sizeof(Ball), cudaMemcpyDeviceToDevice));
            unmapCudaGLBuffer();
        } else {
            // Fallback: traditional method
            CHECK_CUDA(cudaMemcpy(h_balls.data(), d_balls, NUM_BALLS * sizeof(Ball), cudaMemcpyDeviceToHost));
            glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
            glBufferSubData(GL_ARRAY_BUFFER, 0, NUM_BALLS * sizeof(Ball), h_balls.data());
        }

        // Draw balls (optimized - single instanced call)
        glUseProgram(shaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, &projection.m[0]);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, &view.m[0]);

        glBindVertexArray(sphereVAO);
        glDrawElementsInstanced(GL_TRIANGLES, sphere_indices.size(), GL_UNSIGNED_INT, 0, NUM_BALLS);
        glBindVertexArray(0);

        // Draw box (unchanged)
        glUseProgram(lineShaderProgram);
        mat4 box_model; box_model.m[12] = box_center.x; box_model.m[13] = box_center.y; box_model.m[14] = box_center.z;
        glUniformMatrix4fv(glGetUniformLocation(lineShaderProgram, "projection"), 1, GL_FALSE, &projection.m[0]);
        glUniformMatrix4fv(glGetUniformLocation(lineShaderProgram, "view"), 1, GL_FALSE, &view.m[0]);
        glUniformMatrix4fv(glGetUniformLocation(lineShaderProgram, "model"), 1, GL_FALSE, &box_model.m[0]);
        glBindVertexArray(boxVAO);
        glDrawArrays(GL_LINES, 0, 24);

        // Draw cursor (unchanged)
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

        auto render_end = std::chrono::high_resolution_clock::now();
        render_time = std::chrono::duration<float, std::milli>(render_end - render_start).count();

        glfwSwapBuffers(window);
    }

    // 7. Cleanup
    glfwSetMouseButtonCallback(window, nullptr);
    glfwSetCursorPosCallback(window, nullptr);
    glfwSetKeyCallback(window, nullptr);

    // Cleanup CUDA-OpenGL interop
    if (gl_resources.initialized) {
        CHECK_CUDA(cudaGraphicsUnregisterResource(gl_resources.position_resource));
    }

    // Cleanup CUDA events
    CHECK_CUDA(cudaEventDestroy(start_physics));
    CHECK_CUDA(cudaEventDestroy(stop_physics));

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