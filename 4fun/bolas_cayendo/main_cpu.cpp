#include <iostream>
#include <vector>
#include <algorithm>
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include "glm_helper.h"
#include "camera.h"
#include "sphere.h"
#include "input_handler.h"
#include "ball.h"
#include "simulation_parameters.h"

// Use original CUDA parameters but with different timestep approach
#undef STIFFNESS
#undef DAMPING  
#undef LINEAR_DAMPING
#define STIFFNESS 5000.0f    // Same as CUDA
#define DAMPING 10.0f        // Same as CUDA 
#define LINEAR_DAMPING 0.05f // Increased from 0.02f for CPU stability

// --- Estado de la cámara y la caja ---
Camera camera(vec3(0.0f, 0.0f, 20.0f));
vec3 box_center(0.0f, 0.0f, 0.0f);

// --- CPU Physics Implementation ---
class CPUPhysicsEngine {
private:
    std::vector<Ball> balls;
    std::vector<vec3> forces;
    std::vector<unsigned int> grid_keys;
    std::vector<unsigned int> grid_values;
    std::vector<int> cell_starts;
    std::vector<int> cell_ends;
    
public:
    CPUPhysicsEngine() : balls(NUM_BALLS), forces(NUM_BALLS), 
                        grid_keys(NUM_BALLS), grid_values(NUM_BALLS),
                        cell_starts(NUM_CELLS), cell_ends(NUM_CELLS) {
        initializeBalls();
    }
    
    void initializeBalls() {
        for (int i = 0; i < NUM_BALLS; ++i) {
            balls[i].pos = vec3(
                ((float)rand() / RAND_MAX) * (BOX_SIZE*1.8f) - (BOX_SIZE*0.9f),
                ((float)rand() / RAND_MAX) * (BOX_SIZE*1.8f) - (BOX_SIZE*0.9f),
                ((float)rand() / RAND_MAX) * (BOX_SIZE*1.8f) - (BOX_SIZE*0.9f)
            );
            balls[i].vel = vec3(0.0f, 0.0f, 0.0f);
            balls[i].color = vec3((float)rand() / RAND_MAX, (float)rand() / RAND_MAX, (float)rand() / RAND_MAX);
            forces[i] = vec3(0.0f, 0.0f, 0.0f);
        }
    }
    
    void calculateGridHash(const vec3& box_min) {
        // 1. Calculate grid hash for each ball (equivalent to calculate_grid_hash_kernel)
        for (int i = 0; i < NUM_BALLS; ++i) {
            vec3 pos = balls[i].pos;
            int grid_x = (int)((pos.x - box_min.x) / CELL_SIZE);
            int grid_y = (int)((pos.y - box_min.y) / CELL_SIZE);
            int grid_z = (int)((pos.z - box_min.z) / CELL_SIZE);

            // Clamp to grid dimensions
            grid_x = std::max(0, std::min(grid_x, GRID_DIM_X - 1));
            grid_y = std::max(0, std::min(grid_y, GRID_DIM_Y - 1));
            grid_z = std::max(0, std::min(grid_z, GRID_DIM_Z - 1));

            unsigned int hash = grid_z * (GRID_DIM_X * GRID_DIM_Y) + grid_y * GRID_DIM_X + grid_x;
            
            grid_keys[i] = hash;
            grid_values[i] = i;
        }
    }
    
    void sortByKey() {
        // 2. Sort balls by their grid hash (equivalent to thrust::sort_by_key)
        std::vector<std::pair<unsigned int, unsigned int>> key_value_pairs(NUM_BALLS);
        for (int i = 0; i < NUM_BALLS; ++i) {
            key_value_pairs[i] = {grid_keys[i], grid_values[i]};
        }
        
        std::sort(key_value_pairs.begin(), key_value_pairs.end());
        
        for (int i = 0; i < NUM_BALLS; ++i) {
            grid_keys[i] = key_value_pairs[i].first;
            grid_values[i] = key_value_pairs[i].second;
        }
    }
    
    void findCellBoundaries() {
        // 3. Find cell boundaries (equivalent to find_cell_boundaries_kernel)
        std::fill(cell_starts.begin(), cell_starts.end(), -1);
        
        for (int i = 0; i < NUM_BALLS; ++i) {
            unsigned int current_key = grid_keys[i];
            
            // If key is different from previous, it's the start of a new cell
            if (i == 0 || current_key != grid_keys[i - 1]) {
                cell_starts[current_key] = i;
            }
            // If key is different from next, it's the end of a cell
            if (i == NUM_BALLS - 1 || current_key != grid_keys[i + 1]) {
                cell_ends[current_key] = i + 1;
            }
        }
    }
    
    void calculateCollisionForces(const vec3& box_center) {
        // 4. Calculate collision forces (equivalent to collide_kernel)
        // NOTE: Forces are NOT reset here - they're cleared at end of integrate()
        
        // Use temporary forces to better simulate parallel atomicAdd behavior
        std::vector<vec3> temp_forces(NUM_BALLS, vec3(0.0f, 0.0f, 0.0f));
        
        // Iterate over sorted balls (equivalent to CUDA thread iteration)
        for (int i_sorted = 0; i_sorted < NUM_BALLS; ++i_sorted) {
            int i_original = grid_values[i_sorted];
            vec3 pos1 = balls[i_original].pos;
            vec3 vel1 = balls[i_original].vel;
            
            int grid_x = (int)((pos1.x - (box_center.x - BOX_SIZE)) / CELL_SIZE);
            int grid_y = (int)((pos1.y - (box_center.y - BOX_SIZE)) / CELL_SIZE);
            int grid_z = (int)((pos1.z - (box_center.z - BOX_SIZE)) / CELL_SIZE);
            
            // Iterate over current cell and 26 neighboring cells
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        int nx = grid_x + dx;
                        int ny = grid_y + dy;
                        int nz = grid_z + dz;
                        
                        if (nx < 0 || nx >= GRID_DIM_X || ny < 0 || ny >= GRID_DIM_Y || nz < 0 || nz >= GRID_DIM_Z) continue;
                        
                        unsigned int neighbor_hash = nz * (GRID_DIM_X * GRID_DIM_Y) + ny * GRID_DIM_X + nx;
                        int start = cell_starts[neighbor_hash];
                        if (start == -1) continue; // Empty cell
                        int end = cell_ends[neighbor_hash];
                        
                        // Check collisions with balls in neighboring cell
                        for (int j_sorted = start; j_sorted < end; ++j_sorted) {
                            int j_original = grid_values[j_sorted];
                            if (i_original >= j_original) continue; // Avoid double calculations and self-collision
                            
                            vec3 pos2 = balls[j_original].pos;
                            vec3 vel2 = balls[j_original].vel;
                            
                            vec3 delta_pos = {pos1.x - pos2.x, pos1.y - pos2.y, pos1.z - pos2.z};
                            float dist_sq = dot(delta_pos, delta_pos);
                            
                            if (dist_sq < (2 * BALL_RADIUS) * (2 * BALL_RADIUS) && dist_sq > 1e-9) {
                                float dist = sqrtf(dist_sq);
                                vec3 normal = {delta_pos.x / dist, delta_pos.y / dist, delta_pos.z / dist};
                                
                                // Spring force (Hooke's law)
                                float penetration = (2 * BALL_RADIUS) - dist;
                                vec3 spring_force = {normal.x * STIFFNESS * penetration, normal.y * STIFFNESS * penetration, normal.z * STIFFNESS * penetration};
                                
                                // Damping force
                                vec3 rel_vel = {vel1.x - vel2.x, vel1.y - vel2.y, vel1.z - vel2.z};
                                float closing_vel = dot(rel_vel, normal);
                                vec3 damping_force = {normal.x * -DAMPING * closing_vel, normal.y * -DAMPING * closing_vel, normal.z * -DAMPING * closing_vel};
                                
                                vec3 total_force = {spring_force.x + damping_force.x, spring_force.y + damping_force.y, spring_force.z + damping_force.z};
                                
                                // Accumulate in temporary forces (simulating parallel execution)
                                temp_forces[i_original].x += total_force.x;
                                temp_forces[i_original].y += total_force.y;
                                temp_forces[i_original].z += total_force.z;
                                
                                temp_forces[j_original].x -= total_force.x;
                                temp_forces[j_original].y -= total_force.y;
                                temp_forces[j_original].z -= total_force.z;
                            }
                        }
                    }
                }
            }
        }
        
        // Copy accumulated forces to main forces array (simulating synchronization)
        for (int i = 0; i < NUM_BALLS; ++i) {
            forces[i].x += temp_forces[i].x;
            forces[i].y += temp_forces[i].y;
            forces[i].z += temp_forces[i].z;
        }
    }
    
    void integrate(float dt, const vec3& box_accel, const vec3& box_center) {
        for (int i = 0; i < NUM_BALLS; ++i) {
            // Apply gravity and box inertial force
            vec3 total_force = forces[i];
            total_force.y -= 9.8f; // Gravity (assuming mass = 1)
            total_force.x -= box_accel.x;
            total_force.y -= box_accel.y;
            total_force.z -= box_accel.z;
            
            // Update velocity (Euler integration)
            balls[i].vel.x += total_force.x * dt;
            balls[i].vel.y += total_force.y * dt;
            balls[i].vel.z += total_force.z * dt;
            
            // Apply graduated linear damping (stronger for low velocities)
            float speed = sqrtf(dot(balls[i].vel, balls[i].vel));
            float damping_factor = LINEAR_DAMPING;
            if (speed < 1.0f) {
                damping_factor *= (1.0f + (1.0f - speed) * 2.0f);
            }
            balls[i].vel.x *= (1.0f - damping_factor * dt);
            balls[i].vel.y *= (1.0f - damping_factor * dt);
            balls[i].vel.z *= (1.0f - damping_factor * dt);
            
            // Rest threshold to simulate static friction
            float speed_sq = dot(balls[i].vel, balls[i].vel);
            if (speed_sq < VELOCITY_STOP_THRESHOLD * VELOCITY_STOP_THRESHOLD) {
                balls[i].vel = vec3(0.0f, 0.0f, 0.0f);
            }
            
            // Additional friction when balls are near the floor
            if (balls[i].pos.y < box_center.y - BOX_SIZE + BALL_RADIUS * 1.5f) {
                balls[i].vel.x *= 0.99f;
                balls[i].vel.z *= 0.99f;
            }
            
            // Update position
            balls[i].pos.x += balls[i].vel.x * dt;
            balls[i].pos.y += balls[i].vel.y * dt;
            balls[i].pos.z += balls[i].vel.z * dt;
            
            // Wall collisions with friction
            // Floor
            if (balls[i].pos.y < box_center.y - BOX_SIZE + BALL_RADIUS) {
                balls[i].pos.y = box_center.y - BOX_SIZE + BALL_RADIUS;
                balls[i].vel.y *= -RESTITUTION;
                if (abs(balls[i].vel.y) < 0.1f) {
                    balls[i].vel.x *= 0.8f;
                    balls[i].vel.z *= 0.8f;
                }
                balls[i].vel.x *= (1.0f - FRICTION);
                balls[i].vel.z *= (1.0f - FRICTION);
            }
            // Ceiling
            if (balls[i].pos.y > box_center.y + BOX_SIZE - BALL_RADIUS) {
                balls[i].pos.y = box_center.y + BOX_SIZE - BALL_RADIUS;
                balls[i].vel.y *= -RESTITUTION;
                balls[i].vel.x *= (1.0f - FRICTION);
                balls[i].vel.z *= (1.0f - FRICTION);
            }
            // Wall -X
            if (balls[i].pos.x < box_center.x - BOX_SIZE + BALL_RADIUS) {
                balls[i].pos.x = box_center.x - BOX_SIZE + BALL_RADIUS;
                balls[i].vel.x *= -RESTITUTION;
                balls[i].vel.y *= (1.0f - FRICTION);
                balls[i].vel.z *= (1.0f - FRICTION);
            }
            // Wall +X
            if (balls[i].pos.x > box_center.x + BOX_SIZE - BALL_RADIUS) {
                balls[i].pos.x = box_center.x + BOX_SIZE - BALL_RADIUS;
                balls[i].vel.x *= -RESTITUTION;
                balls[i].vel.y *= (1.0f - FRICTION);
                balls[i].vel.z *= (1.0f - FRICTION);
            }
            // Wall -Z
            if (balls[i].pos.z < box_center.z - BOX_SIZE + BALL_RADIUS) {
                balls[i].pos.z = box_center.z - BOX_SIZE + BALL_RADIUS;
                balls[i].vel.z *= -RESTITUTION;
                balls[i].vel.x *= (1.0f - FRICTION);
                balls[i].vel.y *= (1.0f - FRICTION);
            }
            // Wall +Z
            if (balls[i].pos.z > box_center.z + BOX_SIZE - BALL_RADIUS) {
                balls[i].pos.z = box_center.z + BOX_SIZE - BALL_RADIUS;
                balls[i].vel.z *= -RESTITUTION;
                balls[i].vel.x *= (1.0f - FRICTION);
                balls[i].vel.y *= (1.0f - FRICTION);
            }
            
            // Clear forces for next step (equivalent to CUDA kernel)
            forces[i] = {0.0f, 0.0f, 0.0f};
        }
    }
    
    void updatePhysics(float dt, const vec3& box_accel, const vec3& box_center) {
        // Use smaller timesteps for CPU stability with high ball density
        const int CPU_SUBSTEPS = 2;
        float sub_dt = dt / CPU_SUBSTEPS;
        
        for (int substep = 0; substep < CPU_SUBSTEPS; ++substep) {
            vec3 box_min = vec3(box_center.x - BOX_SIZE, box_center.y - BOX_SIZE, box_center.z - BOX_SIZE);
            
            // --- Replicating exact CUDA pipeline ---
            
            // 1. Calculate grid hash for each ball
            calculateGridHash(box_min);
            
            // 2. Sort balls by their grid hash
            sortByKey();
            
            // 3. Find cell boundaries
            findCellBoundaries();
            
            // 4. Calculate collision forces
            calculateCollisionForces(box_center);
            
            // 5. Integrate and handle wall collisions
            integrate(sub_dt, box_accel, box_center);
        }
    }
    
    void applyImpulse(int ball_index, const vec3& impulse) {
        if (ball_index >= 0 && ball_index < NUM_BALLS) {
            balls[ball_index].vel.x += impulse.x;
            balls[ball_index].vel.y += impulse.y;
            balls[ball_index].vel.z += impulse.z;
        }
    }
    
    void applyAreaForce(const vec3& force_center, const vec3& force_direction, float force_radius, float force_strength) {
        for (int i = 0; i < NUM_BALLS; ++i) {
            vec3 diff = vec3(balls[i].pos.x - force_center.x, balls[i].pos.y - force_center.y, balls[i].pos.z - force_center.z);
            float distance = sqrtf(dot(diff, diff));
            
            if (distance <= force_radius && distance > 0.01f) {
                float intensity = 1.0f - (distance / force_radius);
                intensity = intensity * intensity;
                
                vec3 force = vec3(force_direction.x * force_strength * intensity,
                                force_direction.y * force_strength * intensity,
                                force_direction.z * force_strength * intensity);
                
                balls[i].vel.x += force.x;
                balls[i].vel.y += force.y;
                balls[i].vel.z += force.z;
            }
        }
    }
    
    const std::vector<Ball>& getBalls() const { return balls; }
    std::vector<Ball>& getBalls() { return balls; }
};

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
    GLFWwindow* window = glfwCreateWindow(WIN_SIZE, WIN_SIZE, "Bolas Cayendo con CPU (Benchmark - VSync OFF)", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    InputHandler input_handler(camera, box_center);
    input_handler.setup_callbacks(window);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSwapInterval(0); // Disable VSync for maximum FPS

    // 2. Inicializar Physics Engine
    CPUPhysicsEngine physics_engine;

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
    create_sphere(sphere_vertices, sphere_indices, BALL_RADIUS, 12, 12);
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
    glVertexAttribDivisor(1, 1);

    // Atributo de instancia de color (aColor)
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

    glEnable(GL_DEPTH_TEST);
    float lastTime = 0.0f;
    vec3 box_center_prev = box_center;
    vec3 box_velocity(0.0f, 0.0f, 0.0f);
    vec3 box_accel_smoothed(0.0f, 0.0f, 0.0f);
    float accel_smoothing_factor = 0.95f;

    // Variables para el contador de FPS
    double lastFPSTime = 0.0;
    int frameCount = 0;

    // 6. Bucle principal
    while (!glfwWindowShouldClose(window)) {
        float currentTime = glfwGetTime(); float dt = currentTime - lastTime; lastTime = currentTime;
        if (dt > 0.016f) dt = 0.016f; // Limitar dt para estabilidad

        // Calcular y mostrar FPS (más frecuente para benchmarking)
        frameCount++;
        if (currentTime - lastFPSTime >= 0.5) { // Update every 0.5s for more responsive feedback
            char title[256];
            sprintf(title, "CPU Benchmark - %.1f FPS (%.2f ms)", (double)frameCount / (currentTime - lastFPSTime), 1000.0 * (currentTime - lastFPSTime) / frameCount);
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

        // --- Actualización de la física con CPU ---
        physics_engine.updatePhysics(dt, box_accel_smoothed, box_center);

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
            const auto& balls = physics_engine.getBalls();
            float closest_dist = 1e10f;
            int hit_ball_index = -1;
            for (int i = 0; i < NUM_BALLS; ++i) {
                vec3 oc = {balls[i].pos.x - ray_origin.x, balls[i].pos.y - ray_origin.y, balls[i].pos.z - ray_origin.z};
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
                physics_engine.applyImpulse(hit_ball_index, impulse);
            }
            input_handler.interaction_requested = false;
        }

        // Aplicar fuerza en área si se está arrastrando
        if (input_handler.is_area_force_active && !input_handler.box_drag_modifier_active) {
            float force_distance = 10.0f;
            vec3 force_center = {ray_origin.x + ray_direction.x * force_distance, 
                               ray_origin.y + ray_direction.y * force_distance, 
                               ray_origin.z + ray_direction.z * force_distance};
            
            float force_radius = 1.5f;
            float force_strength = 5.0f;
            
            physics_engine.applyAreaForce(force_center, ray_direction, force_radius, force_strength);
        }

        // Dibujar las bolas (con instancing)
        glUseProgram(shaderProgram);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, &projection.m[0]);
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, &view.m[0]);
        
        // Actualizar el VBO de instancias con las nuevas posiciones y colores
        glBindBuffer(GL_ARRAY_BUFFER, instanceVBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, NUM_BALLS * sizeof(Ball), physics_engine.getBalls().data());

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