#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// --- MACRO PARA VERIFICAR ERRORES DE CUDA ---
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// --- Parámetros de la Simulación ---
const int N = 128;
const int WIN_SIZE = 768;
const float DT = 0.04f;
const float DIFF = 0.0f;
const float VISC = 0.0001f;

// --- Helpers ---
#define SWAP(x0,x) do{ float* tmp=x0; x0=x; x=tmp; }while(0)
__device__ __host__ inline int IX(int x, int y) { return y * (N + 2) + x; }

// ==================================================================
// KERNELS DE SIMULACIÓN CUDA
// ==================================================================
__global__ void add_source_kernel(float *x, const float *s, float dt, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) x[i] += dt * s[i];
}

__global__ void set_bnd_kernel(int b, float *x) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i >= (N+2)*(N+2)) return;
    int col = i % (N+2);
    int row = i / (N+2);

    if (col == 0)   x[i] = b == 1 ? -x[i+1] : x[i+1];
    if (col == N+1) x[i] = b == 1 ? -x[i-1] : x[i-1];
    if (row == 0)   x[i] = b == 2 ? -x[i+(N+2)] : x[i+(N+2)];
    if (row == N+1) x[i] = b == 2 ? -x[i-(N+2)] : x[i-(N+2)];
    
    x[IX(0,0)]     = 0.5f * (x[IX(1,0)]   + x[IX(0,1)]);
    x[IX(0,N+1)]   = 0.5f * (x[IX(1,N+1)] + x[IX(0,N)]);
    x[IX(N+1,0)]   = 0.5f * (x[IX(N,0)]   + x[IX(N+1,1)]);
    x[IX(N+1,N+1)] = 0.5f * (x[IX(N,N+1)] + x[IX(N+1,N)]);
}

__global__ void lin_solve_kernel(int b, float *x, const float *x0, float a, float c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > 0 && j > 0 && i <= N && j <= N) {
        x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i-1, j)] + x[IX(i+1, j)] + x[IX(i, j-1)] + x[IX(i, j+1)])) / c;
    }
}

void set_bnd(int b, float *x){
    int size = (N+2)*(N+2);
    set_bnd_kernel<<<(size+255)/256, 256>>>(b, x);
}

void lin_solve(int b, float *x, float *x0, float a, float c, int iter) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (N + 15) / 16);
    for (int k = 0; k < iter; k++) {
        lin_solve_kernel<<<blocks, threads>>>(b, x, x0, a, c);
        set_bnd(b, x);
    }
}

void diffuse(int b, float *x, float *x0, float diff, float dt, int iter) {
    float a = dt * diff * N * N;
    lin_solve(b, x, x0, a, 1 + 4 * a, iter);
}

__global__ void advect_kernel(int b, float *d, const float *d0, const float *u, const float *v, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && j > 0 && i <= N && j <= N) {
        float x = (float)i - dt * N * u[IX(i,j)];
        float y = (float)j - dt * N * v[IX(i,j)];

        if (x < 0.5f) x = 0.5f; if (x > N + 0.5f) x = N + 0.5f;
        int i0 = (int)x; int i1 = i0 + 1;
        if (y < 0.5f) y = 0.5f; if (y > N + 0.5f) y = N + 0.5f;
        int j0 = (int)y; int j1 = j0 + 1;

        float s1 = x - i0; float s0 = 1.0f - s1;
        float t1 = y - j0; float t0 = 1.0f - t1;

        d[IX(i,j)] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) +
                     s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
    }
}

void advect(int b, float *d, float *d0, float *u, float *v, float dt) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (N + 15) / 16);
    advect_kernel<<<blocks, threads>>>(b, d, d0, u, v, dt);
    set_bnd(b, d);
}

__global__ void project_step1(float *div, float *p, const float *u, const float *v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > 0 && j > 0 && i <= N && j <= N) {
        div[IX(i,j)] = -0.5f * (u[IX(i+1,j)] - u[IX(i-1,j)] + v[IX(i,j+1)] - v[IX(i,j-1)]) / N;
        p[IX(i,j)] = 0;
    }
}

__global__ void project_step2(float *u, float *v, const float *p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i > 0 && j > 0 && i <= N && j <= N) {
        u[IX(i,j)] -= 0.5f * N * (p[IX(i+1,j)] - p[IX(i-1,j)]);
        v[IX(i,j)] -= 0.5f * N * (p[IX(i,j+1)] - p[IX(i,j-1)]);
    }
}

void project(float *u, float *v, float *p, float *div) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (N + 15) / 16);
    
    project_step1<<<blocks, threads>>>(div, p, u, v);
    set_bnd(0, div); set_bnd(0, p);
    lin_solve(0, p, div, 1, 4, 20);
    project_step2<<<blocks, threads>>>(u, v, p);
    set_bnd(1, u); set_bnd(2, v);
}

void dens_step(float *x, float *x0, float *u, float *v, float diff, float dt) {
    int size = (N + 2) * (N + 2);
    add_source_kernel<<<(size+255)/256, 256>>>(x, x0, dt, size);
    SWAP(x0, x); diffuse(0, x, x0, diff, dt, 20);
    SWAP(x0, x); advect(0, x, x0, u, v, dt);
}

void vel_step(float *u, float *v, float *u_prev, float *v_prev, float visc, float dt) {
    int size = (N + 2) * (N + 2);
    add_source_kernel<<<(size+255)/256, 256>>>(u, u_prev, dt, size);
    add_source_kernel<<<(size+255)/256, 256>>>(v, v_prev, dt, size);
    SWAP(u_prev, u); diffuse(1, u, u_prev, visc, dt, 20);
    SWAP(v_prev, v); diffuse(2, v, v_prev, visc, dt, 20);
    project(u, v, u_prev, v_prev);
    SWAP(u_prev, u); SWAP(v_prev, v);
    advect(1, u, u_prev, u_prev, v_prev, dt);
    advect(2, v, v_prev, u_prev, v_prev, dt);
    project(u, v, u_prev, v_prev);
}

// ==================================================================
// KERNEL DE RENDERIZADO Y SHADERS
// ==================================================================
__global__ void density_to_rgba_kernel(cudaSurfaceObject_t surface, const float* density, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float d = density[y * width + x] * 0.005f;
    d = fminf(fmaxf(d, 0.0f), 1.0f);
    float4 color = make_float4(d, d, d, 1.0f);
    surf2Dwrite(color, surface, x * sizeof(float4), y);
}

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
    uniform sampler2D fluidTexture;
    void main() {
        FragColor = texture(fluidTexture, TexCoord);
    }
)glsl";

// ==================================================================
// GLOBALES Y MAIN
// ==================================================================
// Buffers de simulación
float *d_dens, *d_dens_prev, *d_u, *d_v, *d_u_prev, *d_v_prev;
// Objetos de OpenGL y CUDA
GLuint VAO, shaderProgram, texture_gl;
struct cudaGraphicsResource *cuda_gl_resource;
// Mouse
double mx = 0, my = 0; bool mouseDown = false;

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) mouseDown = (action == GLFW_PRESS);
}
void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    mx = xpos;
    my = ypos;
}

// ==================================================================
// FUNCIÓN MAIN FINAL
// ==================================================================
int main() {
    // 1. Inicialización de GLFW y GLAD
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(WIN_SIZE, WIN_SIZE, "Simulador de Fluidos 2D - CUDA", NULL, NULL);
    glfwMakeContextCurrent(window);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSwapInterval(1); 

    // 2. Compilar Shaders
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vertex_shader_source, NULL); glCompileShader(vs);
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fragment_shader_source, NULL); glCompileShader(fs);
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vs); glAttachShader(shaderProgram, fs);
    glLinkProgram(shaderProgram); glDeleteShader(vs); glDeleteShader(fs);

    // 3. Crear el Rectángulo
    float vertices[] = { 1.0f,1.0f,1.0f,1.0f,  1.0f,-1.0f,1.0f,0.0f, -1.0f,-1.0f,0.0f,0.0f, -1.0f,1.0f,0.0f,1.0f };
    unsigned int indices[] = { 0,1,3, 1,2,3 };
    GLuint VBO, EBO;
    glGenVertexArrays(1, &VAO); glGenBuffers(1, &VBO); glGenBuffers(1, &EBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO); glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO); glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0); glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2*sizeof(float))); glEnableVertexAttribArray(1);

    // 4. Crear Textura y registrarla
    glGenTextures(1, &texture_gl);
    glBindTexture(GL_TEXTURE_2D, texture_gl);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, N+2, N+2, 0, GL_RGBA, GL_FLOAT, NULL);
    CHECK_CUDA(cudaGraphicsGLRegisterImage(&cuda_gl_resource, texture_gl, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));

    // 5. Alojar memoria en la GPU para la simulación
    size_t bytes = (N+2)*(N+2) * sizeof(float);
    CHECK_CUDA(cudaMalloc(&d_dens, bytes)); CHECK_CUDA(cudaMalloc(&d_dens_prev, bytes));
    CHECK_CUDA(cudaMalloc(&d_u, bytes));    CHECK_CUDA(cudaMalloc(&d_u_prev, bytes));
    CHECK_CUDA(cudaMalloc(&d_v, bytes));    CHECK_CUDA(cudaMalloc(&d_v_prev, bytes));
    CHECK_CUDA(cudaMemset(d_dens, 0, bytes)); CHECK_CUDA(cudaMemset(d_dens_prev, 0, bytes));
    CHECK_CUDA(cudaMemset(d_u, 0, bytes));    CHECK_CUDA(cudaMemset(d_u_prev, 0, bytes));
    CHECK_CUDA(cudaMemset(d_v, 0, bytes));    CHECK_CUDA(cudaMemset(d_v_prev, 0, bytes));

    // 6. Establecer condiciones iniciales (CON VELOCIDAD)
    float* h_temp = new float[(N+2)*(N+2)];
    // Densidad inicial
    memset(h_temp, 0, bytes);
    for (int i = N/2 - 8; i < N/2 + 8; i++)
        for (int j = N/2 - 8; j < N/2 + 8; j++)
            h_temp[IX(i, j)] = 200.0f;
    CHECK_CUDA(cudaMemcpy(d_dens, h_temp, bytes, cudaMemcpyHostToDevice));
    
    // Vórtice de velocidad inicial
    memset(h_temp, 0, bytes);
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            float dx = i - (N/2.0f);
            float dy = j - (N/2.0f);
            float r = sqrt(dx*dx + dy*dy);
            if (r > 0 && r < 20.0f) {
                h_temp[IX(i,j)] = -dy * 10.0f; // Vórtice en 'u'
            }
        }
    }
    CHECK_CUDA(cudaMemcpy(d_u, h_temp, bytes, cudaMemcpyHostToDevice));
    
    memset(h_temp, 0, bytes);
    for (int i = 1; i <= N; i++) {
        for (int j = 1; j <= N; j++) {
            float dx = i - (N/2.0f);
            float dy = j - (N/2.0f);
            float r = sqrt(dx*dx + dy*dy);
             if (r > 0 && r < 20.0f) {
                h_temp[IX(i,j)] = dx * 10.0f; // Vórtice en 'v'
            }
        }
    }
    CHECK_CUDA(cudaMemcpy(d_v, h_temp, bytes, cudaMemcpyHostToDevice));
    delete[] h_temp;

    // 7. Bucle principal
    double last_mx = 0, last_my = 0;
    while (!glfwWindowShouldClose(window)) {
        // --- Interacción con el Mouse ---
        CHECK_CUDA(cudaMemset(d_dens_prev, 0, bytes));
        CHECK_CUDA(cudaMemset(d_u_prev, 0, bytes));
        CHECK_CUDA(cudaMemset(d_v_prev, 0, bytes));
        if (mouseDown) {
            float* h_inject_dens = new float[bytes / sizeof(float)];
            float* h_inject_u = new float[bytes / sizeof(float)];
            float* h_inject_v = new float[bytes / sizeof(float)];
            memset(h_inject_dens, 0, bytes);
            memset(h_inject_u, 0, bytes);
            memset(h_inject_v, 0, bytes);
            
            int ix = (mx / WIN_SIZE) * N;
            int iy = (1.0 - my / WIN_SIZE) * N;

            if (ix > 1 && ix < N && iy > 1 && iy < N) {
                h_inject_dens[IX(ix, iy)] = 300.0f; // Añadir densidad
                h_inject_u[IX(ix, iy)] = (mx - last_mx) * 20.0f; // Añadir velocidad en X
                h_inject_v[IX(ix, iy)] = (last_my - my) * 20.0f; // Añadir velocidad en Y
                
                CHECK_CUDA(cudaMemcpy(d_dens_prev, h_inject_dens, bytes, cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_u_prev, h_inject_u, bytes, cudaMemcpyHostToDevice));
                CHECK_CUDA(cudaMemcpy(d_v_prev, h_inject_v, bytes, cudaMemcpyHostToDevice));
            }
            delete[] h_inject_dens;
            delete[] h_inject_u;
            delete[] h_inject_v;
        }
        last_mx = mx;
        last_my = my;

        // --- Lógica de Simulación ---
        vel_step(d_u, d_v, d_u_prev, d_v_prev, VISC, DT);
        dens_step(d_dens, d_dens_prev, d_u, d_v, DIFF, DT);

        // --- Renderizado ---
        CHECK_CUDA(cudaGraphicsMapResources(1, &cuda_gl_resource, 0));
        cudaArray_t cuda_array;
        CHECK_CUDA(cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_gl_resource, 0, 0));
        cudaResourceDesc res_desc;
        memset(&res_desc, 0, sizeof(res_desc));
        res_desc.resType = cudaResourceTypeArray;
        res_desc.res.array.array = cuda_array;
        cudaSurfaceObject_t surface_obj = 0;
        CHECK_CUDA(cudaCreateSurfaceObject(&surface_obj, &res_desc));

        dim3 threads(16, 16);
        dim3 blocks((N+2+15)/16, (N+2+15)/16);
        density_to_rgba_kernel<<<blocks, threads>>>(surface_obj, d_dens, N+2, N+2);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaDestroySurfaceObject(surface_obj));
        CHECK_CUDA(cudaGraphicsUnmapResources(1, &cuda_gl_resource, 0));

        // Dibujar el resultado
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(shaderProgram);
        glBindTexture(GL_TEXTURE_2D, texture_gl);
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Limpieza
    CHECK_CUDA(cudaFree(d_dens)); CHECK_CUDA(cudaFree(d_dens_prev));
    CHECK_CUDA(cudaFree(d_u)); CHECK_CUDA(cudaFree(d_u_prev));
    CHECK_CUDA(cudaFree(d_v)); CHECK_CUDA(cudaFree(d_v_prev));
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteProgram(shaderProgram);
    glDeleteTextures(1, &texture_gl);
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}