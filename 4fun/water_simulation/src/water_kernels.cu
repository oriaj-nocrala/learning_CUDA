#include "../include/common.h"

// CUDA kernels for water simulation
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

__global__ void density_to_rgba_kernel(cudaSurfaceObject_t surface, const float* density, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float d = density[y * width + x] * 0.05f; // Increased multiplier for better visibility
    d = fminf(fmaxf(d, 0.0f), 1.0f);
    
    // Enhanced water colors - more vibrant and visible
    float r = d * 0.1f + d * d * 0.3f;        // Slight red component for foam
    float g = d * 0.4f + d * d * 0.5f;        // Green component 
    float b = d * 0.8f + d * d * 0.9f;        // Strong blue component
    float a = fminf(d * 3.0f, 1.0f);          // Strong alpha for visibility
    
    float4 color = make_float4(r, g, b, a);
    surf2Dwrite(color, surface, x * sizeof(float4), y);
}

// C wrapper for kernel launch
void launch_density_to_rgba_kernel(cudaSurfaceObject_t surface, const float* density, int width, int height) {
    dim3 threads(16, 16);
    dim3 blocks((width+15)/16, (height+15)/16);
    density_to_rgba_kernel<<<blocks, threads>>>(surface, density, width, height);
    cudaDeviceSynchronize();
}

// Wrapper functions for other kernels
void launch_add_source_kernel(float *x, const float *s, float dt, int size) {
    add_source_kernel<<<(size+255)/256, 256>>>(x, s, dt, size);
}

void launch_set_bnd_kernel(int b, float *x) {
    int size = (N+2)*(N+2);
    set_bnd_kernel<<<(size+255)/256, 256>>>(b, x);
}

void launch_lin_solve_kernel(int b, float *x, const float *x0, float a, float c) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (N + 15) / 16);
    lin_solve_kernel<<<blocks, threads>>>(b, x, x0, a, c);
}

void launch_advect_kernel(int b, float *d, const float *d0, const float *u, const float *v, float dt) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (N + 15) / 16);
    advect_kernel<<<blocks, threads>>>(b, d, d0, u, v, dt);
}

void launch_project_step1(float *div, float *p, const float *u, const float *v) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (N + 15) / 16);
    project_step1<<<blocks, threads>>>(div, p, u, v);
}

void launch_project_step2(float *u, float *v, const float *p) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (N + 15) / 16);
    project_step2<<<blocks, threads>>>(u, v, p);
}