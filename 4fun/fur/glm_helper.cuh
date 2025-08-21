#pragma once

#include <cmath>
#include <cuda_runtime.h>

// --- Estructuras de Vectores y Matrices ---
struct vec3 {
    float x, y, z;
    __host__ __device__ vec3(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}
};

struct vec4 {
    float x, y, z, w;
    vec4(float x = 0, float y = 0, float z = 0, float w = 0) : x(x), y(y), z(z), w(w) {}
};

struct mat4 {
    float m[16] = {0};
    mat4() {
        m[0] = 1.0f; m[5] = 1.0f; m[10] = 1.0f; m[15] = 1.0f;
    }
};

// --- Funciones de Vectores ---
__host__ __device__ inline float dot(const vec3& a, const vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float length(const vec3& v) {
    return sqrtf(dot(v, v));
}

__host__ __device__ inline vec3 cross(const vec3& a, const vec3& b) {
    return vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__host__ __device__ inline vec3 normalize(const vec3& v) {
    float len = length(v);
    if (len > 0) {
        return vec3(v.x / len, v.y / len, v.z / len);
    }
    return v;
}

__host__ __device__ inline vec3 operator-(const vec3& a, const vec3& b) {
    return vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline vec3 operator+(const vec3& a, const vec3& b) {
    return vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline vec3 operator*(const vec3& v, float s) {
    return vec3(v.x * s, v.y * s, v.z * s);
}

__host__ __device__ inline vec3 operator*(float s, const vec3& v) {
    return vec3(v.x * s, v.y * s, v.z * s);
}

__host__ __device__ inline vec3 operator/(const vec3& v, float s) {
    return vec3(v.x / s, v.y / s, v.z / s);
}

__host__ __device__ inline vec3& operator+=(vec3& a, const vec3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

__host__ __device__ inline vec3& operator-=(vec3& a, const vec3& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}


// --- Funciones de Matrices ---
// Multiplicación para matrices Column-Major
inline mat4 operator*(const mat4& a, const mat4& b) {
    mat4 r;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) {
                sum += a.m[k * 4 + i] * b.m[j * 4 + k];
            }
            r.m[j * 4 + i] = sum;
        }
    }
    return r;
}

// Genera una matriz de perspectiva (Column-Major)
inline mat4 perspective(float fovy, float aspect, float zNear, float zFar) {
    mat4 r;
    float tanHalfFovy = tan(fovy / 2.0f);
    r.m[0] = 1.0f / (aspect * tanHalfFovy);
    r.m[5] = 1.0f / (tanHalfFovy);
    r.m[10] = -(zFar + zNear) / (zFar - zNear);
    r.m[11] = -1.0f;
    r.m[14] = -(2.0f * zFar * zNear) / (zFar - zNear);
    r.m[15] = 0.0f;
    return r;
}

// Genera una matriz de vista (Column-Major)
inline mat4 lookAt(const vec3& eye, const vec3& center, const vec3& up) {
    vec3 f = normalize(vec3(center.x - eye.x, center.y - eye.y, center.z - eye.z));
    vec3 s = normalize(cross(f, up));
    vec3 u = cross(s, f);
    mat4 r;
    r.m[0] = s.x;   r.m[4] = s.y;   r.m[8] = s.z;
    r.m[1] = u.x;   r.m[5] = u.y;   r.m[9] = u.z;
    r.m[2] = -f.x;  r.m[6] = -f.y;  r.m[10] = -f.z;
    r.m[12] = -dot(s, eye);
    r.m[13] = -dot(u, eye);
    r.m[14] = dot(f, eye);
    return r;
}

// Calcula la inversa de una matriz (asume Column-Major)
inline mat4 inverse(const mat4& mat) {
    mat4 inv;
    float det;

    inv.m[0] =  mat.m[5] * mat.m[10] * mat.m[15] - mat.m[5] * mat.m[11] * mat.m[14] - mat.m[9] * mat.m[6] * mat.m[15] + mat.m[9] * mat.m[7] * mat.m[14] + mat.m[13] * mat.m[6] * mat.m[11] - mat.m[13] * mat.m[7] * mat.m[10];
    inv.m[4] = -mat.m[4] * mat.m[10] * mat.m[15] + mat.m[4] * mat.m[11] * mat.m[14] + mat.m[8] * mat.m[6] * mat.m[15] - mat.m[8] * mat.m[7] * mat.m[14] - mat.m[12] * mat.m[6] * mat.m[11] + mat.m[12] * mat.m[7] * mat.m[10];
    inv.m[8] =  mat.m[4] * mat.m[9]  * mat.m[15] - mat.m[4] * mat.m[11] * mat.m[13] - mat.m[8] * mat.m[5] * mat.m[15] + mat.m[8] * mat.m[7] * mat.m[13] + mat.m[12] * mat.m[5] * mat.m[11] - mat.m[12] * mat.m[7] * mat.m[9];
    inv.m[12]= -mat.m[4] * mat.m[9]  * mat.m[14] + mat.m[4] * mat.m[10] * mat.m[13] + mat.m[8] * mat.m[5] * mat.m[14] - mat.m[8] * mat.m[6] * mat.m[13] - mat.m[12] * mat.m[5] * mat.m[10] + mat.m[12] * mat.m[6] * mat.m[9];
    inv.m[1] = -mat.m[1] * mat.m[10] * mat.m[15] + mat.m[1] * mat.m[11] * mat.m[14] + mat.m[9] * mat.m[2] * mat.m[15] - mat.m[9] * mat.m[3] * mat.m[14] - mat.m[13] * mat.m[2] * mat.m[11] + mat.m[13] * mat.m[3] * mat.m[10];
    inv.m[5] =  mat.m[0] * mat.m[10] * mat.m[15] - mat.m[0] * mat.m[11] * mat.m[14] - mat.m[8] * mat.m[2] * mat.m[15] + mat.m[8] * mat.m[3] * mat.m[14] + mat.m[12] * mat.m[2] * mat.m[11] - mat.m[12] * mat.m[3] * mat.m[10];
    inv.m[9] = -mat.m[0] * mat.m[9]  * mat.m[15] + mat.m[0] * mat.m[11] * mat.m[13] + mat.m[8] * mat.m[1] * mat.m[15] - mat.m[8] * mat.m[3] * mat.m[13] - mat.m[12] * mat.m[1] * mat.m[11] + mat.m[12] * mat.m[3] * mat.m[9];
    inv.m[13]=  mat.m[0] * mat.m[9]  * mat.m[14] - mat.m[0] * mat.m[10] * mat.m[13] - mat.m[8] * mat.m[1] * mat.m[14] + mat.m[8] * mat.m[2] * mat.m[13] + mat.m[12] * mat.m[1] * mat.m[10] - mat.m[12] * mat.m[2] * mat.m[9];
    inv.m[2] =  mat.m[1] * mat.m[6]  * mat.m[15] - mat.m[1] * mat.m[7]  * mat.m[14] - mat.m[5] * mat.m[2] * mat.m[15] + mat.m[5] * mat.m[3] * mat.m[14] + mat.m[13] * mat.m[2] * mat.m[7]  - mat.m[13] * mat.m[3] * mat.m[6];
    inv.m[6] = -mat.m[0] * mat.m[6]  * mat.m[15] + mat.m[0] * mat.m[7]  * mat.m[14] + mat.m[4] * mat.m[2] * mat.m[15] - mat.m[4] * mat.m[3] * mat.m[14] - mat.m[12] * mat.m[2] * mat.m[7]  + mat.m[12] * mat.m[3] * mat.m[6];
    inv.m[10]=  mat.m[0] * mat.m[5]  * mat.m[15] - mat.m[0] * mat.m[7]  * mat.m[13] - mat.m[4] * mat.m[1] * mat.m[15] + mat.m[4] * mat.m[3] * mat.m[13] + mat.m[12] * mat.m[1] * mat.m[7]  - mat.m[12] * mat.m[3] * mat.m[5];
    inv.m[14]= -mat.m[0] * mat.m[5]  * mat.m[14] + mat.m[0] * mat.m[6]  * mat.m[13] + mat.m[4] * mat.m[1] * mat.m[14] - mat.m[4] * mat.m[2] * mat.m[13] - mat.m[12] * mat.m[1] * mat.m[6]  + mat.m[12] * mat.m[2] * mat.m[5];
    inv.m[3] = -mat.m[1] * mat.m[6]  * mat.m[11] + mat.m[1] * mat.m[7]  * mat.m[10] + mat.m[5] * mat.m[2] * mat.m[11] - mat.m[5] * mat.m[3] * mat.m[10] - mat.m[9]  * mat.m[2] * mat.m[7]  + mat.m[9]  * mat.m[3] * mat.m[6];
    inv.m[7] =  mat.m[0] * mat.m[6]  * mat.m[11] - mat.m[0] * mat.m[7]  * mat.m[10] - mat.m[4] * mat.m[2] * mat.m[11] + mat.m[4] * mat.m[3] * mat.m[10] + mat.m[8]  * mat.m[2] * mat.m[7]  - mat.m[8]  * mat.m[3] * mat.m[6];
    inv.m[11]= -mat.m[0] * mat.m[5]  * mat.m[11] + mat.m[0] * mat.m[7]  * mat.m[9]  + mat.m[4] * mat.m[1] * mat.m[11] - mat.m[4] * mat.m[3] * mat.m[9]  - mat.m[8]  * mat.m[1] * mat.m[7]  + mat.m[8]  * mat.m[3] * mat.m[5];
    inv.m[15]=  mat.m[0] * mat.m[5]  * mat.m[10] - mat.m[0] * mat.m[6]  * mat.m[9]  - mat.m[4] * mat.m[1] * mat.m[10] + mat.m[4] * mat.m[2] * mat.m[9]  + mat.m[8]  * mat.m[1] * mat.m[6]  - mat.m[8]  * mat.m[2] * mat.m[5];

    det = mat.m[0] * inv.m[0] + mat.m[1] * inv.m[4] + mat.m[2] * inv.m[8] + mat.m[3] * inv.m[12];

    if (det == 0)
        return mat4();

    det = 1.0 / det;

    for (int i = 0; i < 16; i++)
        inv.m[i] = inv.m[i] * det;

    return inv;
}

// Desproyecta coordenadas de pantalla a un rayo en el espacio 3D
inline vec3 unproject(vec3 screen_coords, const mat4& view, const mat4& projection, int width, int height) {
    mat4 view_proj = projection * view;
    mat4 view_proj_inv = inverse(view_proj);

    float x = (2.0f * screen_coords.x) / width - 1.0f;
    float y = 1.0f - (2.0f * screen_coords.y) / height;

    vec4 ray_clip = {x, y, -1.0, 1.0};
    vec4 ray_world;

    // Multiplicación Vector-Matriz (Column-Major)
    ray_world.x = view_proj_inv.m[0] * ray_clip.x + view_proj_inv.m[4] * ray_clip.y + view_proj_inv.m[8] * ray_clip.z + view_proj_inv.m[12] * ray_clip.w;
    ray_world.y = view_proj_inv.m[1] * ray_clip.x + view_proj_inv.m[5] * ray_clip.y + view_proj_inv.m[9] * ray_clip.z + view_proj_inv.m[13] * ray_clip.w;
    ray_world.z = view_proj_inv.m[2] * ray_clip.x + view_proj_inv.m[6] * ray_clip.y + view_proj_inv.m[10] * ray_clip.z + view_proj_inv.m[14] * ray_clip.w;
    float w =     view_proj_inv.m[3] * ray_clip.x + view_proj_inv.m[7] * ray_clip.y + view_proj_inv.m[11] * ray_clip.z + view_proj_inv.m[15] * ray_clip.w;

    if (w != 0.0f) {
        ray_world.x /= w;
        ray_world.y /= w;
        ray_world.z /= w;
    }

    return vec3(ray_world.x, ray_world.y, ray_world.z);
}
