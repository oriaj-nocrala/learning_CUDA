#pragma once

#include <glad/glad.h>

class VertexBuffer {
private:
    GLuint vb_ID;
public:
    VertexBuffer(const void* data, unsigned int size);
    VertexBuffer(GLuint vertexBuffer_ID);
    ~VertexBuffer();
    void bind();
    static void unbind();
};