#pragma once

#include <glad/glad.h>

class IndexBuffer {
private:
    GLuint ib_ID;
    int count;
public:
    IndexBuffer(const unsigned int* data, int count);
    ~IndexBuffer();
    void bind();
    int getCount();
    static void unbind();
};