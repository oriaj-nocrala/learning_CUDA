#pragma once


#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stdlib.h>
#include <iostream>

#include "src/Shaders/Shader.h"
#include "libs/camera.h"

#define GLT_IMPLEMENTATION
#include "libs/gltext.h"

#include <iostream>

#define BOOST_STACKTRACE_USE_ADDR2LINE
// //#include <boost/stacktrace.hpp>

#include "src/Renderer/Renderer.h"
#include "src/Renderer/VertexBuffer.h"
#include "src/Renderer/IndexBuffer.h"
#include "src/Renderer/VertexArrayObject.h"
#include "src/Renderer/AttributeBufferLayout.h"

#include "src/sphere.h"
#include "src/box.h"

#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/normal.hpp>
//#include <glm/gtx/string_cast.hpp>

#include <vector>

class Visualizer {
public:
    Visualizer(int objectNum, int maxNumTriangles, float radius, float minBoundX, float minBoundY, float minBoundZ, float maxBoundX, float maxBoundY, float maxBoundZ);
    void draw(int objectNum);
    void drawTriangles(int numTriangles);
    void end();

    GLuint vertexArray;
    GLuint triangleArray;
    
    GLFWwindow* window;

    Renderer* renderer;

    Shader* shader;

    static bool runSimulation;
    static bool marchingCubes;
    static bool openDam;

    float radius, minBoundX, minBoundY, minBoundZ, maxBoundX, maxBoundY, maxBoundZ;
    float dam_pos = 0;
    float dam_vel = 0;
    float dam_acc = 0.0005;
    
    Sphere* sphere;
    Box* box;
    Box* box_dam;
    bool ENABLE_FACE_CULLING;
};

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow *window);
