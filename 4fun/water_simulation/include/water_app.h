#pragma once

#include "common.h"
#include "sph_physics.h"
#include "sph_renderer.h"
#include "input_handler.h"

class WaterApp {
public:
    WaterApp();
    ~WaterApp();

    // Initialize the application
    bool initialize();
    void cleanup();

    // Main application loop
    void run();

private:
    // GLFW window
    GLFWwindow* window;
    
    // Application components
    SPHPhysics physics;
    SPHRenderer renderer;
    InputHandler input;

    // Initialize GLFW and create window
    bool init_glfw();
    
    // Handle user interaction
    void handle_input();
    
    // Update simulation
    void update();
    
    // Render frame
    void render();
};