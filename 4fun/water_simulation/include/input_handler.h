#pragma once

#include "common.h"

class InputHandler {
public:
    InputHandler();
    
    // Set up callbacks
    void setup_callbacks(GLFWwindow* window);
    
    // Get mouse state
    bool is_mouse_down() const { return mouseDown; }
    void get_mouse_position(double& x, double& y) const { x = mx; y = my; }
    void get_last_mouse_position(double& x, double& y) const { x = last_mx; y = last_my; }
    
    // Update mouse state
    void update();

private:
    static double mx, my;
    static double last_mx, last_my;
    static bool mouseDown;
    
    // Static callback functions
    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
    static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
};