#include "../include/input_handler.h"

// Static member definitions
double InputHandler::mx = 0;
double InputHandler::my = 0;
double InputHandler::last_mx = 0;
double InputHandler::last_my = 0;
bool InputHandler::mouseDown = false;

InputHandler::InputHandler() {
}

void InputHandler::setup_callbacks(GLFWwindow* window) {
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
}

void InputHandler::update() {
    last_mx = mx;
    last_my = my;
}

void InputHandler::mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        mouseDown = (action == GLFW_PRESS);
    }
}

void InputHandler::cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    mx = xpos;
    my = ypos;
}