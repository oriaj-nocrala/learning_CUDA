#pragma once

#include <GLFW/glfw3.h>
#include "camera.h"
#include "glm_helper.h"

class InputHandler {
public:
    // Estados
    bool is_dragging_box = false;
    bool is_camera_panning = false;
    bool is_rotating_box = false;
    bool box_drag_modifier_active = false;
    bool box_grab_requested = false;
    bool interaction_requested = false;
    bool is_area_force_active = false;
    vec3 grab_offset;

    // Referencias
    Camera& camera;
    vec3& box_center;

    // Constructor
    InputHandler(Camera& cam, vec3& box) : camera(cam), box_center(box) {}

    // Destructor
    ~InputHandler() {}

    // Registrar callbacks
    void setup_callbacks(GLFWwindow* window) {
        glfwSetWindowUserPointer(window, this);
        glfwSetMouseButtonCallback(window, mouse_button_callback_static);
        glfwSetCursorPosCallback(window, mouse_callback_static);
        glfwSetKeyCallback(window, key_callback_static);
        glfwSetScrollCallback(window, scroll_callback_static);
    }

private:
    // --- Callbacks Estáticos ---
    static void mouse_button_callback_static(GLFWwindow* window, int button, int action, int mods) {
        InputHandler* handler = static_cast<InputHandler*>(glfwGetWindowUserPointer(window));
        if (handler) handler->mouse_button_callback(window, button, action, mods);
    }

    static void mouse_callback_static(GLFWwindow* window, double xpos, double ypos) {
        InputHandler* handler = static_cast<InputHandler*>(glfwGetWindowUserPointer(window));
        if (handler) handler->mouse_callback(window, xpos, ypos);
    }

    static void key_callback_static(GLFWwindow* window, int key, int scancode, int action, int mods) {
        InputHandler* handler = static_cast<InputHandler*>(glfwGetWindowUserPointer(window));
        if (handler) handler->key_callback(window, key, scancode, action, mods);
    }

    static void scroll_callback_static(GLFWwindow* window, double xoffset, double yoffset) {
        InputHandler* handler = static_cast<InputHandler*>(glfwGetWindowUserPointer(window));
        if (handler) handler->scroll_callback(window, xoffset, yoffset);
    }

    // --- Lógica de Teclado ---
    void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
        if (key == GLFW_KEY_Q || key == GLFW_KEY_ESCAPE) {
            if (action == GLFW_PRESS) glfwSetWindowShouldClose(window, true);
        }
        if (key == GLFW_KEY_F || key == GLFW_KEY_F11) {
            if (action == GLFW_PRESS) toggle_fullscreen(window);
        }
        if (key == GLFW_KEY_LEFT_ALT) {
            box_drag_modifier_active = (action == GLFW_PRESS || action == GLFW_REPEAT);
            if(action == GLFW_PRESS) camera.firstMouse = true; // Evitar saltos
        }
    }

    // --- Lógica de Ratón ---
    void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            if (action == GLFW_PRESS) {
                if (box_drag_modifier_active) {
                    box_grab_requested = true;
                } else {
                    interaction_requested = true;
                    is_area_force_active = true; // Activar modo de fuerza en área
                }
            } else if (action == GLFW_RELEASE) {
                is_dragging_box = false;
                is_area_force_active = false; // Desactivar modo de fuerza en área
            }
        }
        if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
            is_camera_panning = (action == GLFW_PRESS);
            if(is_camera_panning) camera.firstMouse = true;
        }
    }

    void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
        if (camera.firstMouse) {
            camera.lastX = xpos;
            camera.lastY = ypos;
            camera.firstMouse = false;
        }

        float xoffset = xpos - camera.lastX;
        float yoffset = camera.lastY - ypos;
        
        if (is_camera_panning) {
            camera.processMouseMovement(xoffset, yoffset);
        }
        
        // Actualizar siempre para la rotación
        camera.lastX = xpos;
        camera.lastY = ypos;
    }
    
    void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
        camera.processMouseScroll(yoffset);
    }

    void toggle_fullscreen(GLFWwindow* window) {
        static bool is_fullscreen = false;
        static int last_win_x, last_win_y, last_win_width, last_win_height;
        if (is_fullscreen) {
            glfwSetWindowMonitor(window, nullptr, last_win_x, last_win_y, last_win_width, last_win_height, 0);
            is_fullscreen = false;
        } else {
            glfwGetWindowPos(window, &last_win_x, &last_win_y);
            glfwGetWindowSize(window, &last_win_width, &last_win_height);
            GLFWmonitor* monitor = glfwGetPrimaryMonitor();
            const GLFWvidmode* mode = glfwGetVideoMode(monitor);
            glfwSetWindowMonitor(window, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
            is_fullscreen = true;
        }
    }
};