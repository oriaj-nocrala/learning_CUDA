#pragma once

#include <GLFW/glfw3.h>
#include "camera.h"
#include "glm_helper.cuh"

class InputHandler {
public:
    // Estados
    bool interaction_active = false;
    bool is_rotating = false;

    // Referencias
    Camera& camera;
    vec3& sphere_center;
    vec3& interaction_sphere_center;

    // Constructor
    InputHandler(Camera& cam, vec3& sphere, vec3& interaction_sphere) : camera(cam), sphere_center(sphere), interaction_sphere_center(interaction_sphere) {}

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

    void handle_input(GLFWwindow* window, float dt) {
        glfwPollEvents();

        // Movimiento de la esfera principal con teclado
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) sphere_center.z -= 3.0f * dt;
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) sphere_center.z += 3.0f * dt;
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) sphere_center.x -= 3.0f * dt;
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) sphere_center.x += 3.0f * dt;
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) sphere_center.y += 3.0f * dt;
        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) sphere_center.y -= 3.0f * dt;
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
    }

    // --- Lógica de Ratón ---
    void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
        if (button == GLFW_MOUSE_BUTTON_LEFT) {
            interaction_active = (action == GLFW_PRESS);
        }
        if (button == GLFW_MOUSE_BUTTON_RIGHT) {
            is_rotating = (action == GLFW_PRESS);
            if(is_rotating) camera.firstMouse = true;
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
        
        if (is_rotating) {
            camera.processMouseMovement(xoffset, yoffset);
        }

        if (interaction_active) {
            interaction_sphere_center.x += xoffset * 0.02f;
            interaction_sphere_center.y -= yoffset * 0.02f;
        }

        camera.lastX = xpos;
        camera.lastY = ypos;
    }
    
    void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
        camera.processMouseScroll(yoffset);
    }
};