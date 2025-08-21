#pragma once

#include <GLFW/glfw3.h>
#include "glm_helper.h"

class Camera {
public:
    vec3 position;
    vec3 front;
    vec3 up;
    float yaw;
    float pitch;
    float lastX, lastY;
    bool firstMouse;

    Camera(vec3 pos = vec3(0.0f, 0.0f, 3.0f)) : position(pos), front(vec3(0.0f, 0.0f, -1.0f)), up(vec3(0.0f, 1.0f, 0.0f)), yaw(-90.0f), pitch(0.0f), lastX(400), lastY(300), firstMouse(true) {}

    mat4 getViewMatrix() {
        return lookAt(position, vec3(position.x + front.x, position.y + front.y, position.z + front.z), up);
    }

    void processMouseMovement(float xoffset, float yoffset) {
        float sensitivity = 0.1f;
        xoffset *= sensitivity;
        yoffset *= sensitivity;

        yaw += xoffset;
        pitch += yoffset;

        if (pitch > 89.0f) pitch = 89.0f;
        if (pitch < -89.0f) pitch = -89.0f;

        vec3 direction;
        direction.x = cos(yaw * (M_PI / 180.0f)) * cos(pitch * (M_PI / 180.0f));
        direction.y = sin(pitch * (M_PI / 180.0f));
        direction.z = sin(yaw * (M_PI / 180.0f)) * cos(pitch * (M_PI / 180.0f));
        front = normalize(direction);
    }

    void processMouseScroll(float yoffset) {
        float speed = 0.5f;
        vec3 direction = front * yoffset * speed;
        position += direction;
    }
};
