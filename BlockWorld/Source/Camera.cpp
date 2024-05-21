#include "Camera.h"

Camera::Camera(RenderContext* context, const glm::vec3 positon, const glm::vec3 up, float yaw, float pitch)
    : outputContext(context), position(position), front(glm::vec3(0.0f, 0.0f, -1.0f)), up(up), yaw(yaw), pitch(pitch),
    movementSpeed(SPEED), mouseSensitivity(SENSITIVITY), fov(ZOOM), updateFlags(0b00)
{
    updateCameraVectors();
}

Camera::Camera() : Camera(nullptr)
{}

void Camera::move(Camera_Controls direction, float magnitude)
{
    switch (direction)
    {
    case Camera_Controls::FORWARD:
        position += front * magnitude * SPEED;
        break;
    case Camera_Controls::BACKWARD:
        position -= front * magnitude * SPEED;
        break;
    case Camera_Controls::RIGHT:
        position += right * magnitude * SPEED;
        break;
    case Camera_Controls::LEFT:
        position -= right * magnitude * SPEED;
        break;
    case Camera_Controls::UP:
        position += WORLD_UP * magnitude * SPEED;
        break;
    case Camera_Controls::DOWN:
        position -= WORLD_UP * magnitude * SPEED;
        break;
    }
    updateFlags |= UPDATE_VIEW_FLAG;

}

void Camera::turn(float xoffset, float yoffset, bool constrainPitch)
{
    xoffset *= mouseSensitivity;
    yoffset *= mouseSensitivity;

    yaw += xoffset;
    pitch += yoffset;

    if (constrainPitch)
    {
        if (pitch > 89.0f)
            pitch = 89.0f;
        if (pitch < -89.0f)
            pitch = -89.0f;
    }
    updateCameraVectors();
}

void Camera::zoom(float yoffset)
{
    fov -= yoffset;
    if (fov < 1.0f) {
        fov = 1.0f;
        return;
    }
    if (fov > 45.0f) {
        fov = 45.0f;
        return;
    }
    updateFlags |= UPDATE_PROJECTION_FLAG;
}

void Camera::updateContext()
{
    if (updateFlags & UPDATE_PROJECTION_FLAG) outputContext->projectionMatrix =
        glm::perspective(glm::radians(fov), static_cast<float>(outputContext->screen_width_px) / outputContext->screen_height_px, 0.1f, 100.0f);
    if (updateFlags & UPDATE_VIEW_FLAG) outputContext->viewMatrix = glm::lookAt(position, position + front, up);

    updateFlags = DEFAULT;
}

void Camera::updateCameraVectors()
{
    // calculate the new Front vector
    glm::vec3 frnt;
    frnt.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
    frnt.y = sin(glm::radians(pitch));
    frnt.z = sin(glm::radians(yaw)) * cos(glm::radians(pitch));
    front = glm::normalize(frnt);

    right = glm::normalize(glm::cross(front, WORLD_UP));

    up = glm::normalize(glm::cross(right, front));

    updateFlags |= UPDATE_VIEW_FLAG;
}