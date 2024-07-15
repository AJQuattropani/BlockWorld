#include "Camera.hpp"

namespace bwrenderer {

    Camera::Camera(const std::shared_ptr<bwgame::UserContext>& input_context,
        const std::shared_ptr<bwrenderer::RenderContext>& output_context, const glm::vec3 position, const glm::vec3 up, float yaw, float pitch)
        : user_context(input_context), output_context(output_context), position(position), front(glm::vec3(0.0f, 0.0f, -1.0f)), up(up), yaw(yaw), pitch(pitch),
        mouse_sensitivity(SENSITIVITY), fov(ZOOM), update_flags(0b00), movement_speed(SPEED)
    {
        updateCameraVectors();
    }


    void Camera::move(bwgame::Controls direction, float magnitude)
    {
        switch (direction)
        {
        case bwgame::Controls::FORWARD:
            position += front * magnitude * movement_speed;
            break;
        case bwgame::Controls::BACKWARD:
            position -= front * magnitude * movement_speed;
            break;
        case bwgame::Controls::RIGHT:
            position += right * magnitude * movement_speed;
            break;
        case bwgame::Controls::LEFT:
            position -= right * magnitude * movement_speed;
            break;
        case bwgame::Controls::UP:
            position += WORLD_UP * magnitude * movement_speed;
            break;
        case bwgame::Controls::DOWN:
            position -= WORLD_UP * magnitude * movement_speed;
            break;
        }
        update_flags |= UPDATE_VIEW_FLAG;

    }

    void Camera::turn(float xoffset, float yoffset, bool constrainPitch)
    {
        xoffset *= mouse_sensitivity;
        yoffset *= mouse_sensitivity;

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
        update_flags |= UPDATE_PROJECTION_FLAG;
    }

    void Camera::updateContext()
    {
        if (update_flags & UPDATE_PROJECTION_FLAG) output_context->projection_matrix =
            glm::perspective(glm::radians(fov), static_cast<float>(output_context->screen_width_px) / output_context->screen_height_px, 0.1f, (user_context->ch_render_load_distance + 3.0f) * 15.0f);
        if (update_flags & UPDATE_VIEW_FLAG) output_context->view_matrix = glm::lookAt(position, position + front, up);

        user_context->player_position_x = position.x;
        user_context->player_position_y = position.y;
        user_context->player_position_z = position.z;

        update_flags = DEFAULT;
    }

    void Camera::updateCameraVectors()
    {
        // calculate the new Front vector
        glm::vec3 frnt(glm::cos(glm::radians(yaw)) * glm::cos(glm::radians(pitch)), glm::sin(glm::radians(pitch)), glm::sin(glm::radians(yaw)) * glm::cos(glm::radians(pitch)));
        front = glm::normalize(frnt);

        right = glm::normalize(glm::cross(front, WORLD_UP));

        up = glm::normalize(glm::cross(right, front));

        update_flags |= UPDATE_VIEW_FLAG;
    }

}
