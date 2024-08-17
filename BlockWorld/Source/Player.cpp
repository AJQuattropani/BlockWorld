#include "Player.hpp"

namespace bwgame
{
    Player::Player(const std::shared_ptr<bwgame::World>& world, const std::shared_ptr<bwgame::Context>& render_context, glm::vec3 position,
        float yaw, float pitch, float fov, float mouse_sensitivity)
        : world(world), render_context(render_context), position(position), yaw(yaw), pitch(pitch), fov(fov), mouse_sensitivity(mouse_sensitivity)
    {
        updateVectors();
    }

    void Player::update()
    {
        if (sprinting)
        {
            movement_speed = SPRINT_SPEED;
        }
        else {
            movement_speed = SPEED;
        }

        if (displacement != glm::zero<glm::vec3>())
        {
            position += glm::normalize(displacement) * movement_speed;
            displacement = glm::vec3(0.0f);
        }


        if (clickCoolDown != 0) --clickCoolDown;

        updateRenderContext();
    }

    void Player::move(bwgame::Controls direction)
    {
        switch (direction)
        {
        case Controls::FORWARD:
            displacement += front;
            break;
        case Controls::BACKWARD:
            displacement -= front;
            break;
        case Controls::RIGHT:
            displacement += right;
            break;
        case Controls::LEFT:
            displacement -= right;
            break;
        case Controls::UP:
            displacement += WORLD_UP;
            break;
        case Controls::DOWN:
            displacement -= WORLD_UP;
            break;
        }
        update_view = true;
    }

    void Player::zoom(float yoffset) {
        fov -= yoffset;
        if (fov < 1.0f) {
            fov = 1.0f;
            return;
        }
        if (fov > 45.0f) {
            fov = 45.0f;
            return;
        }
        update_projection = true;
    }

    void Player::turn(float xoffset, float yoffset, bool constrainPitch)
    {
        xoffset *= mouse_sensitivity;
        yoffset *= mouse_sensitivity;

        yaw += xoffset;
        pitch += yoffset;

        if (constrainPitch)
        {
            if (pitch > 89.9f)
                pitch = 89.9f;
            if (pitch < -89.9f)
                pitch = -89.9f;
        }
        updateVectors();
    }

    void Player::placeBlock()
    {
        if (clickCoolDown != 0) return;
        if (position.y <= 0.f || position.y >= 256.f) return;
        clickCoolDown = CLICK_COOLDOWN_SECONDS * 60;

        glm::vec3 start_pos = position + glm::vec3(0.5f, 0.5f, 0.5f);
        glm::vec3 step_unit_size(glm::length(front/front.x), glm::length(front / front.y), glm::length(front / front.z));

        WorldBlockCoords block_pos{};
        block_pos.x = start_pos.x;
        block_pos.y = start_pos.y;
        block_pos.z = start_pos.z;
        WorldBlockCoords last_block_pos{ block_pos };
        glm::vec3 ray_length_1D(0.0f);

        glm::i64vec3 step;

        if (front.x < 0)
        {
            step.x = -1;
            ray_length_1D.x = (start_pos.x - float(block_pos.x)) * step_unit_size.x;
        }
        else
        {
            step.x = 1;
            ray_length_1D.x = (float(block_pos.x + 1) - start_pos.x) * step_unit_size.x;
        }
        if (front.y < 0)
        {
            step.y = -1;
            ray_length_1D.y = (start_pos.y - float(block_pos.y)) * step_unit_size.y;
        }
        else
        {
            step.y = 1;
            ray_length_1D.y = (float(block_pos.y + 1) - start_pos.y) * step_unit_size.y;
        }
        if (front.z < 0)
        {
            step.z = -1;
            ray_length_1D.z = (start_pos.z - float(block_pos.z)) * step_unit_size.z;
        }
        else
        {
            step.z = 1;
            ray_length_1D.z = (float(block_pos.z + 1) - start_pos.z) * step_unit_size.z;
        }

        float distance = 0.0f;
        while (distance < reach)
        {
            if (ray_length_1D.x == ray_length_1D.y || ray_length_1D.x == ray_length_1D.z || ray_length_1D.y == ray_length_1D.z) break;

            if (ray_length_1D.x < ray_length_1D.y && ray_length_1D.x < ray_length_1D.z)
            {
                block_pos.x += step.x;
                distance = ray_length_1D.x;
                ray_length_1D.x += step_unit_size.x;
            }
            else if (ray_length_1D.y < ray_length_1D.z)
            {
                block_pos.y += step.y;
                distance = ray_length_1D.y;
                ray_length_1D.y += step_unit_size.y;
            }
            else
            {
                block_pos.z += step.z;
                distance = ray_length_1D.z;
                ray_length_1D.z += step_unit_size.z;
            }

            if (world->checkBlock(block_pos))
            {
                world->setBlock(world->getBlockRegister()->cobblestone, last_block_pos);
                break;
            }

            last_block_pos = block_pos;
        }

    }

    void Player::breakBlock()
    {
        if (clickCoolDown != 0) return;
        if (position.y <= 0.f || position.y >= 256.f) return;
        clickCoolDown = CLICK_COOLDOWN_SECONDS * 60;

        glm::vec3 start_pos = position + glm::vec3(0.5f, 0.5f, 0.5f);
        glm::vec3 step_unit_size(glm::length(front / front.x), glm::length(front / front.y), glm::length(front / front.z));

        WorldBlockCoords block_pos{};
        block_pos.x = start_pos.x;
        block_pos.y = start_pos.y;
        block_pos.z = start_pos.z;

        glm::vec3 ray_length_1D(0.0f);

        glm::i64vec3 step;

        if (front.x < 0)
        {
            step.x = -1;
            ray_length_1D.x = (start_pos.x - float(block_pos.x)) * step_unit_size.x;
        }
        else
        {
            step.x = 1;
            ray_length_1D.x = (float(block_pos.x + 1) - start_pos.x) * step_unit_size.x;
        }
        if (front.y < 0)
        {
            step.y = -1;
            ray_length_1D.y = (start_pos.y - float(block_pos.y)) * step_unit_size.y;
        }
        else
        {
            step.y = 1;
            ray_length_1D.y = (float(block_pos.y + 1) - start_pos.y) * step_unit_size.y;
        }
        if (front.z < 0)
        {
            step.z = -1;
            ray_length_1D.z = (start_pos.z - float(block_pos.z)) * step_unit_size.z;
        }
        else
        {
            step.z = 1;
            ray_length_1D.z = (float(block_pos.z + 1) - start_pos.z) * step_unit_size.z;
        }

        float distance = 0.0f;
        while (distance < reach)
        {
            if (ray_length_1D.x == ray_length_1D.y || ray_length_1D.x == ray_length_1D.z || ray_length_1D.y == ray_length_1D.z) break;

            if (world->checkBlock(block_pos))
            {
                world->destroyBlock(block_pos);
                break;
            }

            if (ray_length_1D.x < ray_length_1D.y && ray_length_1D.x < ray_length_1D.z)
            {
                block_pos.x += step.x;
                distance = ray_length_1D.x;
                ray_length_1D.x += step_unit_size.x;
            }
            else if (ray_length_1D.y < ray_length_1D.z)
            {
                block_pos.y += step.y;
                distance = ray_length_1D.y;
                ray_length_1D.y += step_unit_size.y;
            }
            else
            {
                block_pos.z += step.z;
                distance = ray_length_1D.z;
                ray_length_1D.z += step_unit_size.z;
            }
        }
    }

    void Player::updateVectors()
    {
        // calculate the new Front vector
        glm::vec3 frnt(glm::cos(glm::radians(yaw)) * glm::cos(glm::radians(pitch)),
            glm::sin(glm::radians(pitch)),
            glm::sin(glm::radians(yaw)) * glm::cos(glm::radians(pitch)));
        front = glm::normalize(frnt);
        right = glm::normalize(glm::cross(front, WORLD_UP));
        up = glm::normalize(glm::cross(right, front));

        right_screen = glm::normalize(glm::cross(up, front));

        update_view = true;
    }

    void Player::updateRenderContext()
    {
        if (update_projection) render_context->projection_matrix =
            glm::perspective(glm::radians(fov),
                static_cast<float>(render_context->screen_width_px) / render_context->screen_height_px, 0.1f,
                (render_context->ch_render_load_distance + 3.0f) * 15.0f);
        if (update_view) render_context->view_matrix = glm::lookAt(position, position + front, up);

        render_context->player_position = position;

        update_projection = false;
        update_view = false;
    }



}