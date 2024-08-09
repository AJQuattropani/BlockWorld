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
            if (pitch > 89.0f)
                pitch = 89.0f;
            if (pitch < -89.0f)
                pitch = -89.0f;
        }
        updateVectors();
    }

    void Player::placeBlock()
    {
        if (clickCoolDown != 0) return;

        clickCoolDown = CLICK_COOLDOWN_SECONDS * 60;

        if (position.y <= 0.f || position.y >= 256.f) return;

        glm::vec3 temp = position + front * reach;

        glm::vec3 initial;
        glm::vec3 last;
        if (temp.x > position.x)
        {
            last.x = glm::floor(temp.x);
            initial.x = glm::floor(position.x);
        }
        else
        {
            last.x = glm::ceil(temp.x);
            initial.x = glm::ceil(position.x);
        }

        if (temp.y > position.y)
        {
            last.y = glm::floor(temp.y);
            initial.y = glm::floor(position.y);
        }
        else
        {
            last.y = glm::ceil(temp.y);
            initial.y = glm::ceil(position.y);
        }

        if (temp.z > position.z)
        {
            last.z = glm::floor(temp.z);
            initial.z = glm::floor(position.z);
        }
        else
        {
            last.z = glm::ceil(temp.z);
            initial.z = glm::ceil(position.z);
        }

        glm::vec3 abs_displacement = glm::abs(last - initial);

        int stepX = initial.x < last.x ? 1 : -1;
        int stepY = initial.y < last.y ? 1 : -1;
        int stepZ = initial.z < last.z ? 1 : -1;

        float hypotenuse = glm::length(abs_displacement);

        glm::vec3 tDelta = glm::vec3(1.0f / abs_displacement.x, 1.0f / abs_displacement.y, 1.0f / abs_displacement.z) * hypotenuse;
        glm::vec3 tMax = 0.5f * tDelta;

        WorldBlockCoords coords, prev;
        coords.x = initial.x;
        coords.y = initial.y;
        coords.z = initial.z;
        prev = coords;

        while (initial.x != last.x || initial.y != last.y || initial.z != last.z) {
            if (tMax.x < tMax.y) {
                if (tMax.x < tMax.z) {
                    initial.x += stepX;
                    tMax.x += tDelta.x;
                }
                else if (tMax.x > tMax.z) {
                    initial.z += stepZ;
                    tMax.z += tDelta.z;
                }
                else {
                    initial.x += stepX;
                    tMax.x += tDelta.x;
                    initial.z += stepZ;
                    tMax.z += tDelta.z;
                    continue;
                }
            }
            else if (tMax.x > tMax.y) {
                if (tMax.y < tMax.z) {
                    initial.y += stepY;
                    tMax.y += tDelta.y;
                }
                else if (tMax.y > tMax.z) {
                    initial.z += stepZ;
                    tMax.z += tDelta.z;
                }
                else {
                    initial.y += stepY;
                    tMax.y += tDelta.y;
                    initial.z += stepZ;
                    tMax.z += tDelta.z;
                    continue;
                }
            }
            else {
                if (tMax.y < tMax.z) {
                    initial.x += stepX;
                    tMax.x += tDelta.x;
                    initial.y += stepY;
                    tMax.y += tDelta.y;
                    continue;
                }
                else if (tMax.y > tMax.z) {
                    initial.z += stepZ;
                    tMax.z += tDelta.z;
                }
                else {
                    initial.x += stepX;
                    tMax.x += tDelta.x;
                    initial.y += stepY;
                    tMax.y += tDelta.y;
                    initial.z += stepZ;
                    tMax.z += tDelta.z;
                    continue;
                }
            }

            coords.x = initial.x;
            coords.y = initial.y;
            coords.z = initial.z;

            if (world->checkBlock(coords))
            {
                world->setBlock(world->getBlockRegister()->cobblestone, prev);
                return;
            }

            prev = coords;
        }

    }

    void Player::breakBlock()
    {
        if (clickCoolDown != 0) return;

        clickCoolDown = CLICK_COOLDOWN_SECONDS * 60;

        if (position.y <= 0.f || position.y >= 256.f) return;

        glm::vec3 r0 = position;
        glm::vec3 r1 = position + front * reach;

        // find the displacement vector
        glm::vec3 displacement = r1 - r0;
        BW_DEBUG("%f, %f, %f", displacement.x, displacement.y, displacement.z);

        constexpr float precision_per_block = 3.0f;
        uint32_t iterations = glm::length(displacement) * precision_per_block;
        glm::vec3 increment = displacement / (float)iterations;

        WorldBlockCoords current, last;
        last.x = r0.x;
        last.y = r0.y;
        last.z = r0.z;

        for (size_t i = 0; i < iterations; i++)
        {
            current.x = r0.x;
            current.y = r0.y;
            current.z = r0.z;
            r0 += increment;

            if (current.equals(last)) continue;

            if (world->checkBlock(current)) 
            {
                world->setBlock(world->getBlockRegister()->cobblestone, last);
                break;
            }

            // TODO add checker to ensure no overreach for iterations;

            last = current;
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