#pragma once

#include "World.hpp"

#include "glm/gtc/matrix_transform.hpp"

namespace bwgame
{

    enum class Controls
    {
        FORWARD, BACKWARD, RIGHT, LEFT, UP, DOWN
    };

    class Player {
    public:

        Player(const std::shared_ptr<bwgame::World>& world, const std::shared_ptr<bwgame::Context>& render_context,
            glm::vec3 position = SPAWN_POINT, float yaw = YAW, float pitch = PITCH, float fov = ZOOM, float mouse_sensitivity = SENSITIVITY);

        inline void setSprinting(bool sprint) { sprinting = sprint; }
        inline const glm::vec3& getPosition() const { return position; }

        void update();

        void move(bwgame::Controls direction);

        void zoom(float yoffset);

        void turn(float xoffset, float yoffset, bool constrainPitch = true);

        void placeBlock();
        void breakBlock();

        
    private:
        std::shared_ptr<bwgame::World> world;
        std::shared_ptr<bwgame::Context> render_context;

        glm::vec3 position;
        glm::vec3 displacement = glm::vec3(0.0f);

        glm::vec3 front;
        glm::vec3 right;
        glm::vec3 up;

        float movement_speed = SPEED;
        float reach = REACH;

        float yaw, pitch;
        float mouse_sensitivity, fov;

        bool sprinting = false;

        bool update_view = false;
        bool update_projection = false;

        uint16_t clickCoolDown = 0;

    private:
        void updateVectors();
        void updateRenderContext();
    private:
        constexpr static float YAW = -90.0f;
        constexpr static float PITCH = 0.0f;
        constexpr static float REACH = 10.0f;
        constexpr static float SENSITIVITY = 0.1f;
        constexpr static float ZOOM = 30.0f;
        constexpr static float CLICK_COOLDOWN_SECONDS = 0.1f;
        constexpr static glm::vec3 SPAWN_POINT = glm::vec3(0.0f, 60.0f, 0.0f);

    public:
        constexpr static glm::vec3 WORLD_UP = glm::vec3(0.0f, 1.0f, 0.0f);
        constexpr static float SPEED = 40.0f / 60.f;
        constexpr static float SPRINT_SPEED = 100.0f / 60.f;



    };






};