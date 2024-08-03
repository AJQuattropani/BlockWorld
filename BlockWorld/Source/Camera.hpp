#pragma once


#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "Context.hpp"
#include <memory>

namespace bwrenderer
{

	class Camera
	{
	public:
		Camera(const std::shared_ptr<bwgame::UserContext>& input_context, 
			const std::shared_ptr<bwrenderer::RenderContext>& output_context, const glm::vec3 position = glm::vec3(0.0f, 60.0f, 0.0f),
			const glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f), float yaw = YAW, float pitch = PITCH);

		inline float getFov() const { return fov; }

		// TODO move this to a player class
		void turn(float xoffset, float yoffset, bool constrainPitch = true);
		void zoom(float yoffset);

		void updateContext();

		void move(bwgame::Controls direction, float magnitude);
	private:

		glm::vec3 position; // replace back after new data structure created.
		
		std::shared_ptr<bwgame::UserContext> user_context;
		std::shared_ptr<bwrenderer::RenderContext> output_context;

		glm::vec3 front;
		glm::vec3 up;
		glm::vec3 right;

		float yaw, pitch;
		float mouse_sensitivity, fov, movement_speed;
		uint_fast8_t update_flags;
	private:
		void updateCameraVectors();
	private:
		constexpr static uint8_t DEFAULT = 0b00;
		constexpr static uint8_t UPDATE_VIEW_FLAG = 0b01;
		constexpr static uint8_t UPDATE_PROJECTION_FLAG = 0b10;
		constexpr static float YAW = -90.0f;
		constexpr static float PITCH = 0.0f;
		constexpr static float SPEED = 40.0f / 60.f;
		constexpr static float SENSITIVITY = 0.1f;
		constexpr static float ZOOM = 30.0f;
		constexpr static glm::vec3 WORLD_UP = { 0.0f, 1.0f, 0.0f };
	};
}