#pragma once


#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "RenderContext.h"


#include <memory>

namespace bwrenderer
{
	enum class Camera_Controls
	{
		FORWARD, BACKWARD, RIGHT, LEFT, UP, DOWN
	};

	class Camera
	{
	public:
		Camera(bwrenderer::RenderContext* context, const glm::vec3 position = glm::vec3(0.0f, 60.0f, 0.0f),
			const glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f), float yaw = YAW, float pitch = PITCH);

		Camera();

		void attachContext(const std::shared_ptr<bwrenderer::RenderContext>& context)
		{
			outputContext = context;
		}

		inline float getFov() const { return fov; }

		// TODO move this to a player class
		void move(Camera_Controls direction, float magnitude);
		void turn(float xoffset, float yoffset, bool constrainPitch = true);
		void zoom(float yoffset);

		void updateContext();


		glm::vec3 position; // replace back after new data structure created.
	private:

		std::shared_ptr<bwrenderer::RenderContext> outputContext;

		glm::vec3 front;
		glm::vec3 up;
		glm::vec3 right;

		float yaw, pitch;
		float movementSpeed, mouseSensitivity, fov;
		uint_fast8_t updateFlags;
	private:
		void updateCameraVectors();
	private:
		constexpr static uint8_t DEFAULT = 0b00;
		constexpr static uint8_t UPDATE_VIEW_FLAG = 0b01;
		constexpr static uint8_t UPDATE_PROJECTION_FLAG = 0b10;
		constexpr static float YAW = -90.0f;
		constexpr static float PITCH = 0.0f;
		constexpr static float SPEED = 20.0f / 60.f;
		constexpr static float SENSITIVITY = 0.1f;
		constexpr static float ZOOM = 30.0f;
		constexpr static glm::vec3 WORLD_UP = { 0.0f, 1.0f, 0.0f };
	};
}