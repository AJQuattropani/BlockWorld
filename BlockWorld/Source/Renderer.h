#pragma once

#include "Shader.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <set>

struct RenderContext
{
	unsigned int screen_width_px, screen_height_px;
	Shader shader;
	glm::mat4 viewMatrix;
	glm::mat4 projectionMatrix;
};

enum class Camera_Controls
{
	FORWARD, BACKWARD, RIGHT, LEFT, UP, DOWN
};

class Camera
{
public:
	Camera(const std::shared_ptr<RenderContext>& context, const glm::vec3& positon = glm::vec3(0.0f, 0.0f, -3.5f), 
		const glm::vec3& up = glm::vec3(0.0f, 1.0f, 0.0f), float yaw = YAW, float pitch = PITCH);

	Camera() = default;

	inline float getFov() const { return fov; }

	// TODO move this to a player class
	void move(Camera_Controls direction, float magnitude);
	void turn(float xoffset, float yoffset, bool constrainPitch = true);
	void zoom(float yoffset);

	void updateContext();
private:

	std::shared_ptr<RenderContext> outputContext;

	glm::vec3 position;
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
	constexpr static float SPEED = 1.0/60.f;
	constexpr static float SENSITIVITY = 0.1f;
	constexpr static float ZOOM = 45.0f;
	constexpr static glm::vec3 WORLD_UP = { 0.0f, 1.0f, 0.0f };
};


class RendererObject
{
	virtual void renderPre() = 0;
	virtual void render() = 0;
	virtual void renderPost() = 0;
};

class Mesh : RendererObject
{

};

class Layer {
	Shader shader;

};