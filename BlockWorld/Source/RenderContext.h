#pragma once
#include "Shader.h"
#include "Texture.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

struct RenderContext
{
	unsigned int screen_width_px, screen_height_px;
	Shader shader;
	glm::mat4 viewMatrix;
	glm::mat4 projectionMatrix;
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