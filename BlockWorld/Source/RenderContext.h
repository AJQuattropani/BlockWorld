#pragma once
#include "Shader.h"
#include "Texture.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace bwrenderer
{
	struct RenderContext
	{
		unsigned int screen_width_px, screen_height_px;
		glm::mat4 viewMatrix = glm::mat4(1.0f);
		glm::mat4 projectionMatrix = glm::mat4(1.0f);
        bwrenderer::TextureCache texture_cache;
        uint32_t ch_render_load_distance;

		uint32_t ch_shadow_window_distance;

		double player_position_x = 0.0f;
		double player_position_y = 0.0f;
		double player_position_z = 0.0f;
	};



}