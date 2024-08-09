#pragma once
#include "Shader.hpp"
#include "Texture.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace bwgame
{
	struct Context
	{
		unsigned int screen_width_px{ 0 }, screen_height_px{ 0 };
		glm::mat4 view_matrix = glm::mat4(1.0f);
		glm::mat4 projection_matrix = glm::mat4(1.0f);
		bwrenderer::TextureCache texture_cache;
		uint32_t ch_shadow_window_distance{ 8 };
		//todo change user context to const once movement has been abstracted to a player
		uint32_t ch_render_load_distance{ 8 };
        struct GameTime {
            double last_time_seconds = 0.0, delta_time_seconds = 0.0;
            double last_frame_time_seconds = 0.0, last_update_time_seconds = 0.0;
            const double frame_rate_seconds, game_update_rate_seconds;
        } Timer;
        glm::vec3 player_position = glm::vec3(0);

	};
}

