#pragma once
#include "Shader.hpp"
#include "Texture.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace bwrenderer
{
	struct RenderContext
	{
		unsigned int screen_width_px{ 0 }, screen_height_px{ 0 };
		glm::mat4 view_matrix = glm::mat4(1.0f);
		glm::mat4 projection_matrix = glm::mat4(1.0f);
		TextureCache texture_cache;
		uint32_t ch_shadow_window_distance{ 8 };
	};
}

namespace bwgame
{
	enum class Controls
	{
		FORWARD, BACKWARD, RIGHT, LEFT, UP, DOWN
	};

	struct UserContext
	{
		//todo change user context to const once movement has been abstracted to a player
		double player_position_x = 0.0f;
		double player_position_y = 0.0f;
		double player_position_z = 0.0f;
		uint32_t ch_render_load_distance{ 8 };

		struct GameTime {
			double last_time_seconds = 0.0, delta_time_seconds = 0.0;
			double last_frame_time_seconds = 0.0, last_update_time_seconds = 0.0;
			const double frame_rate_seconds, game_update_rate_seconds;
		} Timer;

		const std::shared_ptr<const bwrenderer::RenderContext> render_context;


	};

}

