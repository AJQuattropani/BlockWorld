#pragma once

#include <glm/glm.hpp>

#include <memory>

#include "DayCycle.hpp"
#include "Skybox.hpp"
#include "ThreadList.hpp"
#include "World.hpp"

namespace bwrenderer {

	class WorldRenderer {
	public:
		WorldRenderer(const std::shared_ptr<const bwgame::World> world_ref, const std::shared_ptr<bwgame::UserContext>& user_context,
			const std::shared_ptr<bwrenderer::RenderContext> render_context) 
			: world_ref(world_ref), thread_list(), render_context(render_context), user_context(user_context),
			block_shader("Blocks/World", "block_shader"), shadow_shader("Blocks/World", "shadows")
		{
		initializeDepthBuffer();
		setBlockShaderConsts();
		setShadowShaderConsts();
		}

		void render()
		{
			TIME_FUNC("World Render");

			float radialTime = glm::mod<float>(world_ref->day_light_cycle.time_game_days, 1.0f) * glm::radians(360.0);

			glm::vec3 lightPosition = 15.0f * user_context->ch_render_load_distance * glm::vec3(
				glm::cos(glm::mod(radialTime, glm::radians(180.0f))),
				glm::sin(glm::mod(radialTime, glm::radians(180.0f))),
				0.0);

			const float near_plane = 1.0f, far_plane = 256.0f * 2.0f;
			glm::mat4 lightProjection = glm::ortho(
				-15.0f * render_context->ch_shadow_window_distance / 2,
				15.0f * render_context->ch_shadow_window_distance / 2,
				-15.0f * render_context->ch_shadow_window_distance / 2,
				15.0f * render_context->ch_shadow_window_distance / 2, near_plane, far_plane);
			glm::mat4 lightView = glm::lookAt(
				lightPosition
				+ glm::vec3(user_context->player_position_x, user_context->player_position_y, user_context->player_position_z),
				glm::vec3(0.0f)
				+ glm::vec3(user_context->player_position_x, user_context->player_position_y, user_context->player_position_z),
				glm::vec3(0.0f, 1.0f, 0.0f));
			glm::mat4 lightSpaceMatrix = lightProjection * lightView;

			bwrenderer::TextureBuffer& texture = render_context->texture_cache.findOrLoad("Blocks", "blockmap.jpeg");
			bwrenderer::TextureBuffer& depth_map = render_context->texture_cache.find("world_depth_map");

			glViewport(0, 0, 8 * render_context->screen_width_px, 8 * render_context->screen_height_px);
			depth_buffer.bind();
			glClear(GL_DEPTH_BUFFER_BIT);

			shadow_shader.bind();

			shadow_shader.setUniformMat4f("lightSpaceMatrix", lightSpaceMatrix);
			glDisable(GL_CULL_FACE);
			//glCullFace(GL_FRONT);
			for (const auto& [coords, chunk] : world_ref->chunk_map)
			{
				chunk.render(shadow_shader);
			}
			glEnable(GL_CULL_FACE);
			glCullFace(GL_BACK);

			depth_buffer.unbind();


			glViewport(0, 0, render_context->screen_width_px, render_context->screen_height_px);

#define RENDER_DEBUG 0
#if RENDER_DEBUG == 1
			/**DEBUG*/

			static bwrenderer::Shader debugShader("Blocks/World", "debugshader");

			debugShader.bind();
			debugShader.setUniform1f("near_plane", near_plane);
			debugShader.setUniform1f("far_plane", far_plane);
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, depth_map.textureID);

			static unsigned int quadVAO = 0;
			static unsigned int quadVBO = 0;
			{
				if (quadVAO == 0)
				{
					float quadVertices[] = {
						// positions        // texture Coords
						-1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
						-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
						 1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
						 1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
					};
					// setup plane VAO
					glGenVertexArrays(1, &quadVAO);
					glGenBuffers(1, &quadVBO);
					glBindVertexArray(quadVAO);
					glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
					glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
					glEnableVertexAttribArray(0);
					glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
					glEnableVertexAttribArray(1);
					glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
				}
				glBindVertexArray(quadVAO);
				glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
				glBindVertexArray(0);
			}

			debugShader.unbind();
#else

			/********/


			block_shader.bind();
			glActiveTexture(GL_TEXTURE0);
			glBindTexture(GL_TEXTURE_2D, texture.textureID);
			glActiveTexture(GL_TEXTURE1);
			glBindTexture(GL_TEXTURE_2D, depth_map.textureID);

			block_shader.setUniform3f("dir_light.position", lightPosition.x, lightPosition.y, lightPosition.z);
			block_shader.setUniform1f("dir_light.day_time", world_ref->day_light_cycle.time_game_days);
			block_shader.setUniform1f("dir_light.radial_time", world_ref->day_light_cycle.time_game_days * glm::radians(360.0));
			block_shader.setUniform1f("dir_light.environmental_brightness",
				glm::sqrt(glm::abs(glm::sin(radialTime)))
				* (1.0f - glm::sign(radialTime - glm::radians(180.0f)) / 2.0f) / 1.5f);
#define CAM
#ifdef CAM
			block_shader.setUniformMat4f("view", render_context->view_matrix);
			block_shader.setUniformMat4f("projection", render_context->projection_matrix);
#else
			block_shader.setUniformMat4f("view", lightView);
			block_shader.setUniformMat4f("projection", lightProjection);
#endif
			block_shader.setUniformMat4f("lightSpaceMatrix", lightSpaceMatrix);
			for (auto& [coords, chunk] : world_ref->chunk_map)
			{
				chunk.render(block_shader);
			}

			skybox.render(*render_context, world_ref->day_light_cycle);
#endif
		}


	private:
		const std::shared_ptr<const bwgame::World> world_ref;
		std::unique_ptr<utils::ThreadList> thread_list;
		const std::shared_ptr<const bwgame::UserContext> user_context;
		const std::shared_ptr<bwrenderer::RenderContext> render_context;

		bwrenderer::Shader block_shader, shadow_shader;
		bwrenderer::frame_buffer depth_buffer;
		bwrenderer::SkyBox skybox;
		static constexpr int SHADOW_WIDTH = 720, SHADOW_HEIGHT = 720;
		static constexpr float GRADIENT = 30.0f;
		static constexpr float INV_GRADIENT = 1.0f / GRADIENT;

	private:
		void initializeDepthBuffer()
		{
			GLuint depth_map;
			glGenTextures(1, &depth_map);
			glBindTexture(GL_TEXTURE_2D, depth_map);
			{
				glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT,
					8 * render_context->screen_width_px,
					8 * render_context->screen_height_px, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
				float borderColor[] = { 1.0f, 1.0f, 1.0f, 1.0f };
				glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
			}
			glBindTexture(GL_TEXTURE_2D, 0);

			render_context->texture_cache.push("world_depth_map", 
				bwrenderer::TextureBuffer{ .textureID = depth_map, .width = SHADOW_WIDTH, .height = SHADOW_HEIGHT });

			depth_buffer.bind();
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_map, 0);
			glDrawBuffer(GL_NONE);
			glReadBuffer(GL_NONE);
			GL_ASSERT(glCheckFramebufferStatus(GL_FRAMEBUFFER), "Shadow buffer setup improperly.");
			depth_buffer.unbind();
		}

		void setBlockShaderConsts()
		{
			bwrenderer::TextureBuffer& texture = render_context->texture_cache.findOrLoad("Blocks", "blockmap.jpeg");

			block_shader.bind();
			block_shader.setUniform1i("block_texture", 0);
			block_shader.setUniform1i("shadow_map", 1);
			block_shader.setUniform2f("image_size", static_cast<GLfloat>(texture.width), static_cast<GLfloat>(texture.height));
			block_shader.setUniform3f("dir_light.ambient", 0.2f, 0.2f, 0.2f);
			block_shader.setUniform3f("dir_light.diffuse", 0.85f, 0.85f, 0.85f);
			block_shader.setUniform3f("dir_light.specular", 0.2f, 0.2f, 0.2f);
			block_shader.setUniform1f("fog.gradient", GRADIENT);
			
			static const float GRAD_COEFF = glm::pow(1 - INV_GRADIENT, INV_GRADIENT);
			block_shader.setUniform1f("fog.density", GRAD_COEFF / (user_context->ch_render_load_distance * 15.0f));
			block_shader.unbind();
		}

		void setShadowShaderConsts()
		{
			shadow_shader.bind();
			shadow_shader.unbind();
		}


		friend bwgame::World;
	};

	

}