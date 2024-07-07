#include "World.h"


namespace bwgame {

	void World::build_func(ChunkCoords coords, Chunk* chunk) {
		worldGen->buildChunk(coords, *chunk);
	}

	World::World(const std::shared_ptr<BlockRegister>& block_register, const std::shared_ptr<bwrenderer::RenderContext>& context,
		float ticks_per_second, float minutes_per_day, uint64_t seed)
		: worldGen(std::make_unique<WorldGenerator>(seed, block_register)),
		context(context), dayLightCycle(minutes_per_day, ticks_per_second),
		blockShader("Blocks/World", "block_shader"), shadowShader("Blocks/World", "shadows"), depth_buffer()
	{
		GLuint depth_map;
		glGenTextures(1, &depth_map);
		glBindTexture(GL_TEXTURE_2D, depth_map);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 8 * context->screen_width_px, 8 * context->screen_height_px, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
		float borderColor[] = { 1.0f, 1.0f, 1.0f, 1.0f };
		glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);
		//glBindTexture(GL_TEXTURE_2D, 0);

		this->depth_map = context->texture_cache.push("world_depth_buffer", bwrenderer::TextureBuffer{.textureID = depth_map, .width = SHADOW_WIDTH, .height = SHADOW_HEIGHT});

		depth_buffer.bind();
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth_map, 0);
		glDrawBuffer(GL_NONE);
		glReadBuffer(GL_NONE);

		GL_ASSERT(glCheckFramebufferStatus(GL_FRAMEBUFFER), "Shadow buffer setup improperly.");

		depth_buffer.unbind();


		bwrenderer::TextureBuffer& texture = context->texture_cache.findOrLoad("Blocks", "blockmap.jpeg");
		blockShader.bind();
		blockShader.setUniform1i("block_texture", 0);
		blockShader.setUniform1i("shadow_map", 1);
		blockShader.setUniform2f("image_size", texture.width, texture.height);
		blockShader.setUniform3f("dir_light.ambient", 0.2f, 0.2f, 0.2f);
		blockShader.setUniform3f("dir_light.diffuse", 0.85f, 0.85f, 0.85f);
		blockShader.setUniform3f("dir_light.specular", 0.2f, 0.2f, 0.2f);
		blockShader.unbind();

		shadowShader.bind();
		shadowShader.unbind();

		BW_INFO("World created.");
	}

	World::~World()
	{
		unloadAllChunks();
		BW_INFO("World destroyed.");
	}

	void World::update() // todo created shared instance of camera/player
	{
		mt_loadChunks();

		{
			uint8_t unload_count = 0;
			TIME_FUNC("World Unload");
			for (auto it = chunkMap.begin(); it != chunkMap.end(); )
			{
				if (abs(it->first.x - int64_t(context->player_position_x) / 15) > context->ch_render_unload_distance / 2
					|| abs(it->first.z - int64_t(context->player_position_z) / 15) > context->ch_render_unload_distance / 2)
				{
					it = unloadChunk(it);
					unload_count++;
				}
				else {
					it->second.update();
					it++;
				}
			}
			GL_DEBUG("%i chunks unloaded.", unload_count);

		}

		dayLightCycle.update();
	}

	void World::render()
	{
		TIME_FUNC("World Render");

		float radialTime = glm::mod(dayLightCycle.time_game_days, 1.0f) * glm::radians(360.0);

		glm::vec3 lightPosition = 15.0f * context->ch_render_load_distance * glm::vec3(
			glm::cos(glm::mod(radialTime, glm::radians(180.0f))),
			glm::sin(glm::mod(radialTime, glm::radians(180.0f))),
			0.0);

		const float near_plane = 0.1f, far_plane = 2.0f * 15.0f * context->ch_render_load_distance;
		glm::mat4 lightProjection = glm::ortho(
			-15.0f * context->ch_render_load_distance, 
			15.0f * context->ch_render_load_distance, 
			-15.0f * context->ch_render_load_distance, 
			15.0f * context->ch_render_load_distance, near_plane, far_plane);
		glm::mat4 lightView = glm::lookAt(
			lightPosition 
			+ glm::vec3(context->player_position_x, context->player_position_y, context->player_position_z),
			glm::vec3(0.0f) 
			+ glm::vec3(context->player_position_x, context->player_position_y, context->player_position_z),
			glm::vec3(0.0f, 1.0f, 0.0f));
		glm::mat4 lightSpaceMatrix = lightProjection * lightView;

		bwrenderer::TextureBuffer& texture = context->texture_cache.findOrLoad("Blocks", "blockmap.jpeg");
		
		glViewport(0, 0, 8 * context->screen_width_px, 8 * context->screen_height_px);
		depth_buffer.bind();
		glClear(GL_DEPTH_BUFFER_BIT);

		shadowShader.bind();

		shadowShader.setUniformMat4f("lightSpaceMatrix", lightSpaceMatrix);
		glDisable(GL_CULL_FACE);
		//glCullFace(GL_FRONT);
		for (auto& [coords, chunk] : chunkMap)
		{
			chunk.render(shadowShader);
		}
		glEnable(GL_CULL_FACE);
		glCullFace(GL_BACK);

		depth_buffer.unbind();

		
		glViewport(0, 0, context->screen_width_px, context->screen_height_px);

#define RENDER_DEBUG 0
#if RENDER_DEBUG == 1
		/**DEBUG*/

		static bwrenderer::Shader debugShader("Blocks/World","debugshader");

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
		
		
		blockShader.bind();
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texture.textureID);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, depth_map.textureID);

		blockShader.setUniform3f("dir_light.position", lightPosition.x, lightPosition.y, lightPosition.z);
		blockShader.setUniform1f("dir_light.day_time", dayLightCycle.time_game_days);
		blockShader.setUniform1f("dir_light.radial_time", dayLightCycle.time_game_days * glm::radians(360.0));
		blockShader.setUniform1f("dir_light.environmental_brightness",
			glm::sqrt(glm::abs(glm::sin(radialTime))) 
			* (1.0 - glm::sign(radialTime-glm::radians(180.0f))/2.0f)/1.5f);
#define CAM
#ifdef CAM
		blockShader.setUniformMat4f("view", context->viewMatrix);
		blockShader.setUniformMat4f("projection", context->projectionMatrix);
#else
		blockShader.setUniformMat4f("view", lightView);
		blockShader.setUniformMat4f("projection", lightProjection);
#endif
		blockShader.setUniformMat4f("lightSpaceMatrix", lightSpaceMatrix);
		for (auto& [coords, chunk] : chunkMap)
		{
			chunk.render(blockShader);
		}

		skybox.render(*context, dayLightCycle);
#endif
	}

	void World::storeChunk()
	{
		// todo add storage
	}
	
	void World::mt_loadChunks()
	{
		TIME_FUNC("MT World Update and Load");

		ChunkCoords coords{};
		std::vector<std::jthread> async_chunk_loads;

		uint8_t load_count = 0;

		for (coords.x = context->player_position_x/15 - context->ch_render_load_distance / 2;
			coords.x <= context->player_position_x/15 + context->ch_render_load_distance / 2;
			coords.x++)
		{
			for (coords.z = context->player_position_z/15 - context->ch_render_load_distance / 2;
				coords.z <= context->player_position_z/15 + context->ch_render_load_distance / 2;
				coords.z++)
			{

				if (const auto& It = chunkMap.find(coords); It != chunkMap.end()) continue;

				const auto& [Iterator, success] = chunkMap.emplace(coords, coords);
				auto& chunk = Iterator->second;
				//todo move to own function
				async_chunk_loads.push_back(std::jthread(&World::build_func, this, coords, &chunk));
				load_count++;
			}
		}
		GL_DEBUG("%i chunks loaded.", load_count);

	}

	void World::loadChunks()
	{
		TIME_FUNC("World Update and Load");

		ChunkCoords coords{};
		std::vector<std::jthread> async_chunk_loads;

		for (coords.x = context->player_position_x/15 - context->ch_render_load_distance / 2;
			coords.x <= context->player_position_x/15 + context->ch_render_load_distance / 2;
			coords.x++)
			for (coords.z = context->player_position_z/15 - context->ch_render_load_distance / 2;
				coords.z <= context->player_position_z/15 + context->ch_render_load_distance / 2;
				coords.z++)
		{

			if (const auto& It = chunkMap.find(coords); It != chunkMap.end()) continue;

			const auto& [Iterator, success] = chunkMap.emplace(coords, coords);
			auto& chunk = Iterator->second;
			//todo move to own function
			srand(coords.seed);
			build_func(coords, &chunk);
		}

	}


	void World::loadChunk(ChunkCoords coords)
	{

		if (const auto& It = chunkMap.find(coords); It != chunkMap.end()) return;

		const auto& [Iterator, success] = chunkMap.emplace(coords, coords);
		auto& chunk = Iterator->second;
		//todo move to own function
		
		build_func(coords, &chunk);
	}


	void bwgame::World::unloadChunk(const ChunkCoords& coords)
	{
		//storeChunk();
		chunkMap.erase(coords);
		BW_INFO("Chunk { %i, %i } unloaded.", coords.x, coords.z);
	}

	const std::unordered_map<ChunkCoords, Chunk>::iterator World::unloadChunk(const std::unordered_map<ChunkCoords, Chunk>::iterator& it)
	{
		//storeChunk(it);
		return chunkMap.erase(it);
	}

	void World::unloadAllChunks()
	{
		chunkMap.clear();
		BW_INFO("All chunks unloaded.");
	}

}