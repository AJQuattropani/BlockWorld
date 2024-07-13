#include "World.h"


namespace bwgame {


	World::World(const std::shared_ptr<BlockRegister>& block_register, const std::shared_ptr<bwrenderer::RenderContext>& context,
		float ticks_per_second, float minutes_per_day, uint64_t seed)
		: worldGen(std::make_unique<WorldGenerator>(seed, block_register)),
		context(context), dayLightCycle(minutes_per_day, ticks_per_second),
		blockShader("Blocks/World", "block_shader"), shadowShader("Blocks/World", "shadows"), depth_buffer(), async_world_operations(32)
	{
		chunkMap.reserve((2 * context->ch_render_load_distance + 1) * (2 * context->ch_render_load_distance + 1));


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

		static constexpr float gradient = 30.0f;
		static constexpr float inv_grad = 1.0f / gradient;

		bwrenderer::TextureBuffer& texture = context->texture_cache.findOrLoad("Blocks", "blockmap.jpeg");
		blockShader.bind();
		blockShader.setUniform1i("block_texture", 0);
		blockShader.setUniform1i("shadow_map", 1);
		blockShader.setUniform2f("image_size", texture.width, texture.height);
		blockShader.setUniform3f("dir_light.ambient", 0.2f, 0.2f, 0.2f);
		blockShader.setUniform3f("dir_light.diffuse", 0.85f, 0.85f, 0.85f);
		blockShader.setUniform3f("dir_light.specular", 0.2f, 0.2f, 0.2f);
		blockShader.setUniform1f("fog.gradient", gradient);
		blockShader.setUniform1f("fog.density", glm::pow(1 - inv_grad, inv_grad) / (context->ch_render_load_distance * 15.0));
		blockShader.unbind();

		shadowShader.bind();
		shadowShader.unbind();

		BW_INFO("World created.");

		for (chunk_coord_t x = context->player_position_x - context->ch_render_load_distance; x < context->player_position_x + context->ch_render_load_distance + 1;
			x++)
		{
			for (chunk_coord_t z = context->player_position_x - context->ch_render_load_distance; z < context->player_position_x + context->ch_render_load_distance + 1;
				z++)
			{
				//todo move to own function
				async_world_operations.pushTask(std::bind(&World::loadChunk, this, ChunkCoords{x,z}));
				BW_DEBUG("Chunk {%d, %d} loaded.", x, z);
			}
		}

	}


	World::~World()
	{
		unloadAllChunks();
		BW_INFO("World destroyed.");
	}

	void World::update() // todo created shared instance of camera/player
	{
		mt_loadChunks();

		dayLightCycle.update();

		std::scoped_lock<std::mutex> updateIteratorLock(worldDataLock);
		for (auto& [coords, chunk] : chunkMap) chunk.update();
		

		BW_DEBUG("Total chunks: %d", chunkMap.size());

	}

	void World::render()
	{
		TIME_FUNC("World Render");

		float radialTime = glm::mod(dayLightCycle.time_game_days, 1.0f) * glm::radians(360.0);

		glm::vec3 lightPosition = 15.0f * context->ch_render_load_distance * glm::vec3(
			glm::cos(glm::mod(radialTime, glm::radians(180.0f))),
			glm::sin(glm::mod(radialTime, glm::radians(180.0f))),
			0.0);

		const float near_plane = 1.0f, far_plane = 256.0f * 2.0f;
		glm::mat4 lightProjection = glm::ortho(
			-15.0f * context->ch_shadow_window_distance/2, 
			15.0f * context->ch_shadow_window_distance/2,
			-15.0f * context->ch_shadow_window_distance/2,
			15.0f * context->ch_shadow_window_distance/2, near_plane, far_plane);
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
		blockShader.setUniform3f("camPos", context->player_position_x, context->player_position_y, context->player_position_z);
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

		//ChunkCoords coords{};
		//std::vector<std::jthread> async_chunk_loads;

		//uint8_t load_count = 0;

		//for (coords.x = context->player_position_x/15 - context->ch_render_load_distance / 2;
		//	coords.x <= context->player_position_x/15 + context->ch_render_load_distance / 2;
		//	coords.x++)
		//{
		//	for (coords.z = context->player_position_z/15 - context->ch_render_load_distance / 2;
		//		coords.z <= context->player_position_z/15 + context->ch_render_load_distance / 2;
		//		coords.z++)
		//	{

		//		const auto& [Iterator, success] = chunkMap.try_emplace(coords, Chunk(coords, chunkMap));

		//		if (!success) continue;
		//		auto& chunk = Iterator->second;
		//		//todo move to own function
		//		async_chunk_loads.push_back(std::jthread(&World::build_func, this, coords, &chunk));
		//		load_count++;
		//	}
		//}
		//GL_DEBUG("%i chunks loaded.", load_count);

		chunk_coord_t ch_player_position_x = context->player_position_x / 15;
		chunk_coord_t ch_player_position_z = context->player_position_z / 15;
		chunk_coord_t differenceX = ch_player_position_x - playerLastChunkPosX;
		chunk_coord_t differenceZ = ch_player_position_z - playerLastChunkPosZ;

		const chunk_coord_t& render_load_distance = context->ch_render_load_distance;

		if (differenceX == 0 && differenceZ == 0) return;

		size_t unload_chunks = 0;
		auto unload_filter = [ch_player_position_x,
			ch_player_position_z, render_load_distance](const auto& chunkIt) -> bool {
			return abs(chunkIt.first.x - ch_player_position_x) > render_load_distance
				|| abs(chunkIt.first.z - ch_player_position_z) > render_load_distance;
			};
		for (auto& coords : chunkMap | std::views::filter(unload_filter) | std::views::keys) 
		{
			unloadChunk(coords);
			unload_chunks++;
		}
		BW_DEBUG("%d chunks removed.", unload_chunks);


		// start with square range in x and z of set of all values inclusive
		chunk_coord_t x1 = ch_player_position_x - render_load_distance,
			x2 = ch_player_position_x + render_load_distance,
			z1 = ch_player_position_z - render_load_distance,
			z2 = ch_player_position_z + render_load_distance,
			xmod1 = ch_player_position_x - render_load_distance,
			xmod2 = ch_player_position_x + render_load_distance;

		BW_DEBUG("Last chunk: {%d, %d} into: {%d, %d}", playerLastChunkPosX, playerLastChunkPosZ, ch_player_position_x, ch_player_position_z);

		// if the change in x was positive, we need to prune the lower value
		if (differenceX > 0)
		{
			x1 = playerLastChunkPosX + render_load_distance + 1;
			xmod2 = x1 - 1;
		} // otherwise the change the upper bound
		else 
		{
			x2 = playerLastChunkPosX - render_load_distance - 1;
			xmod1 = x2 + 1;
		}

		// if the change in z was positive, we need to prune the lower value
		if (differenceZ > 0)
		{
			z1 = playerLastChunkPosZ + render_load_distance + 1;
		} // otherwise the change the upper bound
		else
		{
			z2 = playerLastChunkPosZ - render_load_distance - 1;
		}

		BW_ASSERT(x1 <= x2 + 1, "malformed bounds: x1 = %d, x2 = %d", x1, x2);
		BW_DEBUG("X: {%d, %d}", x1, x2);
		BW_ASSERT(z1 <= z2 + 1, "malformed bounds: z1 = %d, z2 = %d", z1, z2);
		BW_DEBUG("Z: {%d, %d}", z1, z2);
		BW_ASSERT(xmod1 <= xmod2 + 1, "malformed bounds: xmod1 = %d, xmod2 = %d", xmod1, xmod2);

		size_t chunk_loads = 0;

		// we always know we add to the upper bound
		for (chunk_coord_t x = x1; x < x2 + 1; x++)
		{
			// make rectangle strip WITH corner
			for (chunk_coord_t z = ch_player_position_z - render_load_distance; z < ch_player_position_z + render_load_distance + 1; z++)
			{
				BW_ASSERT(chunkMap.find({ x,z }) == chunkMap.end(), "Repeat chunk building at {%d,%d}.", x, z);
				async_world_operations.pushTask(std::bind(&World::loadChunk, this, ChunkCoords{x,z}));
				chunk_loads++;
			}
		}

		for (chunk_coord_t z = z1; z < z2 + 1; z++)
		{
			// make rectangle strip WITHOUT corner
			for (chunk_coord_t x = xmod1; x < xmod2 + 1; x++)
			{
				//BW_ASSERT(chunkMap.find({ x,z }) == chunkMap.end(), "Repeat chunk building at {%d,%d}.", x, z);
				async_world_operations.pushTask(std::bind(&World::loadChunk, this, ChunkCoords{ x,z }));
				chunk_loads++;
			}
		}

		size_t expected_chunk_loads = (abs(differenceX) + abs(differenceZ)) * (render_load_distance * 2 + 1) - abs(differenceX * differenceZ);
		BW_ASSERT(chunk_loads == expected_chunk_loads, "Chunk render error: Expected: %d, Got: %d", expected_chunk_loads, chunk_loads);

		BW_DEBUG("%d chunks loaded.", chunk_loads);

		playerLastChunkPosX = ch_player_position_x;
		playerLastChunkPosZ = ch_player_position_z;

	}

	void World::loadChunk(ChunkCoords coords) {
		//todo add
		Chunk chunk = worldGen->buildChunk(coords, &chunkMap);
		{
			std::scoped_lock<std::mutex> addChunkToMapLock(worldDataLock);
			chunkMap.try_emplace(coords, std::move(chunk));
		}
	}



	void bwgame::World::unloadChunk(const ChunkCoords& coords)
	{
		{
			std::scoped_lock<std::mutex> removeChunkFromMapLock(worldDataLock);
			//storeChunk();
			chunkMap.erase(coords);
		}
		BW_INFO("Chunk { %i, %i } unloaded.", coords.x, coords.z);
	}

	void World::unloadAllChunks()
	{
		std::scoped_lock<std::mutex> removeChunksFromMap(worldDataLock);
		//storeChunks
		chunkMap.clear();
		BW_INFO("All chunks unloaded.");
	}

}
