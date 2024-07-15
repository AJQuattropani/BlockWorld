#include "World.hpp"


namespace bwgame {


	World::World(const std::shared_ptr<BlockRegister>& block_register, 
		const std::shared_ptr<UserContext>& user_context, float ticks_per_second, float minutes_per_day, uint64_t seed)
		: world_gen(std::make_unique<WorldGenerator>(seed, block_register)),
		user_context(user_context), day_light_cycle(minutes_per_day, ticks_per_second),
		async_world_operations(32)
	{
		chunk_map.reserve((2 * user_context->ch_render_load_distance + 1) * (2 * user_context->ch_render_load_distance + 1));

		BW_INFO("World created.");

		for (chunk_coord_t x = (chunk_coord_t)user_context->player_position_x/15 - (chunk_coord_t)user_context->ch_render_load_distance;
			x <= (chunk_coord_t)user_context->player_position_x/15 + (chunk_coord_t)user_context->ch_render_load_distance;
			x++)
		{
			for (chunk_coord_t z = (chunk_coord_t)user_context->player_position_z/15 - (chunk_coord_t)user_context->ch_render_load_distance;
				z <= (chunk_coord_t)user_context->player_position_z/15 + (chunk_coord_t)user_context->ch_render_load_distance;
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

		day_light_cycle.update();

		std::scoped_lock<std::mutex> updateIteratorLock(world_data_lock);
		for (auto& [coords, chunk] : chunk_map) chunk.update();
		

		BW_DEBUG("Total chunks: %d", chunk_map.size());

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

		//		const auto& [Iterator, success] = chunk_map.try_emplace(coords, Chunk(coords, chunk_map));

		//		if (!success) continue;
		//		auto& chunk = Iterator->second;
		//		//todo move to own function
		//		async_chunk_loads.push_back(std::jthread(&World::build_func, this, coords, &chunk));
		//		load_count++;
		//	}
		//}
		//GL_DEBUG("%i chunks loaded.", load_count);

		chunk_coord_t ch_player_position_x = (chunk_coord_t)user_context->player_position_x / 15;
		chunk_coord_t ch_player_position_z = (chunk_coord_t)user_context->player_position_z / 15;
		chunk_coord_t differenceX = ch_player_position_x - player_last_chunk_pos_x;
		chunk_coord_t differenceZ = ch_player_position_z - player_last_chunk_pos_z;

		const chunk_coord_t& render_load_distance = user_context->ch_render_load_distance;

		if (differenceX == 0 && differenceZ == 0) return;

		size_t unload_chunks = 0;
		auto unload_filter = [ch_player_position_x,
			ch_player_position_z, render_load_distance](const auto& chunkIt) -> bool {
			return abs(chunkIt.first.x - ch_player_position_x) > render_load_distance
				|| abs(chunkIt.first.z - ch_player_position_z) > render_load_distance;
			};
		for (auto& coords : chunk_map | std::views::filter(unload_filter) | std::views::keys) 
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

		BW_DEBUG("Last chunk: {%d, %d} into: {%d, %d}", player_last_chunk_pos_x, player_last_chunk_pos_z, ch_player_position_x, ch_player_position_z);

		// if the change in x was positive, we need to prune the lower value
		if (differenceX > 0)
		{
			x1 = player_last_chunk_pos_x + render_load_distance + 1;
			xmod2 = x1 - 1;
		} // otherwise the change the upper bound
		else 
		{
			x2 = player_last_chunk_pos_x - render_load_distance - 1;
			xmod1 = x2 + 1;
		}

		// if the change in z was positive, we need to prune the lower value
		if (differenceZ > 0)
		{
			z1 = player_last_chunk_pos_z + render_load_distance + 1;
		} // otherwise the change the upper bound
		else
		{
			z2 = player_last_chunk_pos_z - render_load_distance - 1;
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
				BW_ASSERT(chunk_map.find({ x,z }) == chunk_map.end(), "Repeat chunk building at {%d,%d}.", x, z);
				async_world_operations.pushTask(std::bind(&World::loadChunk, this, ChunkCoords{x,z}));
				chunk_loads++;
			}
		}

		for (chunk_coord_t z = z1; z < z2 + 1; z++)
		{
			// make rectangle strip WITHOUT corner
			for (chunk_coord_t x = xmod1; x < xmod2 + 1; x++)
			{
				//BW_ASSERT(chunk_map.find({ x,z }) == chunk_map.end(), "Repeat chunk building at {%d,%d}.", x, z);
				async_world_operations.pushTask(std::bind(&World::loadChunk, this, ChunkCoords{ x,z }));
				chunk_loads++;
			}
		}

		size_t expected_chunk_loads = (abs(differenceX) + abs(differenceZ)) * (render_load_distance * 2 + 1) - abs(differenceX * differenceZ);
		BW_ASSERT(chunk_loads == expected_chunk_loads, "Chunk render error: Expected: %d, Got: %d", expected_chunk_loads, chunk_loads);

		BW_DEBUG("%d chunks loaded.", chunk_loads);

		player_last_chunk_pos_x = ch_player_position_x;
		player_last_chunk_pos_z = ch_player_position_z;

	}

	void World::loadChunk(ChunkCoords coords) {
		//todo add
		Chunk chunk = world_gen->buildChunk(coords, &chunk_map);
		{
			std::scoped_lock<std::mutex> addChunkToMapLock(world_data_lock);
			chunk_map.try_emplace(coords, std::move(chunk));
		}
	}

	void bwgame::World::unloadChunk(const ChunkCoords& coords)
	{
		{
			std::scoped_lock<std::mutex> removeChunkFromMapLock(world_data_lock);
			//storeChunk();
			chunk_map.erase(coords);
		}
		BW_INFO("Chunk { %i, %i } unloaded.", coords.x, coords.z);
	}

	void World::unloadAllChunks()
	{
		std::scoped_lock<std::mutex> removeChunksFromMap(world_data_lock);
		//storeChunks
		chunk_map.clear();
		BW_INFO("All chunks unloaded.");
	}

}
