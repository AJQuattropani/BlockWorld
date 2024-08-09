#include "World.hpp"


namespace bwgame {


	World::World(const std::shared_ptr<BlockRegister>& block_register, 
		const std::shared_ptr<Context>& user_context, float ticks_per_second, float minutes_per_day, uint64_t seed)
		: world_gen(std::make_unique<WorldGenerator>(seed, block_register)),
		user_context(user_context), day_light_cycle(minutes_per_day, ticks_per_second), block_register(block_register),
		async_world_operations(32)
	{
		chunk_map.reserve((2 * user_context->ch_render_load_distance + 1) * (2 * user_context->ch_render_load_distance + 1));

		BW_INFO("World created.");

		for (chunk_coord_t x = (chunk_coord_t)user_context->player_position.x/CHUNK_WIDTH_BLOCKS 
			- (chunk_coord_t)user_context->ch_render_load_distance;
			x <= (chunk_coord_t)user_context->player_position.x/CHUNK_WIDTH_BLOCKS +
			(chunk_coord_t)user_context->ch_render_load_distance;
			x++)
		{
			for (chunk_coord_t z = (chunk_coord_t)user_context->player_position.z/CHUNK_WIDTH_BLOCKS - (chunk_coord_t)user_context->ch_render_load_distance;
				z <= (chunk_coord_t)user_context->player_position.z/CHUNK_WIDTH_BLOCKS + (chunk_coord_t)user_context->ch_render_load_distance;
				z++)
			{
				//todo move to own function
				async_world_operations.pushTask(std::bind(&World::loadChunk, this, ChunkCoords{x,z}));
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

		{
			std::scoped_lock<std::mutex> updateIteratorLock(world_data_lock);
			for (auto& [coords, chunk] : chunk_map) 
			{
				chunk.update();
			}
		}

		BW_DEBUG("Total chunks: %d", chunk_map.size());

	}

	void World::storeChunk()
	{
		// todo add storage
	}
	
	void World::mt_loadChunks()
	{
		TIME_FUNC("MT World Update and Load");

		chunk_coord_t ch_player_position_x = (chunk_coord_t)(user_context->player_position.x / CHUNK_WIDTH_BLOCKS_FLOAT);
		chunk_coord_t ch_player_position_z = (chunk_coord_t)(user_context->player_position.z / CHUNK_WIDTH_BLOCKS_FLOAT);
		chunk_coord_t differenceX = ch_player_position_x - player_last_chunk_pos_x;
		chunk_coord_t differenceZ = ch_player_position_z - player_last_chunk_pos_z;

		const chunk_coord_t& render_load_distance = user_context->ch_render_load_distance;

		if (differenceX == 0 && differenceZ == 0) return;

		size_t unload_chunks = 0;
		auto unload_filter = [ch_player_position_x,
			ch_player_position_z, render_load_distance](const auto& chunkIt) -> bool {
			return ;
			};

		{
			// todo replace with multithreaded unloading
			std::scoped_lock<std::mutex> unloadChunkLock(world_data_lock);
			for (auto it = chunk_map.begin(); it != chunk_map.end(); )
			{
				unload_chunks++;
				if (abs(it->first.x - ch_player_position_x) > render_load_distance
					|| abs(it->first.z - ch_player_position_z) > render_load_distance)
				//todo add memory loadoff
				{
					it = chunk_map.erase(it);
					continue;
				}
				it++;
			}
		}


		// start with square range in x and z of set of all values inclusive
		chunk_coord_t x1 = ch_player_position_x - render_load_distance,
			x2 = ch_player_position_x + render_load_distance,
			z1 = ch_player_position_z - render_load_distance,
			z2 = ch_player_position_z + render_load_distance,
			xmod1 = ch_player_position_x - render_load_distance,
			xmod2 = ch_player_position_x + render_load_distance;


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
		BW_ASSERT(z1 <= z2 + 1, "malformed bounds: z1 = %d, z2 = %d", z1, z2);
		BW_ASSERT(xmod1 <= xmod2 + 1, "malformed bounds: xmod1 = %d, xmod2 = %d", xmod1, xmod2);

		size_t chunk_loads = 0;

		// we always know we add to the upper bound
		for (chunk_coord_t x = x1; x < x2 + 1; x++)
		{
			// make rectangle strip WITH corner
			for (chunk_coord_t z = ch_player_position_z - render_load_distance; z < ch_player_position_z + render_load_distance + 1; z++)
			{
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

		player_last_chunk_pos_x = ch_player_position_x;
		player_last_chunk_pos_z = ch_player_position_z;

	}

	void World::loadChunk(ChunkCoords coords) {
		//todo add loading from memory
		
		// queue up task to prevent re-running the same task
		{
			std::scoped_lock<std::mutex> pushToLoadChunksLock(loading_chunks_lock);
			if (!loading_chunks.emplace(coords).second) return;
		}

		// do MT work
		Chunk chunk = world_gen->buildChunk(coords, &chunk_map);

		// do concurrent work
		{
			std::scoped_lock<std::mutex, std::mutex> addChunkToMapLock(world_data_lock, loading_chunks_lock);
			chunk_map.try_emplace(coords, std::move(chunk));

			// remove task so it may be ran again
			loading_chunks.erase(coords);
		}
	}

	/*void bwgame::World::unloadChunk(const ChunkCoords& coords)
	{
		chunk_map.erase(coords);
	}*/

	void World::unloadAllChunks()
	{
		std::scoped_lock<std::mutex> removeChunksFromMap(world_data_lock);
		//storeChunks
		chunk_map.clear();
		BW_INFO("All chunks unloaded.");
	}

}
