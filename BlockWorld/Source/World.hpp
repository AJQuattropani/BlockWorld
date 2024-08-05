#pragma once

#include "Debug.hpp"
#include "Timer.hpp"
#include "Chunk.hpp"
#include "Camera.hpp"
#include "WorldGenerator.hpp"
#include "DayCycle.hpp"
#include "ThreadList.hpp"

#include <unordered_map>
#include <future>
#include <thread>
#include <unordered_set>

namespace bwrenderer {

	class WorldRenderer;
}

namespace bwgame {


	/*
	* The goal is to eventually have file writing/reading.
	*/
	class World
	{
	public:
		World(const std::shared_ptr<BlockRegister>& block_register, const std::shared_ptr<UserContext>& user_context,
			float ticks_per_second,
			float minutes_per_day = 1.0, uint64_t seed = 38513759);

		~World();

		void update();

		void setBlock(const Block& block, WorldBlockCoords coords)
		{
			chunk_coord_t c_x = floor((float)(coords.x) / CHUNK_WIDTH_BLOCKS);
			const block_coord_t x = ((coords.x % CHUNK_WIDTH_BLOCKS) + CHUNK_WIDTH_BLOCKS) % CHUNK_WIDTH_BLOCKS;
			chunk_coord_t c_z = floor((float)(coords.z) / CHUNK_WIDTH_BLOCKS);
			const block_coord_t z = ((coords.z % CHUNK_WIDTH_BLOCKS) + CHUNK_WIDTH_BLOCKS) % CHUNK_WIDTH_BLOCKS;
			//if (coords.x < 0) c_x--;
			//if (coords.z < 0) c_z--;
			if (const auto& chunk_it = chunk_map.find(ChunkCoords{ c_x, c_z }); chunk_it != chunk_map.end())
			{
				chunk_it->second.setBlock(BlockCoords{
					x, coords.y, z }, block);
			}
		}

		void destroyBlock(WorldBlockCoords coords)
		{
			chunk_coord_t c_x = floor((float)(coords.x) / CHUNK_WIDTH_BLOCKS);
			const block_coord_t x = ((coords.x % CHUNK_WIDTH_BLOCKS) + CHUNK_WIDTH_BLOCKS) % CHUNK_WIDTH_BLOCKS;
			chunk_coord_t c_z = floor((float)(coords.z) / CHUNK_WIDTH_BLOCKS);
			const block_coord_t z = ((coords.z % CHUNK_WIDTH_BLOCKS) + CHUNK_WIDTH_BLOCKS) % CHUNK_WIDTH_BLOCKS;
			//if (coords.x < 0); c_x--;
			//if (coords.z < 0); c_z--;
			if (const auto& chunk_it = chunk_map.find(ChunkCoords{ c_x, c_z }); chunk_it != chunk_map.end())
			{
				chunk_it->second.deleteBlock(BlockCoords{
					x, coords.y, z });
			}
		}

		void storeChunk();

	private:
		std::unordered_map<ChunkCoords, Chunk> chunk_map;
		std::unique_ptr<WorldGenerator> world_gen;
		DayLightCycle day_light_cycle;
		const std::shared_ptr<UserContext> user_context;

		chunk_coord_t player_last_chunk_pos_x = 0, player_last_chunk_pos_z = 0;

		utils::ThreadList async_world_operations; // number of threads may be subject to change based on CPU
		std::mutex world_data_lock;
		std::unordered_set<ChunkCoords> loading_chunks;
		std::mutex loading_chunks_lock;

	private:
		void mt_loadChunks();
		void loadChunk(ChunkCoords coords);
		
		
		// void unloadChunk(const ChunkCoords& coords);

		void unloadAllChunks();
		
		friend bwrenderer::WorldRenderer;
	};




}