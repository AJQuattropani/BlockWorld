#pragma once

#include "Debug.hpp"
#include "Timer.hpp"
#include "Chunk.hpp"
#include "WorldGenerator.hpp"
#include "DayCycle.hpp"
#include "ThreadList.hpp"
#include "Context.hpp"

#include <unordered_map>
#include <thread>

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
		World(const std::shared_ptr<BlockRegister>& block_register, const std::shared_ptr<Context>& user_context,
			float ticks_per_second,
			float minutes_per_day = 1.0, uint64_t seed = 38513759);

		~World();

		void update();

		void setBlock(const Block& block, WorldBlockCoords coords)
		{
			auto [chunk_coords, block_coords] = decomposeCoordinates(coords);
			if (const auto& chunk_it = chunk_map.find(chunk_coords); chunk_it != chunk_map.end())
			{
				chunk_it->second.setBlock(block_coords, block);
			}
		}

		inline bool checkBlock(const WorldBlockCoords& coords) const
		{
			auto [chunk_coords, block_coords] = decomposeCoordinates(coords);
			if (const auto& chunk_it = chunk_map.find(chunk_coords); chunk_it != chunk_map.end())
				return chunk_it->second.checkBlock(block_coords);
			return false;
		}

		const Block& getBlock(WorldBlockCoords coords) const
		{
			auto [chunk_coords, block_coords] = decomposeCoordinates(coords);
			if (const auto& chunk_it = chunk_map.find(chunk_coords); chunk_it != chunk_map.end()) 
				return chunk_it->second.getBlockConst(block_coords.x, block_coords.y, block_coords.z);
			return block_register->air;
		}


		void destroyBlock(WorldBlockCoords coords)
		{
			auto [chunk_coords, block_coords] = decomposeCoordinates(coords);
			if (const auto& chunk_it = chunk_map.find(chunk_coords); chunk_it != chunk_map.end())
			{
				chunk_it->second.deleteBlock(block_coords);
			}
		}

		inline const std::shared_ptr<BlockRegister>& getBlockRegister() const
		{
			return block_register;
		}

		void storeChunk();

	public:
		static inline std::pair<ChunkCoords, BlockCoords> decomposeCoordinates(const WorldBlockCoords& coords)
		{
			ChunkCoords chunk_coords{ (chunk_coord_t)floor((coords.x) / CHUNK_WIDTH_BLOCKS_FLOAT), (chunk_coord_t)floor((coords.z) / CHUNK_WIDTH_BLOCKS_FLOAT) };
			BlockCoords block_coords{
				(block_coord_t)(((coords.x % CHUNK_WIDTH_BLOCKS) + CHUNK_WIDTH_BLOCKS) % CHUNK_WIDTH_BLOCKS),
				(block_coord_t)coords.y,
				(block_coord_t)(((coords.z % CHUNK_WIDTH_BLOCKS) + CHUNK_WIDTH_BLOCKS) % CHUNK_WIDTH_BLOCKS)
			};
			return { chunk_coords, block_coords };
		}
	private:
		std::unordered_map<ChunkCoords, Chunk> chunk_map;
		std::unique_ptr<WorldGenerator> world_gen;
		DayLightCycle day_light_cycle;
		const std::shared_ptr<const Context> user_context;
		const std::shared_ptr<BlockRegister> block_register;

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