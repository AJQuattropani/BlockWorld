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
			float minutes_per_day = 1.0, uint64_t seed = 0);

		~World();

		void update();

		void storeChunk();

	private:
		std::unordered_map<ChunkCoords, Chunk> chunk_map;
		std::unique_ptr<WorldGenerator> world_gen;
		DayLightCycle day_light_cycle;
		const std::shared_ptr<UserContext> user_context;

		chunk_coord_t player_last_chunk_pos_x = 0, player_last_chunk_pos_z = 0;

		utils::ThreadList async_world_operations; // number of threads may be subject to change based on CPU
		std::mutex world_data_lock;

	private:
		void mt_loadChunks();
		void loadChunk(ChunkCoords coords);
		
		
		void unloadChunk(const ChunkCoords& coords);

		void unloadAllChunks();



		friend bwrenderer::WorldRenderer;
	};




}