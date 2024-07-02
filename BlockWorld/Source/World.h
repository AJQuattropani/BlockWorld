#pragma once

#include "Debug.h"
#include "Timer.h"
#include "Chunk.h"
#include "Camera.h"
#include "WorldGenerator.h"
#include "WorldRenderer.h"
#include "DayCycle.h"

#include <unordered_map>
#include <future>
#include <thread>

namespace bwgame {


	struct WorldLoadData
	{
		chunk_diff_t ch_render_load_distance;
		chunk_diff_t ch_render_unload_distance;
	};

	/*
	* The goal is to eventually have file writing/reading.
	*/
	class World
	{
	public:
		World(const std::shared_ptr<BlockRegister>& block_register, float ticks_per_second, float minutes_per_day = 1.0, WorldLoadData data
			= { .ch_render_load_distance = 24, .ch_render_unload_distance = 24 }, uint64_t seed = 0);

		~World();

		void update(const bwrenderer::Camera& camera);

		void render(bwrenderer::RenderContext& context);

		void storeChunk();

	private:
		std::unordered_map<ChunkCoords, Chunk> chunkMap;
		std::unique_ptr<WorldGenerator> worldGen;
		bwrenderer::Shader blockShader;
		WorldLoadData worldLoadData;
		DayLightCycle dayLightCycle;
		bwrenderer::SkyBox skybox;

	private:
		void mt_loadChunks(const bwrenderer::Camera& camera);
		void loadChunks(const bwrenderer::Camera& camera);
		void build_func(ChunkCoords coords, Chunk* chunk);
		void loadChunk(ChunkCoords coords);
		
		
		void unloadChunk(const ChunkCoords& coords);

		const std::unordered_map<ChunkCoords, Chunk>::iterator unloadChunk(const std::unordered_map<ChunkCoords, Chunk>::iterator& it);

		void unloadAllChunks();
	};




}