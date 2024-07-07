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
	};

	/*
	* The goal is to eventually have file writing/reading.
	*/
	class World
	{
	public:
		World(const std::shared_ptr<BlockRegister>& block_register, const std::shared_ptr<bwrenderer::RenderContext>& context, 
			float ticks_per_second,
			float minutes_per_day = 1.0, uint64_t seed = 0);

		~World();

		void update();

		void render();

		void storeChunk();

	private:
		std::unordered_map<ChunkCoords, Chunk> chunkMap;
		std::unique_ptr<WorldGenerator> worldGen;
		std::shared_ptr<bwrenderer::RenderContext> context;
		bwrenderer::Shader blockShader, shadowShader;
		DayLightCycle dayLightCycle;
		bwrenderer::SkyBox skybox;
		bwrenderer::frame_buffer depth_buffer;
		bwrenderer::TextureBuffer depth_map;
		const int SHADOW_WIDTH = 720, SHADOW_HEIGHT = 720;

	private:
		void mt_loadChunks();
		void loadChunks();
		void build_func(ChunkCoords coords, Chunk* chunk);
		void loadChunk(ChunkCoords coords);
		
		
		void unloadChunk(const ChunkCoords& coords);

		const std::unordered_map<ChunkCoords, Chunk>::iterator unloadChunk(const std::unordered_map<ChunkCoords, Chunk>::iterator& it);

		void unloadAllChunks();
	};




}