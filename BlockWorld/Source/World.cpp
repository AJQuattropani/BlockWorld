#include "World.h"


namespace bwgame {

	void World::build_func(ChunkCoords coords, Chunk* chunk) {
		srand(coords.seed);

		for (uint16_t y = 0; y < 256; y++)
		{
			for (uint8_t z = 0; z < 15; z++)
			{
				for (uint8_t x = 0; x < 15; x++)
				{
					if (y < 60 + 5 * cos((15 * coords.x + x) / 7.5) * sin((15 * coords.z + z) / 7.5))
					{
						if (y < 55) chunk->setBlock({ x, (uint8_t)y, z }, blocks->stone);
						if (y < 59 && y >= 55) chunk->setBlock({ x, (uint8_t)y, z }, blocks->dirt);
						if (y >= 59) chunk->setBlock({ x, (uint8_t)y, z }, blocks->grass);
					}
				}
			}
		}
	}

	void World::update(const bwrenderer::Camera& camera) // todo created shared instance of camera/player
	{
		mt_loadChunks(camera);

		{
			TIME_FUNC("World Unload and Update");
			for (auto it = chunkMap.begin(); it != chunkMap.end(); )
			{
				if (abs(it->first.x - camera.position.x / 15) > worldLoadData.ch_render_unload_distance / 2
					|| abs(it->first.z - camera.position.z / 15) > worldLoadData.ch_render_unload_distance / 2)
					it = unloadChunk(it);
				else {
					it->second.update();
					it++;
				}
			}
		}
	}

	void World::render(bwrenderer::RenderContext& context) const
	{
		TIME_FUNC("World Render");
		for (auto& [coords, chunk] : chunkMap)
		{
			chunk.render(context);
		}
	}
	
	void World::mt_loadChunks(const bwrenderer::Camera& camera)
	{
		TIME_FUNC("MT World Update and Load");

		ChunkCoords coords{};
		std::vector<std::jthread> async_chunk_loads;

		for (coords.x = camera.position.x/15 - worldLoadData.ch_render_load_distance / 2; 
			coords.x <= camera.position.x/15 + worldLoadData.ch_render_load_distance / 2;
			coords.x++)
		{
			for (coords.z = camera.position.z/15 - worldLoadData.ch_render_load_distance / 2;
				coords.z <= camera.position.z/15 + worldLoadData.ch_render_load_distance / 2;
				coords.z++)
			{

				if (const auto& It = chunkMap.find(coords); It != chunkMap.end()) continue;

				const auto& [Iterator, success] = chunkMap.emplace(coords, coords);
				auto& chunk = Iterator->second;
				//todo move to own function
				srand(coords.seed);
				async_chunk_loads.push_back(std::jthread(&World::build_func, this, coords, &chunk));
			}
		}
		
	}

	void World::loadChunks(const bwrenderer::Camera& camera)
	{
		TIME_FUNC("World Update and Load");

		ChunkCoords coords{};
		std::vector<std::jthread> async_chunk_loads;

		for (coords.x = camera.position.x/15 - worldLoadData.ch_render_load_distance / 2;
			coords.x <= camera.position.x/15 + worldLoadData.ch_render_load_distance / 2;
			coords.x++)
			for (coords.z = camera.position.z/15 - worldLoadData.ch_render_load_distance / 2;
				coords.z <= camera.position.z/15 + worldLoadData.ch_render_load_distance / 2;
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
		srand(coords.seed);

		for (uint16_t y = 0; y < 256; y++)
		{
			for (uint8_t z = 0; z < 15; z++)
			{
				for (uint8_t x = 0; x < 15; x++)
				{
					if (y < 60 + 5 * cos((15 * coords.x + x) / 7.5) * sin((15 * coords.z + z) / 7.5))
					{
						if (y < 55) chunk.setBlock({ x, (uint8_t)y, z }, blocks->stone);
						if (y < 59 && y >= 55) chunk.setBlock({ x, (uint8_t)y, z }, blocks->dirt);
						if (y >= 59) chunk.setBlock({ x, (uint8_t)y, z }, blocks->grass);
					}
				}
			}
		}


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

}