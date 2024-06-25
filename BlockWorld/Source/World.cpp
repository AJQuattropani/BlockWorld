#include "World.h"


namespace bwgame {

	void World::build_func(ChunkCoords coords, Chunk* chunk) {
		worldGen->buildChunk(coords, *chunk);
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

}