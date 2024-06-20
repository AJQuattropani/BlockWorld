#pragma once

#include "Debug.h"
#include "Chunk.h"

#include <unordered_map>

namespace bwgame {

	/*
	* The goal is to eventually have file writing/reading.
	*/
	class World
	{
	public:
		World()
		{
			BW_INFO("World created.");
		}

		~World()
		{
			unloadAllChunks();
			BW_INFO("World destroyed.");
		}

		void loadChunk(ChunkCoords coords)
		{
			// when you left off, you were emplacing the Chunks. however, the current Chunk generation requires the Chunk construct
			// to have a reference to BlockRegistry. You need to unroute the generation function from the Chunk constructor.
			chunkMap.emplace(coords, coords);

			BW_INFO("Chunk { %i, %i } loaded.", coords.x, coords.z);
		}

		void unloadChunk(ChunkCoords coords)
		{
			chunkMap.erase(coords);
			BW_INFO("Chunk { %i, %i } unloaded.", coords.x, coords.z);
		}


	private:
		std::unordered_map<ChunkCoords, Chunk> chunkMap;
		std::unordered_set<ChunkCoords> updateList; // todo implement capability to have a chunk loaded in RAM even if is not being updated.
	private:
		void unloadAllChunks()
		{
			chunkMap.clear();
			BW_INFO("All chunks unloaded.");
		}
	};




}