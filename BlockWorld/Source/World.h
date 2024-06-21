#pragma once

#include "Debug.h"
#include "Chunk.h"
#include "Camera.h"

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

		void update(const bwrenderer::Camera& camera, const BlockRegister& blocks) // todo created shared instance of camera/player
		{
			for (auto it = chunkMap.begin(); it != chunkMap.end(); )
			{
				if (abs(it->first.x - camera.position.x / 15) >  16 || abs(it->first.z - camera.position.z / 15) > 16)
					it = unloadChunk(it);
				else it++;
			}
			
			ChunkCoords coords{};
			for (coords.x = camera.position.x/15 - 8; coords.x <= camera.position.x/15 + 8; coords.x++)
				for (coords.z = camera.position.z/15 - 8; coords.z <= camera.position.z/15 + 8; coords.z++)
					loadChunk(coords, blocks);

			for (auto& [coords, chunk] : chunkMap)
			{
				chunk.update();
			}
		}

		void render(bwrenderer::RenderContext& context) const
		{
			for (auto& [coords, chunk] : chunkMap)
			{
				chunk.render(context);
			}
		}

		const std::unordered_map<ChunkCoords,Chunk>::iterator loadChunk(ChunkCoords coords, const BlockRegister& blocks)
		{
			if (const auto& It = chunkMap.find(coords); It != chunkMap.end()) return It;

			const auto& [Iterator, success] = chunkMap.emplace(coords, coords);
			auto& chunk = Iterator->second;

			BW_ASSERT(success, "Loading chunk error.");

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
							if (y < 55) chunk.setBlock({ x, (uint8_t)y, z }, blocks.stone);
							if (y < 59 && y >= 55) chunk.setBlock({ x, (uint8_t)y, z }, blocks.dirt);
							if (y >= 59) chunk.setBlock({ x, (uint8_t)y, z }, blocks.grass);
						}
					}
				}
			}

			BW_INFO("Chunk { %i, %i } loaded.", coords.x, coords.z);
			return Iterator;
		}

		void unloadChunk(const ChunkCoords& coords)
		{
			//storeChunk();
			chunkMap.erase(coords);
			BW_INFO("Chunk { %i, %i } unloaded.", coords.x, coords.z);
		}

		const std::unordered_map<ChunkCoords, Chunk>::iterator unloadChunk(const std::unordered_map<ChunkCoords, Chunk>::iterator& it)
		{
			//storeChunk(it);
			return chunkMap.erase(it);
		}

		void storeChunk()
		{
			// todo add storage
		}

	private:
		std::unordered_map<ChunkCoords, Chunk> chunkMap;
		// todo implement capability to have a chunk loaded in RAM even if is not being updated.
	private:
		void unloadAllChunks()
		{
			chunkMap.clear();
			BW_INFO("All chunks unloaded.");
		}
	};




}