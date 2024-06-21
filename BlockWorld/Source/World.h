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

		void update()
		{
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

		void loadChunk(ChunkCoords coords, const BlockRegister& blocks)
		{
			// when you left off, you were emplacing the Chunks. however, the current Chunk generation requires the Chunk construct
			// to have a reference to BlockRegistry. You need to unroute the generation function from the Chunk constructor.
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
						if (rand() % 1000 == 0 || y < 60 + 5 * cos((15 * coords.x + x) / 7.5) * sin((15 * coords.z + z) / 7.5))
						{
							if (y < 55) chunk.setBlock({ x, (uint8_t)y, z }, blocks.stone);
							if (y < 59 && y >= 55) chunk.setBlock({ x, (uint8_t)y, z }, blocks.dirt);
							if (y >= 59) chunk.setBlock({ x, (uint8_t)y, z }, blocks.grass);
						}
					}
				}
			}

			BW_INFO("Chunk { %i, %i } loaded.", coords.x, coords.z);
		}

		void unloadChunk(ChunkCoords coords)
		{
			chunkMap.erase(coords);
			BW_INFO("Chunk { %i, %i } unloaded.", coords.x, coords.z);
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