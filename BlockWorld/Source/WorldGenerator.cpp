#include "WorldGenerator.hpp"

namespace bwgame {



	[[nodiscard]] Chunk WorldGenerator::buildChunk(ChunkCoords coords, std::unordered_map<ChunkCoords, Chunk> const* chunkMap) const
	{
		Chunk chunk(coords, chunkMap);

		srand(static_cast<unsigned int>(coords.seed));
		
		chunk.reserve(60 * CHUNK_WIDTH_BLOCKS * CHUNK_WIDTH_BLOCKS);

		BlockCoords blockIdx{0,0,0};
			for (blockIdx.z = 0; blockIdx.z < CHUNK_WIDTH_BLOCKS; blockIdx.z++)
			{
				for (blockIdx.x = 0; blockIdx.x < CHUNK_WIDTH_BLOCKS; blockIdx.x++)
				{
					int64_t w_x = static_cast<int64_t>(coords.x) * 15 + blockIdx.x;
					int64_t w_z = static_cast<int64_t>(coords.z) * 15 + blockIdx.z;
					//int64_t threshold = 60;  //1.0 * cos(w_x / 16.0) + 2.0 * sin(w_z/40.0) + 6.0 * sin(w_z/8.0-sin(w_x));
					int64_t threshold = 80;

					threshold += 4.0f * world_noise_gen.sample2D(static_cast<float>(w_x) / 12.0f, static_cast<float>(w_z) / 12.0f);
					threshold += 8.0f * world_noise_gen.sample2D(static_cast<float>(w_x) / 80.0f, static_cast<float>(w_z) / 80.0f);
					threshold += 20.0f * world_noise_gen.sample2D(static_cast<float>(w_x) / 120.0f, static_cast<float>(w_z) / 120.0f);
					
					float biome = world_noise_gen.sample2D(static_cast<float>(w_x) / 300.0f, static_cast<float>(w_z) / 300.0f)
						* world_noise_gen.sample2D(static_cast<float>(w_x) / 651.0f, static_cast<float>(w_z) / 651.0f);

					blockIdx.y = 0;
					while (1)
					{
					if (blockIdx.y < threshold)
						chunk.setBlock(blockIdx, getBlockLayered(threshold - blockIdx.y, blockIdx.y, biome));
					if (blockIdx.y == CHUNK_HEIGHT_BLOCKS - 1) break;
					blockIdx.y++;
					}
				}
			}


		return chunk;
	}

	[[nodiscard]] inline const Block& WorldGenerator::getBlockLayered(int64_t depth, uint8_t height, float biome) const
	{
		if (depth > 5) return blocks->stone;
		if (biome > 0.3 && height < 100)
		{
			return blocks->sand;
		}
		if (depth > 1) return blocks->dirt;
		return blocks->grass;
	
	}

}
