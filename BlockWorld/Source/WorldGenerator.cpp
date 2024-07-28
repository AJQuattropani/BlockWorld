#include "WorldGenerator.hpp"

namespace bwgame {



	[[nodiscard]] Chunk WorldGenerator::buildChunk(ChunkCoords coords, std::unordered_map<ChunkCoords, Chunk> const* chunkMap) const
	{
		Chunk chunk(coords, chunkMap);

		srand(static_cast<unsigned int>(coords.seed));
		
		chunk.reserve(100 * CHUNK_WIDTH_BLOCKS * CHUNK_WIDTH_BLOCKS);

		BlockCoords blockIdx{0,0,0};
			for (blockIdx.z = 0; blockIdx.z < CHUNK_WIDTH_BLOCKS; blockIdx.z++)
			{
				for (blockIdx.x = 0; blockIdx.x < CHUNK_WIDTH_BLOCKS; blockIdx.x++)
				{
					int64_t w_x = static_cast<int64_t>(coords.x) * 15 + blockIdx.x;
					int64_t w_z = static_cast<int64_t>(coords.z) * 15 + blockIdx.z;
					//int64_t threshold = 60;  //1.0 * cos(w_x / 16.0) + 2.0 * sin(w_z/40.0) + 6.0 * sin(w_z/8.0-sin(w_x));
					float macro_elevation = world_noise_gen.sample2D(static_cast<float>(w_x) / 250.0f, static_cast<float>(w_z) / 250.0f);
					macro_elevation += world_noise_gen.sample2D(static_cast<float>(w_x) / 120.0f, static_cast<float>(w_z) / 120.0f);
					macro_elevation += world_noise_gen.sample2D(static_cast<float>(w_z) / 320.0f, static_cast<float>(w_x) / 320.0f) / 2.0;
					macro_elevation += world_noise_gen.sample2D(static_cast<float>(w_z) / 400.0f, static_cast<float>(w_x) / 400.0f) / 2.0;


					int64_t threshold;
					float threshold_f = 0.0f;

					if (macro_elevation < 0.25)
					{
						threshold_f = 0.15;
					}
					else if (macro_elevation < 0.5)
					{
						float x = macro_elevation - 0.5;
						threshold_f = -2 * (x * x) + 0.275;
					}
					else if (macro_elevation < 0.75)
					{
						threshold_f = 0.25 * macro_elevation + 0.15;
					}
					else
					{
						threshold_f = 0.5 * macro_elevation - 0.0375;
					}


					threshold = 250 * threshold_f;

					threshold += 4.0f * world_noise_gen.sample2D(static_cast<float>(w_x) / 90.0f, static_cast<float>(w_z) / 90.0f);
					//threshold += 8.0f * world_noise_gen.sample2D(static_cast<float>(w_x) / 80.0f, static_cast<float>(w_z) / 80.0f);
					threshold += 15.0f * world_noise_gen.sample2D(static_cast<float>(w_x) / 45.0f, static_cast<float>(w_z) / 45.0f);
					
					float biome = world_noise_gen.sample2D(static_cast<float>(w_x) / 300.0f, static_cast<float>(w_z) / 300.0f)
						* world_noise_gen.sample2D(static_cast<float>(w_x) / 651.0f, static_cast<float>(w_z) / 651.0f);

					blockIdx.y = 0;
					while (blockIdx.y <= std::max<int>(threshold, 60))
					{
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
		if (height <= 60)
		{
			if (depth < 0) return blocks->water;
			if (depth < 5) return blocks->sand;
		}
		else
		{
			if (biome < 0.0f)
			{
				if (depth < 5) return blocks->sand;
			}
			else
			{
				if (depth < 1) return blocks->grass;
				if (depth < 2) return blocks->dirt;
			}
		}
		return blocks->stone;
	}

}
