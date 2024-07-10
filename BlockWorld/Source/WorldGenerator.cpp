#include "WorldGenerator.h"

namespace bwgame {



	void WorldGenerator::buildChunk(const ChunkCoords& coords, Chunk& chunk) const
	{
		srand(coords.seed);
		
		chunk.reserve(80 * CHUNK_WIDTH_BLOCKS * CHUNK_WIDTH_BLOCKS);

		BlockCoords blockIdx{0,0,0};
		while (1)
		{
			for (blockIdx.z = 0; blockIdx.z < CHUNK_WIDTH_BLOCKS; blockIdx.z++)
			{
				for (blockIdx.x = 0; blockIdx.x < CHUNK_WIDTH_BLOCKS; blockIdx.x++)
				{
					int64_t w_x = coords.x * 15 + blockIdx.x;
					int64_t w_z = coords.z * 15 + blockIdx.z;
					int64_t threshold = 80.0 + ((rand() % 10 == 0) ? 1 : 0);  //1.0 * cos(w_x / 16.0) + 2.0 * sin(w_z/40.0) + 6.0 * sin(w_z/8.0-sin(w_x));
					if (blockIdx.y < threshold)
						chunk.setBlock(blockIdx, getBlockLayered(threshold - blockIdx.y, blockIdx.y));
				}
			}

			if (blockIdx.y == CHUNK_HEIGHT_BLOCKS - 1) break;
			blockIdx.y++;
		}
	}

	inline const Block& WorldGenerator::getBlockLayered(int64_t depth, uint8_t height) const
	{
		if (height < 70) {
			if (depth < 5) return rand() % 2 ? blocks->sand : blocks->gravel;
		}
		if (depth > 5) return blocks->stone;
		if (depth > 1) return blocks->dirt;
		return blocks->grass;
		}

}
