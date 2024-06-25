#include "WorldGenerator.h"

namespace bwgame {



	void WorldGenerator::buildChunk(const ChunkCoords& coords, Chunk& chunk) const
	{
		BlockCoords blockIdx{0,0,0};
		auto& x = blockIdx.x;
		auto& y = blockIdx.y;
		auto& z = blockIdx.z;
		while (1)
		{
			for (z = 0; z < 15; z++)
			{
				for (x = 0; x < 15; x++)
				{
					int64_t w_x = coords.x * 15 + x;
					int64_t w_z = coords.z * 15 + z;
					int64_t threshold = 100.0 + 5.0 * sin(w_x / 16.0);
					if (y < threshold)
						chunk.setBlock(blockIdx, getBlockLayered(threshold - y));
				}
			}

			if (y == 255) break;
			y++;
		}
	}

	inline const Block& WorldGenerator::getBlockLayered(int64_t depth) const
	{
		if (depth > 5) return blocks->stone;
		if (depth > 1) return blocks->dirt;
		return blocks->grass;
	}

}
