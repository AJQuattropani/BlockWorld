#pragma once

#include "BlockUtils.h"
#include <bitset>
#include <bit>

using chunk_coord_t = int32_t;

constexpr uint16_t CHUNK_WIDTH_BLOCKS = 15;
constexpr uint16_t CHUNK_HEIGHT_BLOCKS = 256;

union ChunkCoords
{
	struct { chunk_coord_t x, z; };
	chunk_coord_t data[2];
	uint64_t seed;
};

enum CHUNK_FLAGS
{
	MODEL_UPDATE_FLAG = 0,
	SIZE
};

using ChunkFlags = std::bitset<CHUNK_FLAGS::SIZE>;

namespace std {
	template<>
	struct hash<bwgame::BlockCoords>
	{
		size_t operator()(const bwgame::BlockCoords& key) const
		{
			return key.index;
		}
	};

	template<>
	struct equal_to<bwgame::BlockCoords>
	{
		bool operator()(const bwgame::BlockCoords& key1, const bwgame::BlockCoords& key2) const
		{
			return key1.index == key2.index;
		}
	};


}