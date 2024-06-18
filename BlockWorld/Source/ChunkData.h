#pragma once

#include <bit>
#include <bitset>

using chunk_coord_t = int32_t;
using block_coord_t = uint8_t;

constexpr uint16_t CHUNK_WIDTH_BLOCKS = 15;
constexpr uint16_t CHUNK_HEIGHT_BLOCKS = 256;

union ChunkCoords
{
	struct { chunk_coord_t x, z; };
	chunk_coord_t data[2];
};

union BlockCoords
{
	struct {
		block_coord_t x,y,z;
	};
	uint32_t index;
	block_coord_t data[3];
	
};

enum CHUNK_FLAGS
{
	MODEL_UPDATE_FLAG = 0,
	SIZE
};

using ChunkFlags = std::bitset<CHUNK_FLAGS::SIZE>;

namespace std {
	template<>
	struct hash<BlockCoords>
	{
		size_t operator()(const BlockCoords& key) const
		{
			return key.index;
		}
	};

	template<>
	struct equal_to<BlockCoords>
	{
		bool operator()(const BlockCoords& key1, const BlockCoords& key2) const
		{
			return key1.index == key2.index;
		}
	};


}