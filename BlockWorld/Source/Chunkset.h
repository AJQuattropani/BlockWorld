#pragma once

#include "Debug.h"

#include <bitset>

// although data sets are larger than necessary, ch_blockcoord_t should NEVER be equal or larger than 16
using ch_blockcoord_t = uint8_t;
using w_chunkcoord_t = uint16_t;
using ch_chunkspace_size_t = uint16_t;
using w_blockcoord_t = uint64_t;

struct in_chunk_coords {
	ch_blockcoord_t x, y, z;

	bool operator==(const in_chunk_coords& other) const
	{
		return (x == other.x) && (y == other.y) && (z == other.z);
	}
};


struct chunk_world_coords {
	w_chunkcoord_t c_x, c_z;
	bool operator==(const chunk_world_coords& other) const
	{
		return (c_x == other.c_x) && (c_z == other.c_z);
	}
};

namespace std {
	template<>
	struct hash<in_chunk_coords> {
		size_t operator()(const in_chunk_coords& k) const {
			return (size_t(k.y) << 16) + (size_t(k.x) << 8) + k.z;
		}
	};

	template<>
	struct hash<chunk_world_coords> {
		size_t operator()(const chunk_world_coords& k) const {
			return (size_t(k.c_x) << 16) + k.c_z;
		}
	};

}

namespace bitset {

using bitset_data_t = std::bitset<16 * 16 * 256>;

struct bitset_16 {
	uint16_t bits;

	void set(size_t n, bool state);
	bool test(size_t n);
	bool operator[] (size_t n);
	bitset_16& operator<<= (size_t n);
};

using aligned_bitset_t = bitset_16;
using bitset_layer_t = aligned_bitset_t[16];
using bitset_space_t = bitset_layer_t[256];

using aligned_bitset_strip_t = std::bitset<256>;
using bitset_layer_vertical_t = aligned_bitset_strip_t[16];
using bitset_space_vertical_t = bitset_layer_vertical_t[16];

/*
* Union represents the raw data of ChunkSet. This data holds onto the 16x256x16 bit array
* representation of the world chunk. It may be indexed as a single memory-aligned bitset via "data",
* accessed as a 2D array with bitsets of size 16 via "aligned", or accessed with bitsets of size 256 via "vertical".
*/
union bitset_space {
	bitset_data_t data;
	bitset_space_t aligned;
	bitset_space_vertical_t vertical;
};

/*
* Data structure betrays a bit representation of a 16x256x16 world chunk.
* The 3D "array" of bits has positions which correspond to a block/no block in the chunk's map.
* Various transforms are used to shift the ordering of the data in memory for successful bitwise operation during mesh-building.
*/
struct ChunkSet
{
	bitset_space chunk_space;

	void flipTo(bool state, in_chunk_coords coords);
	
	// left-right aligned data
	bitset_space getDataXZY() const;
	// forward-backward aligned data
	bitset_space getDataZXY() const;
	// up-down aligned data
	bitset_space getDataYXZ() const;
	
	// returns all three using only one traversal.
	std::tuple<bitset_space, bitset_space, bitset_space> getAlignedData() const;

	// returns all three using only one traversal w heap. user responsible for cleanup
	std::tuple<bitset_space*, bitset_space*, bitset_space*> getAlignedDataHeap() const;
};

/*
* Returns exposed face maps as pointers on heap. User responsible for deletion.
*/
std::tuple<bitset_space*, bitset_space*, bitset_space*, bitset_space*, bitset_space*, bitset_space*> getExposedFaceMap(const ChunkSet& chunk);


}