#include "Chunkset.h"

using namespace bitset;

void bitset_16::set(size_t n, bool state) {
	BW_ASSERT(n < 16, "Bit out of range.");
	bits = bits | state << n;
}

bool bitset_16::test(size_t n) {
	BW_ASSERT(n < 16, "Bit out of range.");
	return (bits >> n) & 1U;
}

bool bitset_16::operator[] (size_t n) {
	BW_ASSERT(n < 16, "Bit out of range.");
	return (bits >> n) & 1U;
}

bitset_16& bitset_16::operator<<= (size_t n) {
	bits <<= n;
	return *this;
}

void ChunkSet::flipTo(bool state, in_chunk_coords coords) {
	// data is aligned in x,z,y order
	// NOTE log_2 16 = 4, log_2 256 = 8
	// location is the ith bit aligned in the bitset
	// explicit casting is to ensure bitwise operations go through coorectly.
	BW_ASSERT(coords.x < 16 & coords.y < 256 & coords.z < 16, "Bit out of range.");

	uint16_t location = (((ch_chunkspace_size_t)coords.x) + ((ch_chunkspace_size_t)coords.y << 4) + ((ch_chunkspace_size_t)coords.z << (4 + 4)));
	chunk_space.data.set(location, state);
}

// left-right aligned data
bitset_space ChunkSet::getDataXZY() const {
	return chunk_space;
}

// forward-backward aligned data
bitset_space ChunkSet::getDataZXY() const {
	bitset_space columnwise_data{};

	ch_chunkspace_size_t i, j, k;
	for (ch_chunkspace_size_t index = 0; ; index++)
	{
		j = index / (1 << 8);
		k = (index % (1 << 8)) / (1 << 4);
		i = index % (1 << 4);

		// NOTE might be faster to OR since i,j,k are independent
		columnwise_data.data.set((j << 8) + (i << 4) + k, chunk_space.data.test(index));

		if (index == (1 << 16) - 1) break;
	}
	return columnwise_data;
}

// up-down aligned data
bitset_space ChunkSet::getDataYXZ() const {
	bitset_space vertical_data{};

	ch_chunkspace_size_t i, j, k;
	for (ch_chunkspace_size_t index = 0; ; index++)
	{
		j = index / (1 << 8);
		k = (index % (1 << 8)) / (1 << 4);
		i = index % (1 << 4);

		vertical_data.data.set((k << 12) + (i << 8) + j, chunk_space.data.test(index));

		if (index == (1 << 16) - 1) break;
	}
	return vertical_data;
}

/*
* 
* Returns data on stack. Use with caution.
*/
[[deprecated("Using heap data to prevent stack overflow.")]]
std::tuple<bitset_space, bitset_space, bitset_space> ChunkSet::getAlignedData() const {
	bitset_space vertical_data{}, columnwise_data{};
	ch_chunkspace_size_t i, j, k;
	for (ch_chunkspace_size_t index = 0; ; index++)
	{
		j = index / (1 << 8);
		k = (index % (1 << 8)) / (1 << 4);
		i = index % (1 << 4);

		columnwise_data.data.set((j << 8) + (i << 4) + k, chunk_space.data.test(index));
		vertical_data.data.set((k << 12) + (i << 8) + j, chunk_space.data.test(index));

		if (index == (1 << 16) - 1) break;
	}
	return { chunk_space, columnwise_data, vertical_data };
}

/*
* Returns data as pointers on heap. User responsible for deletion.
*/
std::tuple<bitset_space*, bitset_space*, bitset_space*> bitset::ChunkSet::getAlignedDataHeap() const
{
	bitset_space* normal_data = new bitset_space{chunk_space};
	bitset_space* vertical_data = new bitset_space{};
	bitset_space* columnwise_data = new bitset_space{};
	ch_chunkspace_size_t i, j, k;
	for (ch_chunkspace_size_t index = 0; ; index++)
	{
		j = index / (1 << 8);
		k = (index % (1 << 8)) / (1 << 4);
		i = index % (1 << 4);

		columnwise_data->data.set((j << 8) + (i << 4) + k, chunk_space.data.test(index));
		vertical_data->data.set((k << 12) + (i << 8) + j, chunk_space.data.test(index));

		if (index == (1 << 16) - 1) break;
	}
	return { normal_data, columnwise_data, vertical_data };
}

std::tuple<bitset_space*, bitset_space*, bitset_space*, bitset_space*, bitset_space*, bitset_space*> getExposedFaceMap(const ChunkSet& chunk)
{
	auto [p_xzy, p_zxy, p_yxz] = chunk.getAlignedDataHeap();

	bitset_space* n_xzy = new bitset_space(*p_xzy);
	bitset_space* n_zxy = new bitset_space(*p_zxy);
	bitset_space* n_yxz = new bitset_space(*p_yxz);

	for (ch_blockcoord_t j = 0; ; j++)
	{
		for (ch_blockcoord_t k = 0; k < 16; k++)
		{
			(p_xzy->aligned)[j][k].bits &= ~((p_xzy->aligned)[j][k].bits >> 1);
			(p_zxy->aligned)[j][k].bits &= ~((p_zxy->aligned)[j][k].bits >> 1);
			(n_xzy->aligned)[j][k].bits &= ~((n_xzy->aligned)[j][k].bits << 1);
			(n_zxy->aligned)[j][k].bits &= ~((n_zxy->aligned)[j][k].bits << 1);
		}
		if (j >= ((1 << 8) - 1)) break;
	}

	for (ch_blockcoord_t i = 0; i < 16; i++)
	{
		for (ch_blockcoord_t k = 0; k < 16; k++)
		{
			(p_yxz->vertical)[i][k] &= ~((p_yxz->vertical)[i][k] >> 1);
			(n_yxz->vertical)[i][k] &= ~((n_yxz->vertical)[i][k] << 1);
		}
	}

	return { p_xzy, p_zxy, p_yxz, n_xzy, n_zxy, n_yxz };
}