#pragma once
#include "ExtendLong.hpp"

// Note: there are 16 AVX registers.

namespace utils
{
	using data_JIK = vec_type256[16][16];
	using data_IKJ = vec_type256[256];

	struct BinaryChunk
	{
		data_JIK p_yxz;	// i = x, j = y, k = z
		data_IKJ p_xzy;	// i = x, j = y, k = z
		data_IKJ p_zxy;	// i = z, j = y, k = x

		data_JIK n_yxz;	// i = x, j = y, k = z
		data_IKJ n_xzy;	// i = x, j = y, k = z
		data_IKJ n_zxy;	// i = z, j = y, k = x
	};

	/*
	Sets the bit in the 3D bit array.
	*/
	inline void set(data_JIK& jik, uint8_t i, uint8_t j, uint8_t k)
	{
		jik[k][i].v4i_u64[j / 64] |= 1ULL << (63 - (j % 64));
	}
	inline void set(data_IKJ& ikj, uint8_t i, uint8_t j, uint8_t k)
	{
		ikj[j].v16i_u16[k] |= 1UL << i;
	}

	/*
	Transforms direction-based face data for each binary array in binary chunk through one iteration.
	*/
	inline void bc_face_bits_inplace(BinaryChunk& binary_chunk)
	{
		for (uint16_t u = 0; u < 256; u++)
		{
			(binary_chunk.p_yxz)[u / 16][u % 16] = to_vec256(_right_face_bits256_256(to_m256i((binary_chunk.p_yxz)[u / 16][u % 16])));
			(binary_chunk.p_zxy)[u] = to_vec256(_right_face_bits256_16(to_m256i((binary_chunk.p_zxy)[u])));
			(binary_chunk.p_xzy)[u] = to_vec256(_right_face_bits256_16(to_m256i((binary_chunk.p_xzy)[u])));

			(binary_chunk.n_yxz)[u / 16][u % 16] = to_vec256(_left_face_bits256_256(to_m256i((binary_chunk.n_yxz)[u / 16][u % 16])));
			(binary_chunk.n_zxy)[u] = to_vec256(_left_face_bits256_16(to_m256i((binary_chunk.n_zxy)[u])));
			(binary_chunk.n_xzy)[u] = to_vec256(_left_face_bits256_16(to_m256i((binary_chunk.n_xzy)[u])));
		}
	}
}