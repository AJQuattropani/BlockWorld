#pragma once
#include "ExtendLong.h"

#include <random>

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



	/*
	Converts binary array so that any logical-1 bits only remain set if
	the bit in front of it in the direction is 0.
	*//*
	void bc_rightface_bits_jik_inplace(data_JIK& jik);
	void bc_leftface_bits_jik_inplace(data_JIK& jik);
	void bc_rightface_bits_ikj_inplace(data_IKJ& ikj);
	void bc_leftface_bits_ikj_inplace(data_IKJ& ikj);*/

	//
	//void random_fill(data_JIK& jik);
	//void random_fill(data_IKJ& ikj);
	//
	//
	//void bc_rightface_bits_jik_inplace(data_JIK& jik)
	//{
	//	for (uint8_t k = 0; k < 16; k++)
	//	{
	//		for (uint8_t i = 0; i < 16; i++)
	//		{
	//			jik[k][i] = to_vec256(_right_face_bits256_256(to_m256i(jik[k][i])));
	//		}
	//	}
	//}
	//
	//void bc_leftface_bits_jik_inplace(data_JIK& jik)
	//{
	//	for (uint8_t k = 0; k < 16; k++)
	//	{
	//		for (uint8_t i = 0; i < 16; i++)
	//		{
	//			jik[k][i] = to_vec256(_left_face_bits256_256(to_m256i(jik[k][i])));
	//		}
	//	}
	//}
	//
	//void bc_rightface_bits_ikj_inplace(data_IKJ& ikj)
	//{
	//	for (uint16_t j = 0; j < 256; j++)
	//	{
	//		ikj[j] = to_vec256(_right_face_bits256_16(to_m256i(ikj[j])));
	//	}
	//}
	//
	//void bc_leftface_bits_ikj_inplace(data_IKJ& ikj)
	//{
	//	for (uint16_t j = 0; j < 256; j++)
	//	{
	//		ikj[j] = to_vec256(_left_face_bits256_16(to_m256i(ikj[j])));
	//	}
	//}

	//
	//void random_fill(data_JIK& jik)
	//{
	//	for (uint8_t k = 0; k < 16; k++)
	//	{
	//		for (uint8_t i = 0; i < 16; i++)
	//		{
	//			for (uint8_t u = 0; u < 32; u++)
	//			{
	//				jik[k][i].v32i_u8[u] = rand();
	//			}
	//		}
	//	}
	//}
	//
	//void random_fill(data_IKJ& ikj)
	//{
	//	for (uint16_t j = 0; j < 256; j++)
	//	{
	//		for (uint8_t u = 0; u < 32; u++)
	//		{
	//			ikj[j].v32i_u8[u] = rand();
	//		}
	//	}
	//}
};