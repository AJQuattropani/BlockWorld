#include "BinaryChunk.h"

namespace utils {

	void set(data_JIK& jik, uint8_t i, uint8_t j, uint8_t k)
	{
		jik[k][i].v4i_u64[j / 64] |= 1ULL << (63 - (j % 64));
	}

	void set(data_IKJ& ikj, uint8_t i, uint8_t j, uint8_t k)
	{
		ikj[j].v16i_u16[k] |= 1UL << i;
	}

	void bc_face_bits_inplace(BinaryChunk& binary_chunk)
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