#include "ExtendLong.h"

void _print_extlonghex(const vec_type256* data)
{
	printf("%llX %llX %llX %llX \n", data->v4i_u64[3], data->v4i_u64[2], data->v4i_u64[1], data->v4i_u64[0]);
}

void _print_extlongbin(const vec_type256* data)
{
	for (uint8_t i = 0; i < 4; i++)
	{
		for (uint8_t j = 0; j < 64; j++)
		{
			printf("%llu", data->v4i_u64[4 - 1 - i] >> (64 - 1 - j) & 1ULL);
		}
	}
	printf("\n");
}

template<typename T>
void _printbin(const T* data)
{
	constexpr uint8_t bitsize = sizeof(T) * 8;
	constexpr uint8_t divisions = sizeof(vec_type256) / sizeof(T);
	for (uint8_t i = 0; i < divisions; i++)
	{
		for (uint8_t j = 0; j < bitsize; j++)
		{
			printf("%llu", data[divisions - 1 - i] >> (bitsize - 1 - j) & 1ULL);
		}
		printf("\n");
	}
	printf("\n");
}

template<>
void _printbin(const vec_type256* data)
{
	constexpr uint8_t bitsize = 64;
	constexpr uint8_t divisions = 4;
	for (uint8_t i = 0; i < divisions; i++)
	{
		for (uint8_t j = 0; j < bitsize; j++)
		{
			printf("%llu", data->v4i_u64[i] >> (bitsize - 1 - j) & 1ULL);
		}
	}
	printf("\n");
}

__m256i to_m256i(const vec_type256& vec) { return _mm256_loadu_epi64(&vec.v4i_i64); }

vec_type256 to_vec256(const __m256i& _mmchunk) { return *(vec_type256*)(&_mmchunk); }

__m256i _bitnot_256(const __m256i& _mmchunk) { return _mm256_xor_si256(_mmchunk, _mm256_cmpeq_epi32(_mmchunk, _mmchunk)); }

__m256i _trailing_zeros_64(const __m256i& _mmchunk) { return _mm256_lzcnt_epi64(_mmchunk); }

__m256i _bitshiftv_right256_16(const __m256i& _mmchunk, uint8_t count) { return _mm256_srli_epi16(_mmchunk, count); }

__m256i _bitshiftv_left256_16(const __m256i& _mmchunk, uint8_t count) { return _mm256_slli_epi16(_mmchunk, count); }

__m256i _bitshiftv_right256_256(__m256i _mmchunk, uint8_t count)
{
	//Copy chunk
	__m256i _map = _mm256_permute4x64_epi64(_mmchunk, _MM_SHUFFLE(2, 1, 0, 3));	//Shifts right by 64 bytes
	constexpr uint64_t mask = ~0ULL;
	_map = _mm256_and_epi32(_map, _mm256_setr_epi64x(0, mask, mask, mask)); //Extracts the 64th bit except for new bit location
	_map = _mm256_slli_epi64(_map, 64 - count);	//Shifts the bit left to the 1th place

	_mmchunk = _mm256_srli_epi64(_mmchunk, count);	//Bitshifts chunk right by 1
	//Adds missing bits to complete overall bitshift
	_mmchunk = _mm256_or_epi32(_mmchunk, _map);
	return _mmchunk;
}

__m256i _bitshiftv_left256_256(__m256i _mmchunk, uint8_t count)
{
	//Copy chunk
	__m256i _map = _map = _mm256_permute4x64_epi64(_mmchunk, _MM_SHUFFLE(0, 3, 2, 1)); //Shifts left by 64 bytes
	constexpr uint64_t mask = ~0ULL;
	_map = _mm256_and_epi32(_map, _mm256_setr_epi64x(mask, mask, mask, 0));	//Extracts the 64th bit except for new bit location
	_map = _mm256_srli_epi64(_map, 64 - count);	//Shifts the bit left to the 1th place

	_mmchunk = _mm256_slli_epi64(_mmchunk, count);	//Bitshifts chunk left by 1	
	//Adds missing bits to complete overall bitshift
	return _mm256_or_epi32(_mmchunk, _map);
}

__m256i _bitshift1_right256_256(__m256i _mmchunk)
{
	//Copy chunk
	__m256i _map = _mm256_permute4x64_epi64(_mmchunk, _MM_SHUFFLE(2, 1, 0, 3));	//Shifts right by 64 bytes
	constexpr uint64_t mask = ~0ULL;
	_map = _mm256_and_epi32(_map, _mm256_setr_epi64x(0, mask, mask, mask)); //Extracts the 64th bit except for new bit location
	_map = _mm256_slli_epi64(_map, 63);	//Shifts the bit left to the 1th place

	_mmchunk = _mm256_srli_epi64(_mmchunk, 1);	//Bitshifts chunk right by 1
	//Adds missing bits to complete overall bitshift
	return _mm256_or_epi32(_mmchunk, _map);
}

__m256i _bitshift1_left256_256(__m256i _mmchunk)
{
	//Copy chunk
	__m256i _map = _mm256_permute4x64_epi64(_mmchunk, _MM_SHUFFLE(0, 3, 2, 1)); //Shifts left by 64 bytes
	constexpr uint64_t mask = ~0ULL;
	_map = _mm256_and_epi32(_map, _mm256_setr_epi64x(mask, mask, mask, 0));	//Extracts the 64th bit except for new bit location
	_map = _mm256_srli_epi64(_map, 63);	//Shifts the bit left to the 1th place

	_mmchunk = _mm256_slli_epi64(_mmchunk, 1);	//Bitshifts chunk left by 1
	//Adds missing bits to complete overall bitshift
	return _mm256_or_epi32(_mmchunk, _map);
}

__m256i _right_face_bits256_16(__m256i _mmchunk)
{
	__m256i _andmap = _bitshiftv_left256_16(_mmchunk, 1); // shift
	_andmap = _bitnot_256(_andmap); // not
	return _mm256_and_epi32(_mmchunk, _andmap); // and
}

__m256i _left_face_bits256_16(__m256i _mmchunk)
{
	__m256i _andmap = _bitshiftv_right256_16(_mmchunk, 1); // shift
	_andmap = _bitnot_256(_andmap); // not
	return _mm256_and_epi32(_mmchunk, _andmap); // and
}

__m256i _right_face_bits256_256(__m256i _mmchunk)
{
	__m256i _andmap = _bitshift1_left256_256(_mmchunk); // shift
	_andmap = _bitnot_256(_andmap); // not
	return _mm256_and_epi32(_mmchunk, _andmap); // and
}

__m256i _left_face_bits256_256(__m256i _mmchunk)
{
	__m256i _andmap = _bitshift1_right256_256(_mmchunk); // shift
	_andmap = _bitnot_256(_andmap); // not
	return _mm256_and_epi32(_mmchunk, _andmap); // and
}