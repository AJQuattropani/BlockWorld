#pragma once
#pragma once

#include <stdio.h>
#include <stdint.h>
#include <immintrin.h>

union vec_type256
{
	int8_t      v32i_i8[32];
	int16_t     v16i_i16[16];
	int32_t     v8i_i32[8];
	int64_t     v4i_i64[4];

	uint8_t		v32i_u8[32];
	uint16_t	v16i_u16[16];
	uint32_t	v8i_u32[8];
	uint64_t	v4i_u64[4];
};

void _print_extlonghex(const vec_type256* data);
void _print_extlongbin(const vec_type256* data);

/*
Bits must be read left to right, bottom to top by default.
*/
template<typename T>
void _printbin(const T* data);

template<>
void _printbin(const vec_type256* data);

inline __m256i to_m256i(const vec_type256& vec) { return _mm256_loadu_epi64(&vec.v4i_i64); }
inline vec_type256 to_vec256(const __m256i& _mmchunk) { return *(vec_type256*)(&_mmchunk); }

inline __m256i _bitnot_256(const __m256i& _mmchunk) { return _mm256_xor_si256(_mmchunk, _mm256_cmpeq_epi32(_mmchunk, _mmchunk)); }

/*
Trailing zeros in each 64-bit space.
*/
inline __m256i _trailing_zeros_64(const __m256i& _mmchunk) { return _mm256_lzcnt_epi64(_mmchunk); }

/*
Left/Right shifts 256 intrinsic type as a 16x16 array of bits, or 16-array of uint_16t
*/
inline __m256i _bitshiftv_right256_16(const __m256i& _mmchunk, uint8_t count) { return _mm256_srli_epi16(_mmchunk, count); }
inline __m256i _bitshiftv_left256_16(const __m256i& _mmchunk, uint8_t count) { return _mm256_slli_epi16(_mmchunk, count); }

/*
Enables full left/right bitshift of 256 intrinsic type by 0-63 bits.
*/
inline __m256i _bitshiftv_right256_256(__m256i _mmchunk, uint8_t count)
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
inline __m256i _bitshiftv_left256_256(__m256i _mmchunk, uint8_t count)
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

/*
Enables full left/right bitshift of 256 intrinsic type by 1 bit.
*/
inline __m256i _bitshift1_right256_256(__m256i _mmchunk)
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
inline __m256i _bitshift1_left256_256(__m256i _mmchunk)
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

/*
Returns a 256 intrinsic filtered so the only set bits have no 0 to its right.
Treated here as a 16x16.
*/
inline __m256i _right_face_bits256_16(__m256i _mmchunk)
{
	__m256i _andmap = _bitshiftv_left256_16(_mmchunk, 1); // shift
	_andmap = _bitnot_256(_andmap); // not
	return _mm256_and_epi32(_mmchunk, _andmap); // and
}

/*
Returns a 256 intrinsic filtered so the only set bits have no 0 to its left.
Treated here as a 16x16.
*/
inline __m256i _left_face_bits256_16(__m256i _mmchunk)
{
	__m256i _andmap = _bitshiftv_right256_16(_mmchunk, 1); // shift
	_andmap = _bitnot_256(_andmap); // not
	return _mm256_and_epi32(_mmchunk, _andmap); // and
}

/*
Returns a 256 intrinsic filtered so the only set bits have no 0 to its right.
*/
inline __m256i _right_face_bits256_256(__m256i _mmchunk)
{
	__m256i _andmap = _bitshift1_left256_256(_mmchunk); // shift
	_andmap = _bitnot_256(_andmap); // not
	return _mm256_and_epi32(_mmchunk, _andmap); // and
}

/*
Returns a 256 intrinsic filtered so the only set bits have no 0 to its left.
*/
inline __m256i _left_face_bits256_256(__m256i _mmchunk)
{
	__m256i _andmap = _bitshift1_right256_256(_mmchunk); // shift
	_andmap = _bitnot_256(_andmap); // not
	return _mm256_and_epi32(_mmchunk, _andmap); // and
}





