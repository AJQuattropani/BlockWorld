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

__m256i to_m256i(const vec_type256& vec);
vec_type256 to_vec256(const __m256i& _mmchunk);

__m256i _bitnot_256(const __m256i& _mmchunk);

/*
Trailing zeros in each 64-bit space.
*/
__m256i _trailing_zeros_64(const __m256i& _mmchunk);

/*
Left/Right shifts 256 intrinsic type as a 16x16 array of bits, or 16-array of uint_16t
1 AVX IN, 1 AVX OUT
*/
__m256i _bitshiftv_right256_16(const __m256i& _mmchunk, uint8_t count);
__m256i _bitshiftv_left256_16(const __m256i& _mmchunk, uint8_t count);

/*
Enables full left/right bitshift of 256 intrinsic type by 0-63 bits.
*/
__m256i _bitshiftv_right256_256(__m256i _mmchunk, uint8_t count);
__m256i _bitshiftv_left256_256(__m256i _mmchunk, uint8_t count);

/*
Enables full left/right bitshift of 256 intrinsic type by 1 bit.
*/
__m256i _bitshift1_right256_256(__m256i _mmchunk);
__m256i _bitshift1_left256_256(__m256i _mmchunk);

/*
Returns a 256 intrinsic filtered so the only set bits have no 0 to its right.
Treated here as a 16x16.
*/
__m256i _right_face_bits256_16(__m256i _mmchunk);

/*
Returns a 256 intrinsic filtered so the only set bits have no 0 to its left.
Treated here as a 16x16.
*/
__m256i _left_face_bits256_16(__m256i _mmchunk);

/*
Returns a 256 intrinsic filtered so the only set bits have no 0 to its right.
*/
__m256i _right_face_bits256_256(__m256i _mmchunk);

/*
Returns a 256 intrinsic filtered so the only set bits have no 0 to its left.
*/
__m256i _left_face_bits256_256(__m256i _mmchunk);






