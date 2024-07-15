#include "ExtendLong.hpp"

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
