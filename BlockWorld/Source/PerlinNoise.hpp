#pragma once

#include <cstdint>
#include <random>
#include <glm/gtc/constants.hpp>
#include <glm/trigonometric.hpp>

#include "ChunkData.hpp"

/*
* 
* REMINDER implement parallel policy to chunk loading
* ULL -> float (0-1)
*/

namespace utils {

	constexpr double PI = 3.14159265;
	constexpr double PI2 = PI * PI;
	constexpr double HALF_PI = PI / 2;
	constexpr double TAU = 2 * PI;

	class PerlinNoiseGenerator
	{
	private: 
		//mutable Lehmer64Generator generator;
		uint64_t world_seed = 0;
	public:
		PerlinNoiseGenerator(uint64_t world_seed) : world_seed(world_seed) {}

		// float between 1 and 0
		float sample2D(float x, float z) const
		{
			// TODO add credit for code courtesy of Zipped
			int32_t x0, x1; 
			if (x >= 0.0f)
			{
				x0 = (int32_t)x;
				x1 = x0 + 1;
			}
			else
			{
				x1 = (int32_t)x;
				x0 = x1 - 1;
			}
			int32_t z0, z1;
			if (z >= 0.0f)
			{
				z0 = (int32_t)z;
				z1 = z0 + 1;
			}
			else
			{
				z1 = (int32_t)z;
				z0 = z1 - 1;
			}

			
			// interpolation weights
			float sx = x - (float)x0;
			float sz = z - (float)z0;

			// interpolate top two corners
			float n0 = genGradient(x0, z0, x, z);
			float n1 = genGradient(x1, z0, x, z);
			float ix0 = interpolate(n0, n1, sx);

			n0 = genGradient(x0, z1, x, z);
			n1 = genGradient(x1, z1, x, z);
			float ix1 = interpolate(n0, n1, sx);

			float value = interpolate(ix0, ix1, sz);


			if (value > 1.0f) value = 1.0f;
			if (value < -1.0f) value = -1.0f;

			return value;
		}
	private:
		inline float genGradient(int32_t cx, int32_t cz, float x, float z) const
		{
			auto [g_x, g_z] = randomGradient(cx, cz);

			float dx = x - (float)cx;
			float dz = z - (float)cz;

			//BW_DEBUG("<%f, %f>", g_x, g_z);

			return (g_x * dx + g_z * dz);
		}

		inline std::pair<float, float> randomGradient(int32_t cx, int32_t cz) const
		{
			// FROM ZIPPED

			// No precomputed gradients mean this works for any number of grid coordinates
			const unsigned w = 8 * sizeof(unsigned long long);
			constexpr unsigned long long s = w / 2;
			unsigned long long a = cx, b = cz;
			a *= 3284157443;

			b ^= a << s | a >> w - s;
			b *= 1911520717;
			b ^= world_seed;
			
			a ^= b << s | b >> w - s;
			a *= 2048419325;
			float random = a * (PI / ~(~0u >> 1)); // in [0, 2*Pi]

			return {cos(random), sin(random)};
		}


		inline static float interpolate(float a0, float a1, float weight)
		{
			return (a1 - a0) * (3.0f - weight * 2.0f) * weight * weight + a0;
		}

	};



}