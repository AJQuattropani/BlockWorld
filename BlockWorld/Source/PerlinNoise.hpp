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

	//class Lehmer64Generator
	//{
	//private:
	//	uint64_t lehmer_state;
	//public:
	//	Lehmer64Generator(uint64_t seed = 0) : lehmer_state(seed)
	//	{	}

	//	inline void setSeed(uint64_t new_seed) { lehmer_state = new_seed; }

	//	inline uint64_t operator() ()
	//	{
	//		return next();
	//	}

	//	float nextFloat(float min, float max)
	//	{
	//		float result = ((float)next() / (float)(std::numeric_limits<uint64_t>::max()) * (max - min)) + min;
	//		return result;
	//	}

	//	int64_t nextInt(int64_t min = std::numeric_limits<int64_t>::min(), int64_t max = std::numeric_limits<int64_t>::max())
	//	{
	//		return next() % (max - min) + min;
	//	}

	//private:
	//	uint64_t next()
	//	{
	//		lehmer_state += 0xe120fc15;
	//		uint64_t temporary = lehmer_state * 0x4a39b70d;
	//		uint64_t shift1 = (temporary >> 32) ^ temporary;
	//		temporary = shift1 * 0x12fad5c9;
	//		uint64_t shift2 = (temporary >> 32) ^ temporary;

	//		return shift2;
	//	}

	//};

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
			//union seedConvert{
			//	struct {
			//		int32_t x;
			//		int32_t z;
			//	};
			//	uint64_t seed;
			//} seedCon = {cx, cz};

			//generator.setSeed(seedCon.seed | world_seed);
			//float angle = generator.nextFloat(0.0f, 1.0f);

			//float supplmentary_angle = angle + 0.25f;
			//if (supplmentary_angle > 1.0f) supplmentary_angle -= 1.0f;
			////return {fast_sin(supplmentary_angle), fast_sin(angle)};
			//return { glm::cos(angle), glm::sin(angle)};

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
			float random = a * (3.14159265 / ~(~0u >> 1)); // in [0, 2*Pi]

			//float random_supp = random - 3.14159265f / 2.0f;
			//if (random_supp < 0.0f) random_supp += 3.14159265f * 2.0f;

			// Create the vector from the angle
			return {glm::cos(random), glm::sin(random)};

		}

		inline static float fast_sin(float angle_revs)
		{
			/*if (angle_revs > 0.5f) return 16.0f * angle_revs * angle_revs - 16.0f * angle_revs - 8.0f * angle_revs + 8.0f;
			return -16.0f * angle_revs * angle_revs + 8.0f * angle_revs;*/
			constexpr float PI = 3.14159265;
			constexpr float PI4 = PI * PI * PI * PI;
			return 8.0f / (PI4) * angle_revs * (PI - angle_revs) * (2.0f * PI - angle_revs);
		}

		inline static float interpolate(float a0, float a1, float weight)
		{
			return (a1 - a0) * (3.0f - weight * 2.0f) * weight * weight + a0;
		}

	};



}