#pragma once

#include "Chunk.h"
#include "Blocks.h"

namespace bwgame {


	class PerlinNoise {
	public:
		PerlinNoise(uint64_t seed) : seed(seed)
		{

		}

	private:
		uint64_t seed;
	};

	class WorldGenerator {
	public:
		WorldGenerator(uint64_t seed, const std::shared_ptr<BlockRegister>& blocks) : seed(seed), blocks(blocks)
		{
			BW_INFO("World Seed: %Ld", seed);
		}

		inline uint64_t getSeed() const { return seed; }

		void buildChunk(const ChunkCoords& coords, Chunk& chunk) const;

	private:
		uint64_t seed;
		std::shared_ptr<const BlockRegister> blocks;

		inline const Block& getBlockLayered(int64_t depth, uint8_t height) const;
	};





}