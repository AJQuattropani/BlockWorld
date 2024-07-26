#pragma once

#include "Chunk.hpp"
#include "Blocks.hpp"
#include "PerlinNoise.hpp"

namespace bwgame {


	class WorldGenerator {
	public:
		WorldGenerator(uint64_t seed, const std::shared_ptr<BlockRegister>& blocks) : seed(seed), blocks(blocks), world_noise_gen(seed)
		{
			BW_INFO("World Seed: %Ld", seed);
		}

		inline uint64_t getSeed() const { return seed; }

		Chunk buildChunk(ChunkCoords coords, std::unordered_map<ChunkCoords, Chunk> const* chunkMap) const;

	private:
		uint64_t seed;
		std::shared_ptr<const BlockRegister> blocks;
		const utils::PerlinNoiseGenerator world_noise_gen;

		inline const Block& getBlockLayered(int64_t depth, uint8_t height, float biome) const;
	};





}