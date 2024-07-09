#pragma once

#include "Debug.h"
#include "ChunkData.h"
#include "ChunkModel.h"
#include "ExtendLong.h"
#include "BinaryChunk.h"
#include "Blocks.h"

#include <ranges>
#include <unordered_set>
#include <future>

namespace bwgame
{
	class Chunk
	{
	public:
		Chunk(ChunkCoords chunkCoords, const std::unordered_map<ChunkCoords, Chunk>& chunkMap);
		
		Chunk(const Chunk&) = default;
		Chunk(Chunk&&) = default;

		Chunk& operator=(const Chunk&) = default;
		Chunk& operator=(Chunk&&) = default;

		~Chunk();

		void update();

		void render(bwrenderer::Shader& shader) const;

		inline const ChunkCoords getChunkCoords() const { return chunkCoords; }
		inline const Block& getBlock(block_coord_t x, block_coord_t y, block_coord_t z) const 
		{	return getBlock({ x,y,z });	}

		inline void reserve(uint16_t amount) {
			if (blockMap.size() >= CHUNK_VOLUME)
			{
				BW_WARN("Reserve failed - Chunk is already at full capacity.");
				return;
			}
			if (blockMap.size() + amount > CHUNK_VOLUME)
			{
				BW_WARN("Reserve attempted - Chunk has been set to maximum capacity.");
				blockMap.reserve(CHUNK_VOLUME-blockMap.size());
				return;
			}
			blockMap.reserve(amount);

		}

		inline const Block& getBlock(const BlockCoords& coords) const
		{
			if (const auto& block = blockMap.find(coords); block != blockMap.end()) return block->second;
		}

		inline void deleteBlock(const BlockCoords& coords)
		{
			blockMap.erase(coords);
		}

		void setBlock(const BlockCoords& coords, const Block& block);

	private:
		const ChunkCoords chunkCoords;
		ChunkFlags flags;
		// TODO find better data structure for holding onto blocks and chunks
		std::unordered_map<BlockCoords, Block> blockMap;
		std::unique_ptr<bwrenderer::ChunkModel> model;
		// TODO make shared_ptr
		const std::unordered_map<ChunkCoords, Chunk>& chunkMap;
	private:
		std::vector<bwrenderer::BlockVertex> packageRenderData() const;
		void bc_vertex_helper_ikj(uint8_t u, utils::data_IKJ& n_xzy, utils::data_IKJ& p_xzy, utils::data_IKJ& n_zxy, utils::data_IKJ& p_zxy,
			std::vector<bwrenderer::BlockVertex>& vertices) const;
		void bc_vertex_helper_jik(uint8_t i, uint8_t k, utils::data_JIK& n_yxz, utils::data_JIK& p_yxz,
			std::vector<bwrenderer::BlockVertex>& vertices) const;
		inline bwrenderer::BlockVertex packageBlockRenderData(bwgame::BlockCoords pos, BlockDirection dir) const
		{
			return { glm::vec3(pos.x, pos.y, pos.z), utils::blockDirectionToNormal(dir), getBlock(pos).getTexture(dir) };
		}



	};

}