#pragma once

#include "Debug.h"
#include "ChunkData.h"
#include "ChunkModel.h"
#include "ExtendLong.h"
#include "BinaryChunk.h"

#include <unordered_set>

namespace bwgame
{
	enum class BlockType {
		AIR = 0,
		DIRT,
		GRASS,
		STONE,
		COBBLESTONE
	};

	enum class BlockDirection {
		UP, DOWN, LEFT, RIGHT, FORWARD, BACKWARD
	};

	glm::vec3 blockDirectionToNormal(BlockDirection dir);

	class Block {
	public:
		Block(BlockType type) : type(type)
		{

		}

		Block() : type(BlockType::AIR) {}

		BlockType type;
		glm::vec2 getPackagedRenderData(BlockDirection dir = BlockDirection::UP) const;
		//static Block Air;
	};

	class Chunk
	{
	public:
		Chunk(ChunkCoords chunkCoords);

		~Chunk();

		void update();

		void render(bwrenderer::RenderContext& context) const;

		inline const ChunkCoords getChunkCoords() const { return chunkCoords; }
		inline const Block& getBlock(block_coord_t x, block_coord_t y, block_coord_t z) const 
		{	return getBlock({ x,y,z });	}

		inline const Block& getBlock(const BlockCoords& coords) const
		{
			if (const auto& block = blockMap.find(coords); block != blockMap.end()) return block->second;
			BW_ASSERT(false, "Block not found.");
		}

		inline void deleteBlock(const BlockCoords& coords)
		{
			blockMap.erase(coords);
		}

		void setBlock(const BlockCoords& coords, BlockType type)
		{
			BW_ASSERT(type != BlockType::AIR, "Cannot convert a block to air.");
			blockMap[coords] = Block(type);
		}

	private:
		const ChunkCoords chunkCoords;
		ChunkFlags flags;
		// TODO find better data structure for holding onto blocks and chunks
		std::unordered_map<BlockCoords, Block> blockMap;
		std::unique_ptr<bwrenderer::ChunkModel> model;
	private:
		std::vector<bwrenderer::BlockVertex> packageRenderData() const;
		void bc_vertex_helper_ikj(uint16_t u, utils::data_IKJ& n_xzy, utils::data_IKJ& p_xzy, utils::data_IKJ& n_zxy, utils::data_IKJ& p_zxy,
			std::vector<bwrenderer::BlockVertex>& vertices) const;
		void bc_vertex_helper_jik(uint16_t i, uint16_t k, utils::data_JIK& n_yxz, utils::data_JIK& p_yxz,
			std::vector<bwrenderer::BlockVertex>& vertices) const;




	};

}