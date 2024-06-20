#pragma once

#include "Debug.h"


#include "BlockUtils.h"

namespace bwgame
{
	class Block {
	public:
		Block(BlockType type, std::shared_ptr<CubeTexData>&& textureData) 
			: type(type), textureData(std::move(textureData))
		{
			BW_INFO("Block initialized.");
		}

		Block() : type(BlockType::AIR), textureData(nullptr)
		{

		}

		inline bool isAir() { return type == BlockType::AIR; }

		glm::vec2 getTexture(BlockDirection dir = BlockDirection::UP) const;
	private:
		BlockType type;
		std::shared_ptr<CubeTexData const> textureData;
	};


	struct BlockRegister {
		const Block air;
		const Block dirt;
		const Block grass;
		const Block full_grass;
		const Block stone;
		const Block cobblestone;

		BlockRegister() :
			air(),
			dirt(BlockType::DIRT, utils::makeCubeTexData({ 0,0 })),
			grass(BlockType::GRASS, utils::makePillarTexData({ 2,0 }, { 1,0 }, { 0,0 })),
			full_grass(BlockType::GRASS, utils::makeCubeTexData({ 2,0 })),
			stone(BlockType::STONE, utils::makeCubeTexData({ 3,0 })),
			cobblestone(BlockType::COBBLESTONE, utils::makeCubeTexData({ 4,0 }))
		{
			BW_INFO("Blocks initialized.");
		}
	};



	
}