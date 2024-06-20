#include "Blocks.h"

namespace bwgame {
	
	Block::Block(BlockType type, std::shared_ptr<CubeTexData>&& textureData)
		: type(type), textureData(std::move(textureData))
	{
		BW_INFO("Block initialized.");
	}

	Block::Block() : type(BlockType::AIR), textureData(nullptr)
	{
		BW_INFO("Block initialized.");
	}

	glm::vec2 Block::getTexture(BlockDirection dir) const
	{
		switch (dir)
		{
		case BlockDirection::UP:
			return glm::vec2(textureData->up.x, textureData->up.y);
		case BlockDirection::DOWN:
			return glm::vec2(textureData->down.x, textureData->down.y);
		case BlockDirection::FORWARD:
			return glm::vec2(textureData->front.x, textureData->front.y);
		case BlockDirection::BACKWARD:
			return glm::vec2(textureData->back.x, textureData->back.y);
		case BlockDirection::RIGHT:
			return glm::vec2(textureData->right.x, textureData->right.y);
		case BlockDirection::LEFT:
			return glm::vec2(textureData->left.x, textureData->left.y);
		}
		BW_ASSERT(false, "Block Direction Enum unrecognized.");
		return { 0,0 };
	}

	BlockRegister::BlockRegister() :
		air(),
		dirt(BlockType::DIRT, utils::makeCubeTexData({ 0,0 })),
		grass(BlockType::GRASS, utils::makePillarTexData({ 2,0 }, { 1,0 }, { 0,0 })),
		full_grass(BlockType::GRASS, utils::makeCubeTexData({ 2,0 })),
		stone(BlockType::STONE, utils::makeCubeTexData({ 3,0 })),
		cobblestone(BlockType::COBBLESTONE, utils::makeCubeTexData({ 4,0 }))
	{
		BW_INFO("Blocks initialized.");
	}
}