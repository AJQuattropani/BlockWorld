#include "Blocks.hpp"

namespace bwgame {
	
	Block::Block(BlockType type, std::shared_ptr<CubeTexData>&& texture_data)
		: type(type), texture_data(std::move(texture_data))
	{
		BW_INFO("Block initialized.");
	}

	Block::Block() : type(BlockType::AIR), texture_data(nullptr)
	{
		BW_INFO("Block initialized.");
	}

	/*Block::Block(const Block& other) : type(other.type), textureData(other.textureData)
	{
		BW_INFO("Block copied.");
	}

	Block::Block(Block&& other) noexcept : type(std::move(other.type)), textureData(std::move(other.textureData))
	{
		BW_INFO("Block moved.");
	}

	Block& Block::operator=(const Block& other)
	{
		this->~Block();
		new (this) Block(other);
		return *this;
	}

	Block& Block::operator=(Block&& other) noexcept
	{
		this->~Block();
		new (this) Block(std::move(other));
		return *this;
	}
	*/


	glm::vec2 Block::getTexture(BlockDirection dir) const
	{
		switch (dir)
		{
		case BlockDirection::UP:
			return glm::vec2(texture_data->up.x, texture_data->up.y);
		case BlockDirection::DOWN:
			return glm::vec2(texture_data->down.x, texture_data->down.y);
		case BlockDirection::FORWARD:
			return glm::vec2(texture_data->front.x, texture_data->front.y);
		case BlockDirection::BACKWARD:
			return glm::vec2(texture_data->back.x, texture_data->back.y);
		case BlockDirection::RIGHT:
			return glm::vec2(texture_data->right.x, texture_data->right.y);
		case BlockDirection::LEFT:
			return glm::vec2(texture_data->left.x, texture_data->left.y);
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
		cobblestone(BlockType::COBBLESTONE, utils::makeCubeTexData({ 4,0 })),
		sand(BlockType::SAND, utils::makeCubeTexData({ 5,0 })),
		gravel(BlockType::GRAVEL, utils::makeCubeTexData({ 6,0 }))
	{
		BW_INFO("Blocks initialized.");
	}
}