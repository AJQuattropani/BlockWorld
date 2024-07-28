#pragma once

#include "Debug.hpp"
#include "BlockUtils.hpp"

namespace bwgame
{
	enum class BlockType {
		AIR = 0,
		DIRT,
		GRASS,
		STONE,
		COBBLESTONE,
		SAND,
		GRAVEL,
		WOOD,
		WATER,
		LEAVES
	};

	class Block {
	public:
		Block(BlockType type, std::shared_ptr<CubeTexData>&& texture_data);
		Block();
		
		Block(const Block& other) = default;
		Block(Block&& other) = default;

		Block& operator=(const Block& other) = default;
		Block& operator=(Block&& other) = default;

		inline bool isAir() const { return type == BlockType::AIR; }

		glm::vec2 getTexture(BlockDirection dir) const;
	private:
		BlockType type;
		std::shared_ptr<CubeTexData const> texture_data;
	};

	struct BlockRegister {
		const Block air;
		const Block dirt;
		const Block grass;
		const Block full_grass;
		const Block stone;
		const Block cobblestone;
		const Block sand;
		const Block gravel;
		const Block logs;
		const Block water;
		const Block leaves;

		BlockRegister();
	};



	
}