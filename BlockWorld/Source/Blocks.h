#pragma once

#include "Debug.h"


#include "BlockUtils.h"

namespace bwgame
{
	class Block {
	public:
		Block(BlockType type, std::shared_ptr<CubeTexData>&& textureData);
		Block();
		
		Block(const Block& other) = default;
		Block(Block&& other) = default;

		Block& operator=(const Block& other) = default;
		Block& operator=(Block&& other) = default;

		inline bool isAir() const { return type == BlockType::AIR; }

		glm::vec2 getTexture(BlockDirection dir) const;
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

		BlockRegister();
	};



	
}