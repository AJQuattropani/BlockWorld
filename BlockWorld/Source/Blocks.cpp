#include "Blocks.h"

namespace bwgame {

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
};