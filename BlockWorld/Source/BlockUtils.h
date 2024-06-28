#pragma once

#include "glm/vec3.hpp"
#include "glm/vec2.hpp"

namespace bwgame
{

	enum class BlockDirection {
		UP, DOWN, LEFT, RIGHT, FORWARD, BACKWARD
	};

	enum class BlockTextureType {
		CUBE, PILLAR
	};

	using block_coord_t = uint8_t;
	union BlockCoords
	{
		struct {
			block_coord_t x, y, z;
		};
		uint32_t index;
		block_coord_t data[3];

	};
	
	using w_block_coord_t = int64_t;

	struct TexData
	{
		uint32_t x, y;
	};

	struct CubeTexData
	{
		TexData front;
		TexData back;
		TexData right;
		TexData left;
		TexData up;
		TexData down;
	};
};

namespace utils {
	inline std::shared_ptr<bwgame::CubeTexData> makeCubeTexData (const bwgame::TexData& side)
	{
		return std::make_shared<bwgame::CubeTexData>(bwgame::CubeTexData{
			.front = side, .back = side, 
			.right = side, .left = side, 
			.up = side, .down = side,
		});
	}

	inline std::shared_ptr<bwgame::CubeTexData> makePillarTexData (const bwgame::TexData& top,
		const bwgame::TexData& side, const bwgame::TexData& bottom)
	{
		return std::make_shared<bwgame::CubeTexData>(bwgame::CubeTexData{
			.front = side, .back = side, 
			.right = side,	.left = side,
			.up = top, .down = bottom,
		});
	}

	inline constexpr glm::vec3 blockDirectionToNormal (bwgame::BlockDirection dir)
	{
		switch (dir)
		{
			case bwgame::BlockDirection::UP:
				return glm::vec3(0, 1, 0);
			case bwgame::BlockDirection::DOWN:
				return glm::vec3(0, -1, 0);
			case bwgame::BlockDirection::RIGHT:
				return glm::vec3(0, 0, 1);
			case bwgame::BlockDirection::LEFT:
				return glm::vec3(0, 0, -1);
			case bwgame::BlockDirection::FORWARD:
				return glm::vec3(1, 0, 0);
			case bwgame::BlockDirection::BACKWARD:
				return glm::vec3(-1, 0, 0);
		}
		BW_ASSERT(false, "Block Direction Enum exception.");
		return { 0,0,0 };
	}
};

namespace bwrenderer {
	union BlockVertex
	{
		struct
		{
			glm::vec3 position;
			glm::vec3 normal;
			glm::vec2 tex_coords;
		};
		GLfloat data[3 + 3 + 2];
	};


};