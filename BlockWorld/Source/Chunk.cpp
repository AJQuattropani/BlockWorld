#include "Chunk.h"

glm::vec3 bwgame::blockDirectionToNormal(BlockDirection dir)
{
	switch (dir)
	{
	case BlockDirection::UP:
		return glm::vec3( 0,1,0 );
	case BlockDirection::DOWN:
		return glm::vec3( 0,-1,0 );
	case BlockDirection::RIGHT:
		return glm::vec3( 0,0,1 );
	case BlockDirection::LEFT:
		return glm::vec3( 0,0,-1 );
	case BlockDirection::FORWARD:
		return glm::vec3( 1,0,0 );
	case BlockDirection::BACKWARD:
		return glm::vec3( -1,0,0 );
	}
	BW_ASSERT(false, "Block Direction Enum exception.");
}

glm::vec2 bwgame::Block::getPackagedRenderData(BlockDirection dir) const
{
	switch (type)
	{
	case BlockType::COBBLESTONE:
		return glm::vec2( 4,0 );
	case BlockType::STONE:
		return glm::vec2( 3,0 );
	case BlockType::GRASS:
		// TODO replace w lambda
		switch (dir)
		{
		case BlockDirection::UP:
			return glm::vec2( 2,0 );
		case BlockDirection::DOWN:
			return glm::vec2( 0,0 );
		default:
			return glm::vec2( 1,0 );
		}
	case BlockType::DIRT:
	default:
		return glm::vec2( 0,0 );
	}
}

bwgame::Chunk::Chunk(ChunkCoords chunkCoords)
	: chunkCoords(chunkCoords), blockMap(), model(std::make_unique<bwrenderer::ChunkModel>())
{
		flags = ~flags;
		BW_INFO("Chunk generated.");

		srand(0);

		for (uint16_t y = 0; y < 256; y++)
		{
			for (uint8_t z = 0; z < 15; z++)
			{
				for (uint8_t x = 0; x < 15; x++)
				{
					if (rand() % 4 == 0)
					{
						if (y < 12) setBlock({ x, (uint8_t)y, z }, BlockType::STONE);
						if (y < 15 && y >= 12) setBlock({ x, (uint8_t)y, z }, BlockType::DIRT);
						if (y >= 15) setBlock({ x, (uint8_t)y, z }, BlockType::GRASS);
					}
				}
			}

		}
		

		model->setModelMatrix(chunkCoords);

		update();
}

bwgame::Chunk::~Chunk()
{
	BW_INFO("Chunk deleted.");
}

void bwgame::Chunk::update()
{
	if (flags.test(CHUNK_FLAGS::MODEL_UPDATE_FLAG))
	{
		model->updateRenderData(packageRenderData());
		flags.flip(CHUNK_FLAGS::MODEL_UPDATE_FLAG);
	}
}

void bwgame::Chunk::render(bwrenderer::RenderContext& context) const
{
	model->render(context);
}

std::vector<bwrenderer::BlockVertex> bwgame::Chunk::packageRenderData() const
{

	//// may need to reallocate up to 6 times. TODO change allocator
	std::vector<bwrenderer::BlockVertex> vertices;
	vertices.reserve(256 * 15 * 15);

	/*for (const auto& block : blockMap)
	{
		vertices.emplace_back(bwrenderer::BlockVertex{
			glm::vec3(block.first.x,block.first.y,block.first.z),
			blockDirectionToNormal(BlockDirection::UP),
			block.second.getPackagedRenderData(BlockDirection::UP)
			});
		vertices.emplace_back(bwrenderer::BlockVertex{
			glm::vec3(block.first.x,block.first.y,block.first.z),
			blockDirectionToNormal(BlockDirection::DOWN),
			block.second.getPackagedRenderData(BlockDirection::DOWN)
			});
		vertices.emplace_back(bwrenderer::BlockVertex{
			glm::vec3(block.first.x,block.first.y,block.first.z),
			blockDirectionToNormal(BlockDirection::RIGHT),
			block.second.getPackagedRenderData(BlockDirection::RIGHT)
			});
		vertices.emplace_back(bwrenderer::BlockVertex{
			glm::vec3(block.first.x,block.first.y,block.first.z),
			blockDirectionToNormal(BlockDirection::LEFT),
			block.second.getPackagedRenderData(BlockDirection::LEFT)
			});
		vertices.emplace_back(bwrenderer::BlockVertex{
			glm::vec3(block.first.x,block.first.y,block.first.z),
			blockDirectionToNormal(BlockDirection::FORWARD),
			block.second.getPackagedRenderData(BlockDirection::FORWARD)
			});
		vertices.emplace_back(bwrenderer::BlockVertex{
			glm::vec3(block.first.x,block.first.y,block.first.z),
			blockDirectionToNormal(BlockDirection::BACKWARD),
			block.second.getPackagedRenderData(BlockDirection::BACKWARD)
			});
	}*/

	utils::BinaryChunk* binary_chunk = new utils::BinaryChunk{};

	//// package all data into blockmap
	for (const auto& block : blockMap)
	{
		utils::set(binary_chunk->n_xzy, block.first.x, block.first.y, block.first.z);
		utils::set(binary_chunk->p_xzy, block.first.x, block.first.y, block.first.z);

		utils::set(binary_chunk->n_yxz, block.first.x, block.first.y, block.first.z);
		utils::set(binary_chunk->p_yxz, block.first.x, block.first.y, block.first.z);

		utils::set(binary_chunk->n_zxy, block.first.z, block.first.y, block.first.x);
		utils::set(binary_chunk->p_zxy, block.first.z, block.first.y, block.first.x);
	}

	//// convert block placement data to block exposure data
	bc_face_bits_inplace(*binary_chunk);

	for (uint16_t u = 0; u < 256; u++)
	{
		bc_vertex_helper_ikj(u, binary_chunk->n_xzy, binary_chunk->p_xzy, binary_chunk->n_zxy, binary_chunk->p_zxy, vertices);
		bc_vertex_helper_jik(u % 16, u / 16, binary_chunk->n_yxz, binary_chunk->p_yxz, vertices);
	}

	delete binary_chunk;

	return vertices;

}

void bwgame::Chunk::bc_vertex_helper_ikj(uint16_t u, utils::data_IKJ& n_xzy, utils::data_IKJ& p_xzy, 
	utils::data_IKJ& n_zxy, utils::data_IKJ& p_zxy, std::vector<bwrenderer::BlockVertex>& vertices) const
{
	// chunks are 15x15x256
	for (uint16_t b = 0; b < 16; b++)
	{
		for (uint8_t trailing_zeros = std::countr_zero<uint16_t>(n_xzy[u].v16i_u16[b]);
			trailing_zeros < 16;
			trailing_zeros = std::countr_zero<uint16_t>(n_xzy[u].v16i_u16[b]))
		{
			vertices.emplace_back(bwrenderer::BlockVertex
				{
				glm::vec3( trailing_zeros, u, b ),
				blockDirectionToNormal(BlockDirection::FORWARD),
				getBlock( trailing_zeros, u, b ).getPackagedRenderData(BlockDirection::FORWARD)
				}
			);
			n_xzy[u].v16i_u16[b] &= ~(1U << trailing_zeros);
		}
		for (uint8_t trailing_zeros = std::countr_zero<uint16_t>(p_xzy[u].v16i_u16[b]);
			trailing_zeros < 16;
			trailing_zeros = std::countr_zero<uint16_t>(p_xzy[u].v16i_u16[b]))
		{
			vertices.emplace_back(bwrenderer::BlockVertex
				{
				glm::vec3( trailing_zeros, u, b ),
				blockDirectionToNormal(BlockDirection::BACKWARD),
				getBlock( trailing_zeros, u, b ).getPackagedRenderData(BlockDirection::BACKWARD)
				}
			);
			p_xzy[u].v16i_u16[b] &= ~(1U << trailing_zeros);
		}
		for (uint8_t trailing_zeros = std::countr_zero<uint16_t>(n_zxy[u].v16i_u16[b]);
			trailing_zeros < 16;
			trailing_zeros = std::countr_zero<uint16_t>(n_zxy[u].v16i_u16[b]))
		{
			vertices.emplace_back(bwrenderer::BlockVertex
				{
				glm::vec3( b, u, trailing_zeros ),
				blockDirectionToNormal(BlockDirection::RIGHT),
				getBlock( b, u, trailing_zeros ).getPackagedRenderData(BlockDirection::RIGHT)
				}
			);
			n_zxy[u].v16i_u16[b] &= ~(1U << trailing_zeros);
		}
		for (uint8_t trailing_zeros = std::countr_zero<uint16_t>(p_zxy[u].v16i_u16[b]);
			trailing_zeros < 16;
			trailing_zeros = std::countr_zero<uint16_t>(p_zxy[u].v16i_u16[b]))
		{
			vertices.emplace_back(bwrenderer::BlockVertex
				{
				glm::vec3( b, u, trailing_zeros ),
				blockDirectionToNormal(BlockDirection::LEFT),
				getBlock( b, u, trailing_zeros ).getPackagedRenderData(BlockDirection::LEFT)
				}
			);
			p_zxy[u].v16i_u16[b] &= ~(1U << trailing_zeros);
		}
	}
}

void bwgame::Chunk::bc_vertex_helper_jik(uint16_t i, uint16_t k, 
	utils::data_JIK& n_yxz, utils::data_JIK& p_yxz, std::vector<bwrenderer::BlockVertex>& vertices) const

{
	for (uint16_t b = 0; b < 4; b++)
	{
		for (uint16_t trailing_zeros = std::countr_zero<uint64_t>(n_yxz[k][i].v4i_u64[b]);
			trailing_zeros < 64;
			trailing_zeros = std::countr_zero<uint64_t>(n_yxz[k][i].v4i_u64[b]))
		{
			vertices.emplace_back(bwrenderer::BlockVertex
				{
				glm::vec3( i, (b << 6) + (63 - trailing_zeros), k ),
				blockDirectionToNormal(BlockDirection::DOWN),
				getBlock(i, (b << 6) + (63 - trailing_zeros), k).getPackagedRenderData(BlockDirection::DOWN)
				}
			);
			n_yxz[k][i].v4i_u64[b] &= ~(1ULL << trailing_zeros);
		}
		for (uint16_t trailing_zeros = std::countr_zero<uint64_t>(p_yxz[k][i].v4i_u64[b]);
			trailing_zeros < 64;
			trailing_zeros = std::countr_zero<uint64_t>(p_yxz[k][i].v4i_u64[b]))
		{
			vertices.emplace_back(bwrenderer::BlockVertex
				{
				glm::vec3( i, (b << 6) + (63 - trailing_zeros), k ),
				blockDirectionToNormal(BlockDirection::UP),
				getBlock(i, (b << 6) + (63 - trailing_zeros), k).getPackagedRenderData(BlockDirection::UP)
				}
			);
			p_yxz[k][i].v4i_u64[b] &= ~(1ULL << trailing_zeros);
		}
	}
}
