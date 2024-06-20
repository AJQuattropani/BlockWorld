#include "Chunk.h"

namespace bwgame {


	Chunk::Chunk(ChunkCoords chunkCoords, const BlockRegister& blocks)
		: chunkCoords(chunkCoords), blockMap(), model(std::make_unique<bwrenderer::ChunkModel>())
	{
		flags = ~flags;
		BW_INFO("Chunk generated.");

		srand(chunkCoords.seed);

		for (uint16_t y = 0; y < 256; y++)
		{
			for (uint8_t z = 0; z < 15; z++)
			{
				for (uint8_t x = 0; x < 15; x++)
				{
					if (rand() % 10 == 0 || y < 60)
					{
						if (y < 55) setBlock({ x, (uint8_t)y, z }, blocks.stone);
						if (y < 60 && y >= 55) setBlock({ x, (uint8_t)y, z }, blocks.dirt);
						if (y >= 60) setBlock({ x, (uint8_t)y, z }, blocks.grass);
					}
				}
			}

		}


		model->setModelMatrix(chunkCoords);

		update();
	}

	Chunk::~Chunk()
	{
		BW_INFO("Chunk deleted.");
	}

	void Chunk::update()
	{
		if (flags.test(CHUNK_FLAGS::MODEL_UPDATE_FLAG))
		{
			model->updateRenderData(packageRenderData());
			flags.flip(CHUNK_FLAGS::MODEL_UPDATE_FLAG);
		}
	}

	void Chunk::render(bwrenderer::RenderContext& context) const
	{
		model->render(context);
	}

	std::vector<bwrenderer::BlockVertex> Chunk::packageRenderData() const
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

		uint8_t u = 0;
		do
		{
			bc_vertex_helper_ikj(u, binary_chunk->n_xzy, binary_chunk->p_xzy, binary_chunk->n_zxy, binary_chunk->p_zxy, vertices);
			bc_vertex_helper_jik(u % 16, u / 16, binary_chunk->n_yxz, binary_chunk->p_yxz, vertices);
		} while (u++ < 255);

		delete binary_chunk;

		return vertices;

	}

	void Chunk::bc_vertex_helper_ikj(uint8_t u, utils::data_IKJ& n_xzy, utils::data_IKJ& p_xzy,
		utils::data_IKJ& n_zxy, utils::data_IKJ& p_zxy, std::vector<bwrenderer::BlockVertex>& vertices) const
	{
		// chunks are 15x15x256
		for (uint8_t b = 0; b < 16; b++)
		{
			for (uint8_t trailing_zeros = std::countr_zero<uint16_t>(n_xzy[u].v16i_u16[b]);
				trailing_zeros < 16;
				trailing_zeros = std::countr_zero<uint16_t>(n_xzy[u].v16i_u16[b]))
			{
				vertices.emplace_back(
					packageBlockRenderData({ trailing_zeros, u, b }, BlockDirection::FORWARD));
				n_xzy[u].v16i_u16[b] &= ~(1U << trailing_zeros);
			}
			for (uint8_t trailing_zeros = std::countr_zero<uint16_t>(p_xzy[u].v16i_u16[b]);
				trailing_zeros < 16;
				trailing_zeros = std::countr_zero<uint16_t>(p_xzy[u].v16i_u16[b]))
			{
				vertices.emplace_back(
					packageBlockRenderData({ trailing_zeros, u, b }, BlockDirection::BACKWARD));
				p_xzy[u].v16i_u16[b] &= ~(1U << trailing_zeros);
			}
			for (uint8_t trailing_zeros = std::countr_zero<uint16_t>(n_zxy[u].v16i_u16[b]);
				trailing_zeros < 16;
				trailing_zeros = std::countr_zero<uint16_t>(n_zxy[u].v16i_u16[b]))
			{
				vertices.emplace_back(
					packageBlockRenderData({ b, u, trailing_zeros }, BlockDirection::RIGHT));
				n_zxy[u].v16i_u16[b] &= ~(1U << trailing_zeros);
			}
			for (uint8_t trailing_zeros = std::countr_zero<uint16_t>(p_zxy[u].v16i_u16[b]);
				trailing_zeros < 16;
				trailing_zeros = std::countr_zero<uint16_t>(p_zxy[u].v16i_u16[b]))
			{
				vertices.emplace_back(
					packageBlockRenderData({b, u, trailing_zeros}, BlockDirection::LEFT));
				p_zxy[u].v16i_u16[b] &= ~(1U << trailing_zeros);
			}
		}
	}

	void Chunk::bc_vertex_helper_jik(uint8_t i, uint8_t k,
		utils::data_JIK& n_yxz, utils::data_JIK& p_yxz, std::vector<bwrenderer::BlockVertex>& vertices) const

	{
		for (uint8_t b = 0; b < 4; b++)
		{
			for (uint8_t trailing_zeros = std::countr_zero<uint64_t>(n_yxz[k][i].v4i_u64[b]);
				trailing_zeros < 64;
				trailing_zeros = std::countr_zero<uint64_t>(n_yxz[k][i].v4i_u64[b]))
			{
				uint8_t j = b << 6 | (63 - trailing_zeros);
				vertices.emplace_back(
					packageBlockRenderData({ i, j, k }, BlockDirection::DOWN));
				n_yxz[k][i].v4i_u64[b] &= ~(1ULL << trailing_zeros);
			}
			for (uint8_t trailing_zeros = std::countr_zero<uint64_t>(p_yxz[k][i].v4i_u64[b]);
				trailing_zeros < 64;
				trailing_zeros = std::countr_zero<uint64_t>(p_yxz[k][i].v4i_u64[b]))
			{
				uint8_t j = b << 6 | (63 - trailing_zeros);
				vertices.emplace_back(
					packageBlockRenderData({ i, j, k }, BlockDirection::UP));
				p_yxz[k][i].v4i_u64[b] &= ~(1ULL << trailing_zeros);
			}
		}
	}
};