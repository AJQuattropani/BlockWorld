#include "Chunk.h"

namespace bwgame {


	Chunk::Chunk(ChunkCoords chunkCoords)
		: chunkCoords(chunkCoords), blockMap(), model(std::make_unique<bwrenderer::ChunkModel>())
	{
		flags = ~flags;
		BW_INFO("Chunk generated.");

		blockMap.reserve(256*100);
		model->setModelMatrix(chunkCoords);
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

	void Chunk::render(bwrenderer::Shader& shader) const
	{
		model->render(shader);
	}

	void Chunk::setBlock(const BlockCoords& coords, const Block& block)
	{
		BW_ASSERT(coords.x <= CHUNK_WIDTH_BLOCKS
			&& coords.z <= CHUNK_WIDTH_BLOCKS, // Note: block_coord_t already puts cap on y coord range
			"Block outside chunk range.");
		if (block.isAir())
		{
			BW_WARN("Air block instruction ignored.");
			return;
		}
		blockMap.try_emplace(coords, block);
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
		for (const auto& [coords, block] : blockMap)
		{
			utils::set(binary_chunk->n_xzy, coords.x, coords.y, coords.z);
			utils::set(binary_chunk->p_xzy, coords.x + 1, coords.y, coords.z);

			utils::set(binary_chunk->n_yxz, coords.x, coords.y, coords.z);
			utils::set(binary_chunk->p_yxz, coords.x, coords.y, coords.z);

			utils::set(binary_chunk->n_zxy, coords.z, coords.y, coords.x);
			utils::set(binary_chunk->p_zxy, coords.z + 1, coords.y, coords.x);
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
				uint8_t i = trailing_zeros;
				vertices.emplace_back(
					packageBlockRenderData({ i, u, b }, BlockDirection::FORWARD));
				n_xzy[u].v16i_u16[b] &= ~(1U << trailing_zeros);
			}
			for (uint8_t trailing_zeros = std::countr_zero<uint16_t>(p_xzy[u].v16i_u16[b]);
				trailing_zeros < 16;
				trailing_zeros = std::countr_zero<uint16_t>(p_xzy[u].v16i_u16[b]))
			{
				uint8_t i = trailing_zeros - 1;
				vertices.emplace_back(
					packageBlockRenderData({ i, u, b }, BlockDirection::BACKWARD));
				p_xzy[u].v16i_u16[b] &= ~(1U << trailing_zeros);
			}
			for (uint8_t trailing_zeros = std::countr_zero<uint16_t>(n_zxy[u].v16i_u16[b]);
				trailing_zeros < 16;
				trailing_zeros = std::countr_zero<uint16_t>(n_zxy[u].v16i_u16[b]))
			{
				uint8_t i = trailing_zeros;
				vertices.emplace_back(
					packageBlockRenderData({ b, u, i }, BlockDirection::RIGHT));
				n_zxy[u].v16i_u16[b] &= ~(1U << trailing_zeros);
			}
			for (uint8_t trailing_zeros = std::countr_zero<uint16_t>(p_zxy[u].v16i_u16[b]);
				trailing_zeros < 16;
				trailing_zeros = std::countr_zero<uint16_t>(p_zxy[u].v16i_u16[b]))
			{
				uint8_t i = trailing_zeros - 1;
				vertices.emplace_back(
					packageBlockRenderData({b, u, i}, BlockDirection::LEFT));
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
}