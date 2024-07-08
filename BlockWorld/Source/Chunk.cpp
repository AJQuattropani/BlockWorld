#include "Chunk.h"

#include "Timer.h"
#include <ranges>

namespace bwgame {


	Chunk::Chunk(ChunkCoords chunkCoords, const std::unordered_map<ChunkCoords, Chunk>& chunkMap)
		: chunkCoords(chunkCoords), blockMap(), chunkMap(chunkMap), model(std::make_unique<bwrenderer::ChunkModel>())
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
		TIME_FUNC("Packaging");

		//// may need to reallocate up to 6 times. TODO change allocator
		std::vector<bwrenderer::BlockVertex> vertices;
		vertices.reserve(256 * 15);

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
#define OPTIMIZATION 1
#if OPTIMIZATION == 1
		{
			TIME_FUNC("Optimization");
			if (const auto& n_x_Chunk = chunkMap.find(ChunkCoords{ chunkCoords.x + 1, chunkCoords.z }); n_x_Chunk != chunkMap.end())
			{
				auto n_x_It_func = [](const BlockCoords& coords) -> bool { return coords.x == 0; };
				for (const auto coords : std::views::keys(n_x_Chunk->second.blockMap) | std::views::filter(n_x_It_func))
				{
					utils::set(binary_chunk->n_xzy, 15, coords.y, coords.z);
				}
			}
			if (const auto& n_z_Chunk = chunkMap.find(ChunkCoords{ chunkCoords.x, chunkCoords.z + 1 }); n_z_Chunk != chunkMap.end())
			{
				auto n_z_It_func = [](const BlockCoords& coords) -> bool { return coords.z == 0; };
				for (const auto coords : std::views::keys(n_z_Chunk->second.blockMap) | std::views::filter(n_z_It_func))
				{
					utils::set(binary_chunk->n_zxy, 15, coords.y, coords.x);
				}
			}
			if (const auto& p_x_Chunk = chunkMap.find(ChunkCoords{ chunkCoords.x - 1, chunkCoords.z }); p_x_Chunk != chunkMap.end())
			{
				auto p_x_It_func = [](const BlockCoords& coords) -> bool { return coords.x == 14; };
				for (const auto coords : std::views::keys(p_x_Chunk->second.blockMap) | std::views::filter(p_x_It_func))
				{
					utils::set(binary_chunk->p_xzy, 0, coords.y, coords.z);
				}
			}
			if (const auto& p_z_Chunk = chunkMap.find(ChunkCoords{ chunkCoords.x, chunkCoords.z - 1 }); p_z_Chunk != chunkMap.end())
			{
				auto p_z_It_func = [](const BlockCoords& coords) -> bool { return coords.z == 14; };
				for (const auto coords : std::views::keys(p_z_Chunk->second.blockMap) | std::views::filter(p_z_It_func))
				{
					utils::set(binary_chunk->p_zxy, 0, coords.y, coords.x);
				}
			}
		}
#endif

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
		uint8_t b, trailing_zeros, i;
		// chunks are 15x15x256
		for (b = 0; b < 16; b++)
		{
			for (trailing_zeros = std::countr_zero<uint16_t>(n_xzy[u].v16i_u16[b]);
				trailing_zeros < 16 - 1;
				trailing_zeros = std::countr_zero<uint16_t>(n_xzy[u].v16i_u16[b]))
			{
				n_xzy[u].v16i_u16[b] &= ~(1U << trailing_zeros);
				i = trailing_zeros;
				vertices.emplace_back(
					packageBlockRenderData({ i, u, b }, BlockDirection::FORWARD));
			}
			for (trailing_zeros = std::countr_zero<uint16_t>(p_xzy[u].v16i_u16[b]);
				trailing_zeros < 16;
				trailing_zeros = std::countr_zero<uint16_t>(p_xzy[u].v16i_u16[b]))
			{
				p_xzy[u].v16i_u16[b] &= ~(1U << trailing_zeros);
				if (trailing_zeros == 0) continue;
				i = trailing_zeros - 1;
				vertices.emplace_back(
					packageBlockRenderData({ i, u, b }, BlockDirection::BACKWARD));
			}
			for (trailing_zeros = std::countr_zero<uint16_t>(n_zxy[u].v16i_u16[b]);
				trailing_zeros < 16 - 1;
				trailing_zeros = std::countr_zero<uint16_t>(n_zxy[u].v16i_u16[b]))
			{
				n_zxy[u].v16i_u16[b] &= ~(1U << trailing_zeros);
				i = trailing_zeros;
				vertices.emplace_back(
					packageBlockRenderData({ b, u, i }, BlockDirection::RIGHT));
			}
			for (trailing_zeros = std::countr_zero<uint16_t>(p_zxy[u].v16i_u16[b]);
				trailing_zeros < 16;
				trailing_zeros = std::countr_zero<uint16_t>(p_zxy[u].v16i_u16[b]))
			{
				p_zxy[u].v16i_u16[b] &= ~(1U << trailing_zeros);
				if (trailing_zeros == 0) continue;
				i = trailing_zeros - 1;
				vertices.emplace_back(
					packageBlockRenderData({b, u, i}, BlockDirection::LEFT));
			}
		}
	}

	void Chunk::bc_vertex_helper_jik(uint8_t i, uint8_t k,
		utils::data_JIK& n_yxz, utils::data_JIK& p_yxz, std::vector<bwrenderer::BlockVertex>& vertices) const

	{
		uint8_t j, trailing_zeros;
		for (uint8_t b = 0; b < 4; b++)
		{
			for (trailing_zeros = std::countr_zero<uint64_t>(n_yxz[k][i].v4i_u64[b]);
				trailing_zeros < 64;
				trailing_zeros = std::countr_zero<uint64_t>(n_yxz[k][i].v4i_u64[b]))
			{
				n_yxz[k][i].v4i_u64[b] &= ~(1ULL << trailing_zeros);
				//if (trailing_zeros == 1) continue;
				j = b << 6 | (63 - trailing_zeros);
				vertices.emplace_back(
					packageBlockRenderData({ i, j, k }, BlockDirection::DOWN));
			}
			for (trailing_zeros = std::countr_zero<uint64_t>(p_yxz[k][i].v4i_u64[b]);
				trailing_zeros < 64;
				trailing_zeros = std::countr_zero<uint64_t>(p_yxz[k][i].v4i_u64[b]))
			{
				p_yxz[k][i].v4i_u64[b] &= ~(1ULL << trailing_zeros);
				j = b << 6 | (63 - trailing_zeros);
				vertices.emplace_back(
					packageBlockRenderData({ i, j, k }, BlockDirection::UP));
			}
		}
	}
}