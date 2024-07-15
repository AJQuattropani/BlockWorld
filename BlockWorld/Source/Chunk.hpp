#pragma once

#include "Debug.hpp"
#include "ChunkData.hpp"
#include "ChunkModel.hpp"
#include "ExtendLong.hpp"
#include "BinaryChunk.hpp"
#include "Blocks.hpp"
#include "ThreadList.hpp"
#include "Timer.hpp"

#include <ranges>
#include <unordered_set>
#include <future>

namespace bwgame
{
	class Chunk
	{
	public:
		Chunk(ChunkCoords chunk_coords, std::unordered_map<ChunkCoords, Chunk> const* chunk_map);
		

		Chunk(const Chunk&) = delete;
		Chunk(Chunk&& other) noexcept : flags(other.flags), chunk_coords(other.chunk_coords), chunk_map(other.chunk_map) {
			chunk_data_mutex.lock();
			block_map = std::move(other.block_map);
			model = std::move(other.model);
			async_chunk_operations = std::move(other.async_chunk_operations);

			chunk_data_mutex.unlock();
		}

		Chunk& operator=(const Chunk&) = delete;
		Chunk& operator=(Chunk&&) = delete;

		~Chunk();

		void update();

		void render(bwrenderer::Shader& shader) const;

		inline const ChunkCoords& getChunkCoords() const { return chunk_coords; }

		inline const Block& getBlock(block_coord_t x, block_coord_t y, block_coord_t z) 
		{	return getBlock({ x,y,z });	}

		inline void reserve(uint16_t amount) {
			if (block_map.size() >= CHUNK_VOLUME)
			{
				BW_WARN("Reserve failed - Chunk is already at full capacity.");
				return;
			}
			{
				std::scoped_lock<std::mutex> transferModelDataLock(chunk_data_mutex);
				if (block_map.size() + amount > CHUNK_VOLUME)
				{
					BW_WARN("Reserve attempted - Chunk has been set to maximum capacity.");
					block_map.reserve(CHUNK_VOLUME - block_map.size());
					return;
				}
				block_map.reserve(amount);
			}

		}

		inline const Block& getBlock(const BlockCoords& coords)
		{
			//if (const auto& block = block_map.find(coords); block != block_map.end()) return block->second;
			return block_map[coords];
		}

		inline void deleteBlock(const BlockCoords& coords)
		{
			std::scoped_lock<std::mutex> transferModelDataLock(chunk_data_mutex);
			block_map.erase(coords);
		}

		void setBlock(const BlockCoords& coords, const Block& block);

	private:
		const ChunkCoords chunk_coords;
		ChunkFlags flags;
		// TODO find better data structure for holding onto blocks and chunks
		std::unordered_map<BlockCoords, Block> block_map;
		std::unique_ptr<bwrenderer::ChunkModel> model;
		// TODO make shared_ptr
		std::unordered_map<ChunkCoords, Chunk> const* chunk_map;

		std::unique_ptr<utils::ThreadList> async_chunk_operations;
		mutable std::mutex chunk_data_mutex;

	private:
		inline const Block& getBlockConst(const BlockCoords& coords) const
		{
			if (const auto& block = block_map.find(coords); block != block_map.end()) return block->second;
			BW_ASSERT("Block not found");
		}

		inline void reloadModelData() const {
			auto data = packageRenderData();
			{
				std::scoped_lock<std::mutex> transferModelDataLock(chunk_data_mutex);
				model->updateRenderData(std::move(data));
			}
		}
		std::vector<bwrenderer::BlockVertex> packageRenderData() const;
		void bc_vertex_helper_ikj(uint8_t u, utils::data_IKJ& n_xzy, utils::data_IKJ& p_xzy, utils::data_IKJ& n_zxy, utils::data_IKJ& p_zxy,
			std::vector<bwrenderer::BlockVertex>& vertices) const;
		void bc_vertex_helper_jik(uint8_t i, uint8_t k, utils::data_JIK& n_yxz, utils::data_JIK& p_yxz,
			std::vector<bwrenderer::BlockVertex>& vertices) const;
		inline bwrenderer::BlockVertex packageBlockRenderData(bwgame::BlockCoords pos, BlockDirection dir) const
		{
			return { glm::vec3(pos.x, pos.y, pos.z), utils::blockDirectionToNormal(dir), getBlockConst(pos).getTexture(dir) };
		}



	};

}