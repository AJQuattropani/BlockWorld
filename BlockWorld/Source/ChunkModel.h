#pragma once

#include "BlockMesh.h"

namespace bwrenderer {

	class ChunkModel
	{
	public:
		
		ChunkModel() {
			BW_INFO("ChunkModel generated.");
		}

		~ChunkModel() {
			BW_INFO("ChunkModel destroyed.");
		}

		ChunkModel(const ChunkModel& other) = default;
		ChunkModel& operator=(const ChunkModel& other) = default;

		ChunkModel(ChunkModel&& other) = default;
		ChunkModel& operator=(ChunkModel&& other) = default;

		void render(RenderContext& context) const;

		inline void updateRenderData(std::vector<BlockVertex> blockVertices)
		{
			mesh.setVertexBuffer(blockVertices);
		}

		inline void setModelMatrix(const glm::vec3& position)
		{
			model = modelMatrixInit(glm::translate(glm::mat4(1.0), position));
		}

	private:
		BlockMesh mesh;
		glm::mat4 model = modelMatrixInit(glm::mat4(1.0));
	private:
		inline static const glm::mat4 modelMatrixInit(const glm::mat4& premodel)
		{
			return glm::scale(premodel, glm::vec3(1.0));
		}
	};

}