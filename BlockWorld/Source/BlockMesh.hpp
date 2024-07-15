#pragma once

#include "Buffers.hpp"
#include "Vertices.hpp"
#include "Context.hpp"
#include "BlockUtils.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace bwrenderer
{
	
	
	class BlockMesh
	{
		using vertex_vector = std::vector<BlockVertex>;
	public:
		BlockMesh();

		BlockMesh(const vertex_vector& vertices);

		void draw() const;

		inline const vertex_buffer& getVertexBuffer() const { return vbo; }
		inline const vertex_array& getVertexArray() const { return vao; }

		inline void setVertexBuffer(std::vector<BlockVertex>&& vertices) 
		{ 
			//GL_INFO("Vertex Buffer set for BlockMesh.");
			vbo.bind();
			vbo.attachBuffer(vertices[0].data, static_cast<GLuint>(vertices.size() * sizeof(BlockVertex))); 
			this->vertices = std::move(vertices);
		}
	private:
		std::vector<BlockVertex> vertices;

		vertex_buffer vbo;
		vertex_array vao;
	private:
		vertex_buffer setupVertexBuffer();
		vertex_array setupVertexArray();
	};
}