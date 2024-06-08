#pragma once

#include "Buffers.h"
#include "Vertices.h"
#include "RenderContext.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace bwrenderer
{
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
	
	class BlockMesh
	{
		using vertex_vector = std::vector<BlockVertex>;
	public:
		BlockMesh();

		BlockMesh(const vertex_vector& vertices);

		void draw(const Shader& shader) const;

		inline const vertex_buffer& getVertexBuffer() const { return vbo; }
		inline const vertex_array& getVertexArray() const { return vao; }

		inline void setVertexBuffer(const std::vector<BlockVertex>& vertices) 
		{ 
			GL_INFO("Vertex Buffer set for BlockMesh.");
			vbo.bind();
			vbo.attachBuffer(vertices[0].data, vertices.size() * sizeof(BlockVertex)); 
			this->vertices = vertices;
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