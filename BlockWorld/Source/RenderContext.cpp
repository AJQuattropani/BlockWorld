#include "RenderContext.h"

bwrenderer::TexturedCubeMesh::TexturedCubeMesh()
	: texture(nullptr), worldPosition(glm::vec3(0.0, 0.0, 0.0))
{
	GL_INFO("Empty instance of TexturedCubeMesh created.");
	buildNewBuffer();
}

bwrenderer::TexturedCubeMesh::TexturedCubeMesh(TextureBuffer* buffer, glm::vec3 worldPos) 
	: texture(buffer), worldPosition(worldPos)
{
	GL_INFO("Completed instance of TexturedCubeMesh created.");
	buildNewBuffer();
}

void bwrenderer::TexturedCubeMesh::render(Shader& shader)
{
	glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
	glBindVertexArray(vertex_array);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture->textureID);

	shader.setUniformMat4f("model", glm::translate(glm::mat4(1.0f), worldPosition));
	shader.setUniform1i("block_texture", 0);
	glDrawArrays(GL_TRIANGLES, 0, 36);
}

void bwrenderer::TexturedCubeMesh::buildNewBuffer()
{
	glGenVertexArrays(1, &vertex_array);
	glGenBuffers(1, &vertex_buffer);

	glBindVertexArray(vertex_array);

	glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_data), vertex_data, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));
	glEnableVertexAttribArray(1);

	glBindVertexArray(0);
}
////////////////////////////////

bwrenderer::TexturedNormalCubeMesh::TexturedNormalCubeMesh()
	: texture(nullptr), worldPosition(glm::vec3(0.0, 0.0, 0.0))
{
	GL_INFO("Empty instance of TexturedNormalCubeMesh created.");
	buildNewBuffer();
}

bwrenderer::TexturedNormalCubeMesh::TexturedNormalCubeMesh(TextureBuffer* buffer, glm::vec3 worldPos)
	: texture(buffer), worldPosition(worldPos)
{
	GL_INFO("Completed instance of TexturedNormalCubeMesh created.");
	buildNewBuffer();
}

void bwrenderer::TexturedNormalCubeMesh::render(Shader& shader)
{
	glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
	glBindVertexArray(vertex_array);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, texture->textureID);

	shader.setUniformMat4f("model", glm::translate(glm::mat4(1.0f), worldPosition));
	shader.setUniform1i("block_texture", 0);
	glDrawArrays(GL_POINTS, 0, sizeof(vertex_data)/(8*sizeof(GLfloat)));
}

void bwrenderer::TexturedNormalCubeMesh::buildNewBuffer()
{
	glGenVertexArrays(1, &vertex_array);
	glGenBuffers(1, &vertex_buffer);

	glBindVertexArray(vertex_array);

	glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_data), vertex_data, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (void*)(6 * sizeof(GLfloat)));
	glEnableVertexAttribArray(2);

	glBindVertexArray(0);
}