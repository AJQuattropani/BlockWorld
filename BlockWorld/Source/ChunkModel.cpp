#include "ChunkModel.hpp"

void bwrenderer::ChunkModel::render(Shader& shader) const
{
	shader.setUniformMat4f("model", model);
	mesh.draw();

}