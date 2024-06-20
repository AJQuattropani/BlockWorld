#include "ChunkModel.h"

void bwrenderer::ChunkModel::render(RenderContext& context) const
{
	context.shader.bind();
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, context.texture_cache.findOrLoad("Blocks", "blockmap.jpeg").textureID);

	context.shader.setUniform2f("image_size", 256.0, 256.0);
	context.shader.setUniformMat4f("view", context.viewMatrix);
	context.shader.setUniformMat4f("projection", context.projectionMatrix);
	context.shader.setUniformMat4f("model", model);

	context.shader.setUniform1f("chunk_width", bwgame::CHUNK_WIDTH_BLOCKS);

	context.shader.setUniform1i("block_texture", 0);
	
	mesh.draw(context.shader);

}