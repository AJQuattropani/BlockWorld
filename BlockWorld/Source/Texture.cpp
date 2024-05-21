#include "Texture.h"

#include "Vendor/stb_image.h"

using namespace bwrenderer;

GLuint initializeTexture(TextureBuffer* buffer);
int updateFormat(int nrChannels);
std::string makePath(const std::string& type, const std::string& file)
{
	return TEXTURE_PATH + type + "/" + file;
}

GLuint bwrenderer::createTexture(TextureBuffer* buffer, const std::string& type, const std::string& filePath)
{
	buffer->type = type;
	buffer->filePath = filePath;
	return initializeTexture(buffer);
}

GLuint initializeTexture(TextureBuffer* buffer)
{
	GLuint texture;
	glGenTextures(1, &texture);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	// set texture filtering parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	//// STB IMAGE
	stbi_set_flip_vertically_on_load(true);
	unsigned char* data = stbi_load(buffer->filePath.c_str(), &buffer->width, &buffer->height, &buffer->nrChannels, 0);
	buffer->format = updateFormat(buffer->nrChannels);

	BW_ASSERT(data, "Failed to load image at %s", buffer->filePath.c_str());

	glTexImage2D(GL_TEXTURE_2D, 0, buffer->format, buffer->width, buffer->height, 0, buffer->format, GL_UNSIGNED_BYTE, data);
	glGenerateMipmap(GL_TEXTURE_2D);

	BW_INFO("Texture successfully generated as: %s | Type: %s | ID: %x | %d x %d : %d", 
		buffer->filePath.c_str(), buffer->type.c_str(), buffer->textureID, buffer->width, buffer->height, buffer->nrChannels);

	stbi_image_free(data);

	return buffer->textureID;
}

int updateFormat(int nrChannels)
{
	switch (nrChannels)
	{
	case 1:
		return GL_R;
	case 2:
		return GL_RG;
	case 3:
		return GL_RGB;
	case 4:
		return GL_RGBA;
	default:
		BW_ASSERT(false, "Malformed image format.");
		return 0;
	}
}

void bwrenderer::deleteTexture(TextureBuffer* buffer)
{
	GL_INFO("Texture deleted: %s | Type: %s | ID: %x", buffer->filePath, buffer->type, buffer->textureID);
	glDeleteTextures(1, &buffer->textureID);
}

bwrenderer::TextureCache::TextureCache() = default;

bwrenderer::TextureCache::~TextureCache()
{
	for (auto texBuff : loaded_textures)
	{
		deleteTexture(&texBuff.second);
	}

}

TextureBuffer& bwrenderer::TextureCache::findOrLoad(const std::string& type, const std::string& name)
{
	return findOrLoad_impl(type, makePath(type, name));
}

TextureBuffer& bwrenderer::TextureCache::findOrLoad_impl(const std::string& type, const std::string& filePath)
{
	if (const auto& texBuff = loaded_textures.find(filePath); texBuff != loaded_textures.end()) return texBuff->second;

	TextureBuffer loadedBuffer;
	createTexture(&loadedBuffer, type, filePath);
	
	return (loaded_textures[filePath] = std::move(loadedBuffer));
}
