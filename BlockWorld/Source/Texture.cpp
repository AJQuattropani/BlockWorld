#include "Texture.hpp"

namespace bwrenderer {

	GLuint initializeTexture(TextureBuffer* buffer);

	std::string makePath(const std::string& type, const std::string& file)
	{
		return TEXTURE_PATH + type + "/" + file;
	}

	GLuint bwrenderer::createTexture(TextureBuffer* buffer, const std::string& type, const std::string& file_path)
	{
		buffer->type = type;
		buffer->file_path = file_path;
		return initializeTexture(buffer);
	}
	void bwrenderer::deleteTexture(TextureBuffer* buffer)
	{
		GL_INFO("Texture deleted: %s | Type: %s | ID: %x", buffer->file_path.c_str(), buffer->type.c_str(), buffer->textureID);
		glDeleteTextures(1, &buffer->textureID);
	}

	GLuint initializeTexture(TextureBuffer* buffer)
	{
		GLuint texture;
		glGenTextures(1, &texture);

		//// STB IMAGE
		stbi_set_flip_vertically_on_load(false);
		unsigned char* data = stbi_load(buffer->file_path.c_str(), &buffer->width, &buffer->height, &buffer->nrChannels, 0);
		buffer->format = updateFormat(buffer->nrChannels);

		GL_ASSERT(data, "Failed to load image at %s", buffer->file_path.c_str());

		glBindTexture(GL_TEXTURE_2D, texture);

		glTexImage2D(GL_TEXTURE_2D, 0, (buffer->format == GL_RGB ? GL_SRGB : buffer->format), buffer->width, buffer->height, 0, buffer->format, GL_UNSIGNED_BYTE, data);
		//glGenerateMipmap(GL_TEXTURE_2D);

		GL_DEBUG("Texture successfully generated as: %s | Type: %s | ID: %x | %d x %d : %d",
			buffer->file_path.c_str(), buffer->type.c_str(), buffer->textureID, buffer->width, buffer->height, buffer->nrChannels);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		// set texture filtering parameters
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		stbi_image_free(data);

		buffer->textureID = texture;

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
			GL_ASSERT(false, "Malformed image format: %i", nrChannels);
			return 0;
		}
	}

	TextureCache::TextureCache() = default;

	TextureCache::~TextureCache()
	{
		for (auto texBuff : loaded_textures)
		{
			deleteTexture(&texBuff.second);
		}

	}

	TextureBuffer& TextureCache::find(const std::string& type)
	{
		if (const auto& texBuff = loaded_textures.find(type); texBuff != loaded_textures.end()) return texBuff->second;
		GL_ASSERT(false, "'%s': Texture not found.", type);
	}

	TextureBuffer& TextureCache::findOrLoad(const std::string& type, const std::string& name)
	{
		return findOrLoad_impl(type, makePath(type, name));
	}

	TextureBuffer& TextureCache::findOrLoad_impl(const std::string& type, const std::string& file_path)
	{
		if (const auto& texBuff = loaded_textures.find(file_path); texBuff != loaded_textures.end()) return texBuff->second;

		TextureBuffer loadedBuffer;
		createTexture(&loadedBuffer, type, file_path);

		return (loaded_textures[file_path] = std::move(loadedBuffer));
	}

	TextureBuffer& TextureCache::push(const std::string& name, TextureBuffer&& textureBuffer)
	{
		return loaded_textures.emplace(name, textureBuffer).first->second;
	}
}