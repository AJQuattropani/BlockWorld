#include "Texture.h"

#include "Vendor/stb_image.h"

using namespace bwrenderer;

GLuint initializeTexture(TextureBuffer* buffer);
int updateFormat(int nrChannels);

GLuint createTexture(TextureBuffer* buffer, const std::string& type, const std::string& file)
{
	buffer->type = type;
	buffer->filePath = TEXTURE_PATH + type + "/" + file;
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
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	//// STB IMAGE
	stbi_set_flip_vertically_on_load(true);
	unsigned char* data = stbi_load(buffer->filePath.c_str(), &buffer->width, &buffer->height, &buffer->nrChannels, 0);
	buffer->format = updateFormat(buffer->nrChannels);

	BW_ASSERT(data, "Failed to load image at %s", buffer->filePath.c_str());

	glTexImage2D(GL_TEXTURE_2D, 0, buffer->format, buffer->width, buffer->height, 0, buffer->format, GL_UNSIGNED_BYTE, data);
	glGenerateMipmap(GL_TEXTURE_2D);

	BW_INFO("Texture successfully generated as: %s | Type: %s | ID: %x | %d x %d : %d", buffer->filePath, buffer->type, buffer->width, buffer->height, buffer->nrChannels);

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

void deleteTexture(TextureBuffer* buffer) 
{
	glDeleteTextures(1, &buffer->textureID);
}
