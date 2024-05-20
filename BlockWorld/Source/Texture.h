#pragma once
#include <GL/glew.h>
#include "Debug.h"

#include "Vendor/stb_image.h"

#include <string>
#include <ranges>

// Texture Register. Responsible for loading and caching textures on the GPU, and memory management of texture buffers.
class TextureRegister
{
public:
	TextureRegister();
	TextureRegister(const std::string& type);
	~TextureRegister();

	TextureRegister(TextureRegister&& other) noexcept;

	TextureRegister& operator=(TextureRegister&& other) noexcept;

	GLuint findOrLoad(const std::string& name);
	GLuint findOrLoadLazy(const std::string& name);
private:
	std::string type;
	std::unordered_map<std::string, size_t> cached_names;
	std::vector<GLuint> textureBuffers;
private:
	const static std::string TEXTURE_PATH;
public:
	GLuint loadTexture(const std::string& name, GLint wrap_s = GL_REPEAT, GLint wrap_t = GL_REPEAT,
		GLint min_filter = GL_LINEAR_MIPMAP_LINEAR, GLint mag_filter = GL_LINEAR);
};