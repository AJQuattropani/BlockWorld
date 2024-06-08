#pragma once

#include "Debug.h"

#include "GL/glew.h"
#include <string>

#include "Vendor/stb_image.h"

namespace bwrenderer
{
	inline const std::string TEXTURE_PATH = "Resources/Textures/";

	struct TextureBuffer
	{
		std::string type = "";
		std::string filePath = "";
		GLuint textureID = 0;

		int width = 0, height = 0;
		int nrChannels = 0, format = 0;
	};

	GLuint createTexture(TextureBuffer* buffer, const std::string& type, const std::string& filePath);

	void deleteTexture(TextureBuffer* buffer);

	class TextureCache
	{
	public:
		TextureCache();

		~TextureCache();

		TextureBuffer& findOrLoad(const std::string& type, const std::string& name);

		TextureBuffer& findOrLoad_impl(const std::string& type, const std::string& filePath);

	private:
		std::unordered_map<std::string, TextureBuffer> loaded_textures;
	};

}