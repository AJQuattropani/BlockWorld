#pragma once

#include "Debug.hpp"

#include "GL/glew.h"
#include <string>

#include "Vendor/stb_image.h"

namespace bwrenderer
{
	inline const std::string TEXTURE_PATH = "Resources/Textures/";
	std::string makePath(const std::string& type, const std::string& file);
	
	struct TextureBuffer
	{
		std::string type = "";
		std::string file_path = "";
		GLuint textureID = 0;

		int width = 0, height = 0;
		int nrChannels = 0, format = 0;
	};

	GLuint createTexture(TextureBuffer* buffer, const std::string& type, const std::string& filePath);

	void deleteTexture(TextureBuffer* buffer);
	int updateFormat(int nrChannels);

	class TextureCache
	{
	public:
		TextureCache();

		~TextureCache();

		TextureBuffer& find(const std::string& type);

		TextureBuffer& findOrLoad(const std::string& type, const std::string& name);

		TextureBuffer& findOrLoad_impl(const std::string& type, const std::string& filePath);

		TextureBuffer& push(const std::string& name, TextureBuffer&& textureBuffer);

	private:
		std::unordered_map<std::string, TextureBuffer> loaded_textures;
	};

}