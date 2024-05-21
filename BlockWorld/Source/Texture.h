#pragma once

#include "Debug.h"

#include "GL/glew.h"
#include <string>

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

	GLuint createTexture(TextureBuffer* buffer, const std::string& type, const std::string& file);

	void deleteTexture(TextureBuffer* buffer);
}