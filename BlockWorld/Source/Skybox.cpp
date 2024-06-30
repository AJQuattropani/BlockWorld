#include "Skybox.h"

namespace bwrenderer {

	SkyBox::SkyBox() : vbo(setupVertexBuffer()), vao(setupVertexArray()), shader("World", "skybox")
	{

		buffer[0].type = "World";
		buffer[1].type = "World";
		buffer[2].type = "World";
		buffer[3].type = "World";
		buffer[4].type = "World";
		buffer[5].type = "World";

		buffer[0].filePath = makePath("World", "skybox_pos_x.jpeg");
		buffer[1].filePath = makePath("World", "skybox_neg_x.jpeg");
		buffer[2].filePath = makePath("World", "skybox_pos_y.jpeg");
		buffer[3].filePath = makePath("World", "skybox_neg_y.jpeg");
		buffer[4].filePath = makePath("World", "skybox_pos_z.jpeg");
		buffer[5].filePath = makePath("World", "skybox_neg_z.jpeg");

		createCubeMap();


	}

	SkyBox::~SkyBox()
	{
		for (int i = 0; i < 6; i++)
		{
				deleteTexture(buffer + i);
		}
	}

	void SkyBox::render(RenderContext& context)
	{
		glDepthFunc(GL_LEQUAL);

		shader.bind();
		shader.setUniformMat4f("projection", context.projectionMatrix);
		shader.setUniformMat4f("view", context.viewMatrix);

		vao.bind();
		glBindTexture(GL_TEXTURE_CUBE_MAP, buffer[0].textureID);

		glDrawArrays(GL_TRIANGLES, 0, 36);
		vao.unbind();

		glDepthFunc(GL_LESS);
	}

	void SkyBox::createCubeMap()
	{
		unsigned int textureID;
		glGenTextures(1, &textureID);
		glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);

		GL_DEBUG("Texture Skybox loading from: %s", buffer[0].filePath.c_str());

		unsigned char* data;
		for (int i = 0; i < 6; i++)
		{
			stbi_set_flip_vertically_on_load(false);
			unsigned char* data = stbi_load(buffer[i].filePath.c_str(), &buffer[i].width, &buffer[i].height, &buffer[i].nrChannels, 0);
			buffer[i].format = updateFormat(buffer[i].nrChannels);

			GL_ASSERT(data, "Failed to load image at %s", buffer[i].filePath.c_str());

			glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
				0, buffer[i].format,
				buffer[i].width,
				buffer[i].height,
				0, buffer[i].format,
				GL_UNSIGNED_BYTE,
				data);

			GL_INFO("Texture successfully generated as: %s | Type: %s | ID: %x | %d x %d : %d",
				buffer[i].filePath.c_str(), buffer[i].type.c_str(), buffer[i].textureID, buffer[i].width, buffer[i].height, buffer[i].nrChannels);

			stbi_image_free(data);

			buffer[i].textureID = textureID;
		}

		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
		// set texture filtering parameters
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	}

	vertex_buffer SkyBox::setupVertexBuffer()
	{
		//BW_INFO("New vertex buffer created.");
		static GLfloat skyboxVertices[] = {
			// positions          
		-1.0f,  1.0f, -1.0f,
		-1.0f, -1.0f, -1.0f,
		 1.0f, -1.0f, -1.0f,
		 1.0f, -1.0f, -1.0f,
		 1.0f,  1.0f, -1.0f,
		-1.0f,  1.0f, -1.0f,

		-1.0f, -1.0f,  1.0f,
		-1.0f, -1.0f, -1.0f,
		-1.0f,  1.0f, -1.0f,
		-1.0f,  1.0f, -1.0f,
		-1.0f,  1.0f,  1.0f,
		-1.0f, -1.0f,  1.0f,

		 1.0f, -1.0f, -1.0f,
		 1.0f, -1.0f,  1.0f,
		 1.0f,  1.0f,  1.0f,
		 1.0f,  1.0f,  1.0f,
		 1.0f,  1.0f, -1.0f,
		 1.0f, -1.0f, -1.0f,

		-1.0f, -1.0f,  1.0f,
		-1.0f,  1.0f,  1.0f,
		 1.0f,  1.0f,  1.0f,
		 1.0f,  1.0f,  1.0f,
		 1.0f, -1.0f,  1.0f,
		-1.0f, -1.0f,  1.0f,

		-1.0f,  1.0f, -1.0f,
		 1.0f,  1.0f, -1.0f,
		 1.0f,  1.0f,  1.0f,
		 1.0f,  1.0f,  1.0f,
		-1.0f,  1.0f,  1.0f,
		-1.0f,  1.0f, -1.0f,

		-1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f,  1.0f,
		 1.0f, -1.0f, -1.0f,
		 1.0f, -1.0f, -1.0f,
		-1.0f, -1.0f,  1.0f,
		 1.0f, -1.0f,  1.0f
		};

		return vertex_buffer(skyboxVertices, sizeof(skyboxVertices));
	}

	vertex_array SkyBox::setupVertexArray()
	{
		//BW_INFO("New vertex array created.");
		vertex_layout layout_object;
		layout_object.push(GL_FLOAT, 3);

		return vertex_array(vbo, layout_object);
	}
}