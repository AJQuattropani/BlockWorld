#include "Skybox.hpp"

namespace bwrenderer {

	SkyBox::SkyBox() : vbo(setupVertexBuffer()), vao(setupVertexArray()), shader("World", "skybox")
	{
		shader.bind();
		shader.setUniform1i("skyBox", 0);
		shader.unbind();
	}

	SkyBox::~SkyBox()
	{
	}

	void SkyBox::render(RenderContext& context, const DayLightCycle& day_light_cycle)
	{
		glm::vec4 sky_color = glm::mix(glm::vec4(0.05, 0.05, 0.15f, 1.0f), glm::vec4(0.0, 0.55, 1.0f, 1.0f), glm::clamp(4.0 * glm::sin(day_light_cycle.sun_angle), 0.0, 1.0));

		glClearColor(sky_color.r, sky_color.g, sky_color.b, sky_color.a);
		
		glDepthFunc(GL_LEQUAL);

		shader.bind();
		shader.setUniformMat4f("projection", context.projection_matrix);
		shader.setUniformMat4f("view", context.view_matrix);

		shader.setUniformMat4f("rotation", glm::rotate(glm::mat4(1.0), day_light_cycle.sun_angle, glm::vec3(0.0, 0.0, 1.0)));

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, context.texture_cache.findOrLoad("World", "sky_textures.jpeg").textureID);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, 0);

		vao.bind();

		glDrawArrays(GL_TRIANGLES, 0, 12);
		vao.unbind();

		glDepthFunc(GL_LESS);
	}

	vertex_buffer SkyBox::setupVertexBuffer()
	{
		//BW_INFO("New vertex buffer created.");
		static GLfloat skyboxVertices[] = {
			// positions    

		-1.0f, -1.0f,  1.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
		-1.0f,  1.0f, -1.0f, 2.0f, 1.0f,
		-1.0f,  1.0f, -1.0f, 2.0f, 1.0f,
		-1.0f,  1.0f,  1.0f, 2.0f, 0.0f,
		-1.0f, -1.0f,  1.0f, 1.0f, 0.0f,

		 1.0f, -1.0f, -1.0f, 0.0f, 1.0f,
		 1.0f, -1.0f,  1.0f, 0.0f, 0.0f,
		 1.0f,  1.0f,  1.0f, 1.0f, 0.0f,
		 1.0f,  1.0f,  1.0f, 1.0f, 0.0f,
		 1.0f,  1.0f, -1.0f, 1.0f, 1.0f,
		 1.0f, -1.0f, -1.0f, 0.0f, 1.0f	
		};

		return vertex_buffer(skyboxVertices, sizeof(skyboxVertices));
	}

	vertex_array SkyBox::setupVertexArray()
	{
		//BW_INFO("New vertex array created.");
		vertex_layout layout_object;
		layout_object.push(GL_FLOAT, 3);
		layout_object.push(GL_FLOAT, 2);

		return vertex_array(vbo, layout_object);
	}
}