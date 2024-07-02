#pragma once

#include "DayCycle.h"
#include "Shader.h"
#include "RenderContext.h"
#include "Vertices.h"
#include "Texture.h"
#include "WorldRenderer.h"

namespace bwrenderer {

	class SkyBox {
	public:
		SkyBox();

		~SkyBox();

		void render(RenderContext& context, DayLightCycle& dayLightCycle);
	private:
		//TextureBuffer buffer;
		vertex_buffer vbo;
		vertex_array vao;
		Shader shader;
	private:
		/*void createCubeMap();*/
		vertex_buffer setupVertexBuffer();
		vertex_array setupVertexArray();
	};
}