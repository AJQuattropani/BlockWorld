#pragma once

#include "DayCycle.hpp"
#include "Shader.hpp"
#include "Context.hpp"
#include "Vertices.hpp"
#include "Texture.hpp"

namespace bwrenderer {

	class SkyBox {
	public:
		SkyBox();

		~SkyBox();

		void render(RenderContext& context, const DayLightCycle& day_light_cycle);
	private:
		vertex_buffer vbo;
		vertex_array vao;
		Shader shader;
	private:
		vertex_buffer setupVertexBuffer();
		vertex_array setupVertexArray();
	};
}