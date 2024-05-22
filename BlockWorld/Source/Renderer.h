#pragma once

#include "RenderContext.h"
#include "Camera.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace bwrenderer {

	enum class LayerType
	{
		BLOCK,
		AMBIENT
	};

	class Layer {
	public:
		virtual ~Layer() = 0;

		Layer(const Layer&) = delete;
		Layer(Layer&&) = delete;

		Layer& operator=(const Layer&) = delete;
		Layer& operator=(Layer&&) = delete;

		virtual void render() = 0;

	private:
		std::vector<RendererObject> render_objects;
		Shader shader;
	};

	class BlockLayer {
		Layer()


	};

}