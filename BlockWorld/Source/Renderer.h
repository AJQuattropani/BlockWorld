#pragma once

#include "RenderContext.h"
#include "Camera.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <set>

class Layer {
	Shader shader;

};


//
//class RendererObject
//{
//public:
//	RendererObject(Texture&& texture) = 0;
//
//	virtual void renderPre(const& RenderContext) = 0;
//	virtual void render(const& RenderContext) = 0;
//	virtual void renderPost(const& RenderContext) = 0;
//};
//
//class CubeMesh : RendererObject
//{
//
//private:
//	Texture texture;
//};
//
//class WorldMesh : RendererObject
//{
//
//};
//
//class Layer {
//	Shader shader;
//	
//};