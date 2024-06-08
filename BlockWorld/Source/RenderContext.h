#pragma once
#include "Shader.h"
#include "Texture.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace bwrenderer
{
	struct RenderContext
	{
		unsigned int screen_width_px, screen_height_px;
		Shader shader;
		glm::mat4 viewMatrix;
		glm::mat4 projectionMatrix;
        bwrenderer::TextureCache texture_cache;
	};

    using RENDER = void(Shader& shader);

	class RendererObject
	{
		virtual RENDER render = 0;
	};

	class TexturedCubeMesh : RendererObject
	{
	public:
		TexturedCubeMesh();

        TexturedCubeMesh(TextureBuffer* buffer, glm::vec3 worldPos = glm::vec3(0.0, 0.0, 0.0));

        inline void attachTexture(TextureBuffer* textBuffer) {
            texture = textBuffer;
        }

        void render(Shader& shader);

	private:
        GLuint vertex_buffer, index_buffer, vertex_array;
		const TextureBuffer* texture;
        glm::vec3 worldPosition;
    private:
        void buildNewBuffer();
	public:
		inline static const float vertex_data[] =
		{
        -1.0f, -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f, -1.0f,  1.0f, 1.0f,
         1.0f,  1.0f, -1.0f,  1.0f, 1.0f,
        -1.0f,  1.0f, -1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f, -1.0f,  0.0f, 0.0f,

        -1.0f, -1.0f,  1.0f,  1.0f, 0.0f,
         1.0f, -1.0f,  1.0f,  2.0f, 0.0f,
         1.0f,  1.0f,  1.0f,  2.0f, 1.0f,
         1.0f,  1.0f,  1.0f,  2.0f, 1.0f,
        -1.0f,  1.0f,  1.0f,  1.0f, 1.0f,
        -1.0f, -1.0f,  1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f,  1.0f,  3.0f, 0.0f,
        -1.0f,  1.0f, -1.0f,  3.0f, 1.0f,
        -1.0f, -1.0f, -1.0f,  2.0f, 1.0f,
        -1.0f, -1.0f, -1.0f,  2.0f, 1.0f,
        -1.0f, -1.0f,  1.0f,  2.0f, 0.0f,
        -1.0f,  1.0f,  1.0f,  3.0f, 0.0f,

         1.0f,  1.0f,  1.0f,  4.0f, 0.0f,
         1.0f,  1.0f, -1.0f,  4.0f, 1.0f,
         1.0f, -1.0f, -1.0f,  3.0f, 1.0f,
         1.0f, -1.0f, -1.0f,  3.0f, 1.0f,
         1.0f, -1.0f,  1.0f,  3.0f, 0.0f,
         1.0f,  1.0f,  1.0f,  4.0f, 0.0f,

        -1.0f, -1.0f, -1.0f,  4.0f, 1.0f,
         1.0f, -1.0f, -1.0f,  5.0f, 1.0f,
         1.0f, -1.0f,  1.0f,  5.0f, 0.0f,
         1.0f, -1.0f,  1.0f,  5.0f, 0.0f,
        -1.0f, -1.0f,  1.0f,  4.0f, 0.0f,
        -1.0f, -1.0f, -1.0f,  4.0f, 1.0f,

        -1.0f,  1.0f, -1.0f,  0.0f, 1.0f,
         1.0f,  1.0f, -1.0f,  1.0f, 1.0f,
         1.0f,  1.0f,  1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f,  1.0f, 0.0f,
        -1.0f,  1.0f,  1.0f,  0.0f, 0.0f,
        -1.0f,  1.0f, -1.0f,  0.0f, 1.0f
		};

	};

    class TexturedNormalCubeMesh : RendererObject
    {
    public:
        TexturedNormalCubeMesh();

        TexturedNormalCubeMesh(TextureBuffer* buffer, glm::vec3 worldPos = glm::vec3(0.0, 0.0, 0.0));

        inline void attachTexture(TextureBuffer* textBuffer) {
            texture = textBuffer;
        }

        void render(Shader& shader);

    private:
        GLuint vertex_buffer, index_buffer, vertex_array;
        const TextureBuffer* texture;
        glm::vec3 worldPosition;
    private:
        void buildNewBuffer();
    public:
        inline static const float vertex_data[] =
        {
        0.0f, 0.0f, 0.0f,  1.0f, 0.0f, 0.0f,  1.0f, 0.0f,
        0.0f, 0.0f, 0.0f,  -1.0f, 0.0f, 0.0f,  1.0f, 0.0f,
        0.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f,  1.0f, 0.0f,
        0.0f, 0.0f, 0.0f,  0.0f, -1.0f, 0.0f,  1.0f, 0.0f,
        0.0f, 0.0f, 0.0f,  0.0f, 0.0f, 1.0f,  1.0f, 0.0f,
        0.0f, 0.0f, 0.0f,  0.0f, 0.0f, -1.0f,  1.0f, 0.0f,

        0.0f, 1.0f, 0.0f,  -1.0f, 0.0f, 0.0f,  3.0f, 0.0f,
        
        1.0f, 1.0f, 0.0f,  0.0f, 1.0f, 0.0f,  2.0f, 0.0f,

        0.0f, 1.0f, 1.0f,  0.0f, 1.0f, 0.0f,  4.0f, 0.0f,

        1.0f, 1.0f, 1.0f,  0.0f, 1.0f, 0.0f,  2.0f, 0.0f,
        
        2.0f, 1.0f, 2.0f,  1.0f, 0.0f, 0.0f,  1.0f, 0.0f,

        2.0f, 3.0f, 2.0f,  0.0f, -1.0f, 0.0f,  0.0f, 0.0f,

        2.0f, 3.0f, 2.0f,  -1.0f, 0.0f, 0.0f,  2.0f, 0.0f,

        3.0f, 3.0f, 2.0f,  -1.0f, 0.0f, 0.0f,  2.0f, 0.0f,

        4.0f, 3.0f, 2.0f,  -1.0f, 0.0f, 0.0f,  2.0f, 0.0f,

        2.0f, 3.0f, 3.0f,  0.0f, -1.0f, 0.0f,  1.0f, 0.0f,

        2.0f, 3.0f, 4.0f,  -1.0f, 0.0f, 0.0f,  4.0f, 0.0f,

        3.0f, 3.0f, 5.0f,  -1.0f, 0.0f, 0.0f, 0.0f, 0.0f,

        4.0f, 3.0f, 6.0f,  -1.0f, 0.0f, 0.0f,  2.0f, 0.0f,

        };

    };


}