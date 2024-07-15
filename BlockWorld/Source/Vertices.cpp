#include "Vertices.hpp"

using namespace bwrenderer;

void vertex_layout::push(GLenum type, GLuint count, bool normalized)
{
	attribute_list.emplace_back(type, count, normalized);
	stride += count * vertex_attribute::getSizeOfType(type);
}

constexpr GLuint vertex_attribute::getSizeOfType(const GLenum& type) {
	switch (type) {
	case GL_FLOAT: return sizeof(GLfloat);
	case GL_UNSIGNED_INT: return sizeof(GLuint);
	case GL_INT: return sizeof(GLint);
	case GL_UNSIGNED_BYTE: return sizeof(GLubyte);
	case GL_BYTE: return sizeof(GLbyte);
	default:
		GL_ASSERT(false, "Bad enum Type.");
		return 0;
	}
}

bwrenderer::vertex_array::vertex_array()
{
	glGenVertexArrays(1, &vao);
	GL_INFO("[Vertex Array] Array %d created.", vao);
}

vertex_array::vertex_array(const vertex_buffer& vb, const vertex_layout& layout)
{
	glGenVertexArrays(1, &vao);
	GL_INFO("[Vertex Array] Array %d created.", vao);
	attachBuffer(vb, layout);
}

void vertex_array::attachBuffer(const vertex_buffer& vb, const vertex_layout& layout)
{
	bind();
	vb.bind();
	const auto& attributes = layout.elements();
	GLuint offset = 0;
	for (int i = 0; i < attributes.size(); i++)
	{
		const auto& attribute = attributes[i];
		glVertexAttribPointer(i, attribute.size, attribute.type, attribute.normalized, layout.getStride(), (const void*)offset);
		glEnableVertexAttribArray(i);
		offset += attribute.size * vertex_attribute::getSizeOfType(attribute.type);
	}
	GL_INFO("[Vertex Array] Buffer made for %d.", vao);
}

vertex_array::~vertex_array()
{
	GL_INFO("[Vertex Array] Buffer %d deleted.", vao);
	GL_TRY(glDeleteVertexArrays(1, &vao));
}

void vertex_array::bindWithAttribs() const
{
	bind();
	for (size_t i = 0; i < attrib_count; i++)
	{
		glEnableVertexAttribArray(i);
	}
	GL_INFO("[Vertex Array] Attributes enabled.", vao);
}