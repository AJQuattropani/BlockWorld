#include "Buffers.h"

using namespace bwrenderer;

vertex_buffer::vertex_buffer()
{
	glGenBuffers(1, &buff_address);
	byte_size = 0;
	bind();
	GL_INFO("[Vertex Buffer] %d created.", buff_address);

	control_head = new size_t(1);
}

vertex_buffer::vertex_buffer(GLfloat* data, GLsizeiptr size, GLenum usage)
{
	glGenBuffers(1, &buff_address);
	byte_size = size;
	bind();
	GL_INFO("[Vertex Buffer] %d created", buff_address);
	attachBuffer(data, size, usage);

	control_head = new size_t(1);


}

vertex_buffer::~vertex_buffer()
{
	if (--(*control_head) == 0)
	{
		glDeleteBuffers(1, &buff_address);
		GL_INFO("[Vertex Buffer] Buffer %d destroyed.", buff_address);
		delete control_head;
	}
}

vertex_buffer::vertex_buffer(vertex_buffer&& other) noexcept
	: buff_address(other.buff_address), byte_size(other.byte_size), control_head(other.control_head)
{
	other.buff_address = 0;
}

vertex_buffer& vertex_buffer::operator= (vertex_buffer&& other) noexcept
{
	this->~vertex_buffer();
	new (this) vertex_buffer(other);
	return *this;
}

vertex_buffer::vertex_buffer(const vertex_buffer& other)
	: buff_address(other.buff_address), byte_size(other.byte_size), control_head(other.control_head)
{
	(*control_head)++;
}

vertex_buffer& vertex_buffer::operator= (const vertex_buffer& other)
{
	this->~vertex_buffer();
	new (this) vertex_buffer(other);
	return *this;
}

index_buffer::index_buffer(const GLuint* data, GLsizeiptr size, GLenum usage)
{
	glGenBuffers(1, &buff_address);
	byte_size = size;
	bind();
	attachBuffer(data, size, usage);

	control_head = new size_t(1);

	GL_INFO("[Index Buffer] %d created", buff_address);

}

index_buffer::~index_buffer()
{
	if (--(*control_head) == 0)
	{
		glDeleteBuffers(1, &buff_address);
		GL_INFO("[Index Buffer] %d destroyed.", buff_address);
		delete control_head;
	}
}

index_buffer::index_buffer(index_buffer&& other) noexcept
	: buff_address(other.buff_address), byte_size(other.byte_size), control_head(other.control_head)
{
	other.buff_address = 0;
}

index_buffer& index_buffer::operator= (index_buffer&& other) noexcept
{
	this->~index_buffer();
	new (this) index_buffer(other);
	return *this;
}

index_buffer::index_buffer(const index_buffer& other)
	: buff_address(other.buff_address), byte_size(other.byte_size), control_head(other.control_head)
{
	(*control_head)++;
}

index_buffer& index_buffer::operator= (const index_buffer& other)
{
	this->~index_buffer();
	new (this) index_buffer(other);
	return *this;
}