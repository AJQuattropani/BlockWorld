#pragma once

#include <GL/glew.h>

#include "Debug.h"

namespace bwrenderer {

    class vertex_buffer
    {
    public:
        vertex_buffer();
        vertex_buffer(GLfloat* data, GLsizeiptr size, GLenum usage = GL_STATIC_DRAW);
        ~vertex_buffer();
        vertex_buffer(vertex_buffer&& other) noexcept;
        vertex_buffer& operator= (vertex_buffer&& other) noexcept;
        vertex_buffer(const vertex_buffer& other);
        vertex_buffer& operator= (const vertex_buffer& other);

        inline void bind() const
        {
            GL_ASSERT(buff_address != 0, "Bound buffer is uninitialized.");
            glBindBuffer(GL_ARRAY_BUFFER, buff_address);
            GL_INFO("[Vertex Buffer] %d bound.", buff_address);
        }
        inline void unbind() const
        {
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            GL_INFO("[Vertex Buffer] unbound.");
        }

        inline void attachBuffer(const GLfloat* data, GLsizeiptr size, GLenum usage = GL_STATIC_DRAW)
        {
            GL_INFO("[Vertex Buffer] Buffer attached.");
            glBufferData(GL_ARRAY_BUFFER, size, data, usage);
            byte_size = size;
        }

        inline GLuint getSizeBytes() const { return byte_size; }


    private:
        GLuint buff_address = 0;
        GLuint byte_size = 0;
        size_t* control_head = nullptr;
    };

    class index_buffer
    {
    public:
        index_buffer() = default;
        index_buffer(const GLuint* data, GLsizeiptr size, GLenum usage = GL_STATIC_DRAW);
        ~index_buffer();
        index_buffer(index_buffer&& other) noexcept;
        index_buffer& operator= (index_buffer&& other) noexcept;
        index_buffer(const index_buffer& other);
        index_buffer& operator= (const index_buffer& other);

        inline void bind() const
        {
            GL_ASSERT(buff_address != 0, "Bound buffer is uninitialized.");
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buff_address);
            GL_INFO("[Index Buffer] %d bound.", buff_address);
        }
        inline void unbind() const
        {
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
            GL_INFO("[Index Buffer] unbound.");
        }

        inline void attachBuffer(const GLuint* data, GLsizeiptr size, GLenum usage = GL_STATIC_DRAW)
        {
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, data, usage);
            byte_size = size;
        }

        inline GLuint getSizeBytes() const { return byte_size; }
    private:
        GLuint buff_address = 0;
        GLuint byte_size = 0;
        size_t* control_head = nullptr;
    };



}