#pragma once

#include "Debug.hpp"
#include "Buffers.hpp"

#include <vector>

namespace bwrenderer
{
    struct vertex_attribute {
        GLenum type;
        GLuint size;
        GLboolean normalized;

        constexpr static GLuint getSizeOfType(const GLenum& type);
    };

    class vertex_layout {
    private:
        std::vector<vertex_attribute> attribute_list;
        GLuint stride = 0;
    public:
        vertex_layout() = default;
        void push(GLenum type, GLuint count, bool normalized = false);
        inline const std::vector<vertex_attribute>& elements() const { return attribute_list; }
        inline GLuint getStride() const { return stride; }
    };

    class vertex_array
    {
    public:
        vertex_array();
        vertex_array(const vertex_buffer& vb, const vertex_layout& layout);
        ~vertex_array();

        vertex_array(vertex_array&&) noexcept = default;
        vertex_array& operator= (vertex_array&&) noexcept = default;

        vertex_array(const vertex_array&) = default;
        vertex_array& operator= (const vertex_array&) = default;

        void bindWithAttribs() const;

        inline void bind() const
        {
            glBindVertexArray(vao);
        }

        inline void unbind() const
        {
            glBindVertexArray(0);
        }

        inline void enableVertexAttribArray(GLuint attrib_location, bool no_bind = false) const
        {
            GL_ASSERT(attrib_location < attrib_count, "Attribute overflow.");
            bind();
            glEnableVertexAttribArray(attrib_location);
            if (no_bind) unbind();
        }

        inline void disableVertexAttribArray(GLuint attrib_location, bool no_bind = false) const
        {
            GL_ASSERT(attrib_location < attrib_count, "Attribute overflow.");
            bind();
            glDisableVertexAttribArray(attrib_location);
            if (no_bind) unbind();
        }

    private:
        GLuint vao = 0;
        GLuint attrib_count = 0;
    private:
        void attachBuffer(const vertex_buffer& vb, const vertex_layout& layout);

    };

}