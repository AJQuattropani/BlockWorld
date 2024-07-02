#version 330 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec2 tex_coords;

out vec2 TexCoords;

uniform mat4 rotation;
uniform mat4 projection;
uniform mat4 view;

void main()
{
	TexCoords = tex_coords / vec2(2.0, 1.0);
	vec4 pos = projection * mat4(mat3(view)) * mat4(mat3(rotation)) * (vec4(position, 1.0) * vec4(1.0f, 0.125f, 0.125f, 1.0f));
	gl_Position = pos.xyww;
}