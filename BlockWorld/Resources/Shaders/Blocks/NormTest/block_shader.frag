#version 330 core

in vec2 f_texCoords;
in vec3 f_normal;

out vec4 o_Color;

uniform sampler2D block_texture;

void main()
{
	o_Color = texture(block_texture, f_texCoords);
}