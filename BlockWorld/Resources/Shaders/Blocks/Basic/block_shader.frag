#version 330 core

in vec2 f_texCoords;

out vec4 o_Color;

uniform sampler2D block_texture;

void main()
{
	o_Color = texture(block_texture, f_texCoords);
}