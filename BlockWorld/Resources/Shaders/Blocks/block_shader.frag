#version 330 core

in vec3 color;
in vec2 texCoords;

out vec4 o_Color;

uniform sampler2D block_texture;

void main()
{
	o_Color = texture(block_texture, texCoords);
}