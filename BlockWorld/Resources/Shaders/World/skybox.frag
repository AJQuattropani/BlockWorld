#version 330 core
out vec4 o_Color;

in vec2 TexCoords;

uniform sampler2D skyBox;

void main()
{
	o_Color = texture(skyBox, TexCoords);
}