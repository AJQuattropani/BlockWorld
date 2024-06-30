#version 330 core
out vec4 o_Color;

in vec3 TexCoords;

uniform samplerCube skyBox;

void main()
{
	o_Color = texture(skyBox, TexCoords);
}