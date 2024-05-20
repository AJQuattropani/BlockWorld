#version 330 core

layout(location=0) in vec3 position;


uniform vec3 inColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 color;

void main()
{
	gl_Position = projection * view * model * vec4(position, 1.0);

	color = inColor;
}