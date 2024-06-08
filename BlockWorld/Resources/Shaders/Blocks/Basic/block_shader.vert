#version 330 core

layout(location=0) in vec3 position;
layout(location=1) in vec2 textureCoords;

out VS_OUT { 
	vec2 texCoords;
} vs_out;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform vec2 image_size;


void main()
{
	gl_Position = projection * view * model * vec4(position, 1.0);

	vs_out.texCoords = 16 * textureCoords / image_size;
}