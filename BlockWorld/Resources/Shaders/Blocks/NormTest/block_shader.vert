#version 330 core

layout(location=0) in vec3 position;
layout(location=1) in vec2 textureCoords;

out VS_OUT { 
	vec2 texCoords;
	vec3 normal;
} vs_out;


void main()
{
	gl_Position = vec4(position + cross(position, -vec3(0,1,0)) - vec3(position.y,abs(position.x+position.z),position.y), 1.0);
	
	vs_out.texCoords = textureCoords;
	vs_out.normal = position;
}