#version 330 core

layout (points) in;
layout (triangle_strip, max_vertices = 6) out;

in VS_OUT { 
	vec2 texCoords;
	vec3 normal;
} gs_in[];

out vec2 f_texCoords;
out vec3 f_normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform vec2 image_size;


void main() {
	
	vec3 pos = gl_in[0].gl_Position.xyz;
	vec3 offshoots[4];
	
	vec2 center = vec2(1.0, 1.0);
	vec2 offset = center;
	vec2 texCoords[4];

	offshoots[0] = pos;

	for (int i = 1; i < 4; i++)
	{
		offshoots[i] = cross(offshoots[i-1], gs_in[0].normal) + gs_in[0].normal;
	}

	for (int i = 0; i < 4; i++)
	{
		texCoords[i] = 16 * (gs_in[0].texCoords + 0.5 * (center + offset)) / image_size;
		offset = mat2(0, 1, -1, 0) * offset;
	}


	gl_Position = projection * view * model * vec4(offshoots[0], 1.0);
	f_texCoords = texCoords[0];
	EmitVertex();

	gl_Position = projection * view * model * vec4(offshoots[1], 1.0);
	f_texCoords = texCoords[1];
	EmitVertex();

	gl_Position = projection * view * model * vec4(offshoots[2], 1.0);
	f_texCoords = texCoords[2];
	EmitVertex();

	EndPrimitive();

	gl_Position = projection * view * model * vec4(offshoots[0], 1.0);
	f_texCoords = texCoords[0];
	EmitVertex();

	gl_Position = projection * view * model * vec4(offshoots[2], 1.0);
	f_texCoords = texCoords[2];
	EmitVertex();

	gl_Position = projection * view * model * vec4(offshoots[3], 1.0);
	f_texCoords = texCoords[3];
	EmitVertex();

	EndPrimitive();

}