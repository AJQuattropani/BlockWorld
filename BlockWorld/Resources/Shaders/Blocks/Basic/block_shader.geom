#version 330 core

layout (triangles) in;
layout (triangle_strip, max_vertices = 3) out;

in VS_OUT { 
	vec2 texCoords;
} gs_in[];

out vec2 f_texCoords;

vec4 convertToGLSpace();

void main() {
//	gl_Position = projection * view * model * gl_in[0].gl_Position;
//	EmitVertex();
//	gl_Position = projection * view * model * gl_in[1].gl_Position;
//	EmitVertex();
//	gl_Position = projection * view * model * gl_in[2].gl_Position;
//	EmitVertex();
//
	gl_Position = gl_in[0].gl_Position;
	f_texCoords = gs_in[0].texCoords;
	EmitVertex();
	gl_Position = gl_in[1].gl_Position;
	f_texCoords = gs_in[1].texCoords;
	EmitVertex();
	gl_Position = gl_in[2].gl_Position;
	f_texCoords = gs_in[2].texCoords;
	EmitVertex();

	EndPrimitive();
}