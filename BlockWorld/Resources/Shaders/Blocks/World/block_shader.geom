#version 330 core

layout (points) in;
layout (triangle_strip, max_vertices = 6) out;

in VS_OUT { 
	vec3 normal;
	vec2 texCoords;
} gs_in[];

out vec2 f_texCoords;
out vec3 f_normal;
out vec4 f_pos;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform vec2 image_size;
uniform float chunk_width;

void main() {
	
	vec3 pos = vec3(gl_in[0].gl_Position.xyz);
	vec3 normal = gs_in[0].normal;
	
	vec3 offshoots[4];

	offshoots[0] = normal + cross(normal, -vec3(0,1,0)) - vec3(normal.y,abs(normal.x+normal.z),normal.y);

	vec2 center = vec2(1.0, 1.0);
	// slight short-scaling is meant to prevent some minor texture overlap
	vec2 offset = 0.97 * center;
	vec2 texCoords[4];

	for (int i = 1; i < 4; i++)
	{
		offshoots[i] = cross(gs_in[0].normal, offshoots[i-1]) + gs_in[0].normal;
	}

	for (int i = 0; i < 4; i++)
	{
		offset = mat2(0, 1, -1, 0) * offset; 
		texCoords[i] = 16 * (gs_in[0].texCoords + 0.5 * (center + offset)) / image_size;
	}

	f_normal = mat3(transpose(inverse(view * model))) * normal;

	f_pos = view * model * (vec4(pos + offshoots[0]/2, 1.0)/chunk_width);
	gl_Position = projection * f_pos;
	f_texCoords = texCoords[0];
	EmitVertex();
	
	f_pos = view * model * (vec4(pos + offshoots[1]/2, 1.0)/chunk_width);
	gl_Position = projection * f_pos;
	f_texCoords = texCoords[1];
	EmitVertex();

	f_pos = view * model * (vec4(pos + offshoots[2]/2, 1.0)/chunk_width);
	gl_Position =  projection * f_pos;
	f_texCoords = texCoords[2];
	EmitVertex();

	EndPrimitive();
	
	f_pos = view * model * (vec4(pos + offshoots[0]/2, 1.0)/chunk_width);
	gl_Position = projection * f_pos;
	f_texCoords = texCoords[0];
	EmitVertex();

	f_pos = view * model * (vec4(pos + offshoots[2]/2, 1.0)/chunk_width);
	gl_Position = projection * f_pos;
	f_texCoords = texCoords[2];
	EmitVertex();

	f_pos = view * model * (vec4(pos + offshoots[3]/2, 1.0)/chunk_width);
	gl_Position = projection * f_pos;
	f_texCoords = texCoords[3];
	EmitVertex();

	EndPrimitive();

}