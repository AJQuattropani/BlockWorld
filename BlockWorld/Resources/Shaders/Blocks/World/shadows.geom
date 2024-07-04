#version 330 core

layout (points) in;
layout (triangle_strip, max_vertices = 6) out;

in VS_OUT { 
	vec3 normal;
} gs_in[];

uniform mat4 model;
uniform mat4 lightSpaceMatrix;

uniform float chunk_width;

void main()
{

	vec3 pos = vec3(gl_in[0].gl_Position.xyz);
	vec3 normal = gs_in[0].normal;
	
	vec3 offshoots[4];

	offshoots[0] = normal + cross(normal, -vec3(0,1,0)) - vec3(normal.y,abs(normal.x+normal.z),normal.y);

	for (int i = 1; i < 4; i++)
	{
		offshoots[i] = cross(gs_in[0].normal, offshoots[i-1]) + gs_in[0].normal;
	}

	gl_Position = lightSpaceMatrix * model * (vec4(pos + offshoots[0]/2, 1.0)/chunk_width);
	EmitVertex();
	
	gl_Position = lightSpaceMatrix * model * (vec4(pos + offshoots[1]/2, 1.0)/chunk_width);
	EmitVertex();

	gl_Position = lightSpaceMatrix * model * (vec4(pos + offshoots[2]/2, 1.0)/chunk_width);
	EmitVertex();

	EndPrimitive();
	
	gl_Position = lightSpaceMatrix * model * (vec4(pos + offshoots[0]/2, 1.0)/chunk_width);
	EmitVertex();

	gl_Position = lightSpaceMatrix * model * (vec4(pos + offshoots[2]/2, 1.0)/chunk_width);
	EmitVertex();

	gl_Position = lightSpaceMatrix * model * (vec4(pos + offshoots[3]/2, 1.0)/chunk_width);
	EmitVertex();

	EndPrimitive();


}