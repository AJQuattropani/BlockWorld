#version 330 core

layout (points) in;
layout (triangle_strip, max_vertices = 6) out;

in VS_OUT { 
	vec3 normal;
} gs_in[];

uniform mat4 model;
uniform mat4 lightSpaceMatrix;

void main()
{

	vec3 pos = gl_in[0].gl_Position.xyz;
	vec3 normal = gs_in[0].normal;
	
	vec3 offshoots[4];

	offshoots[0] = normal + cross(normal, -vec3(0,1,0)) - vec3(normal.y,abs(normal.x+normal.z),normal.y);

	for (int i = 1; i < 4; i++)
	{
		offshoots[i] = cross(normal, offshoots[i-1]) + normal;
	}

	gl_Position = lightSpaceMatrix * model * vec4(pos + offshoots[0]/2.0, 1.0);
	EmitVertex();
	
	gl_Position = lightSpaceMatrix * model * vec4(pos + offshoots[1]/2.0, 1.0);
	EmitVertex();

	gl_Position = lightSpaceMatrix * model * vec4(pos + offshoots[2]/2.0, 1.0);
	EmitVertex();

	EndPrimitive();
	
	gl_Position = lightSpaceMatrix * model * vec4(pos + offshoots[0]/2.0, 1.0);
	EmitVertex();

	gl_Position = lightSpaceMatrix * model * vec4(pos + offshoots[2]/2.0, 1.0);
	EmitVertex();

	gl_Position = lightSpaceMatrix * model * vec4(pos + offshoots[3]/2.0, 1.0);
	EmitVertex();

	EndPrimitive();


}