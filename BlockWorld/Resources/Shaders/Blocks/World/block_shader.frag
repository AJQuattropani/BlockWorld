#version 330 core

in GS_OUT {
	vec2 f_texCoords;
	vec3 f_normal;
	vec4 f_pos_vs;
	vec4 f_pos_ls;
} fs_in;

out vec4 o_Color;

struct SunLight {
	vec3 ambient;
	vec3 diffuse;
	vec3 specular;

	vec3 position;

	float day_time;
	float radial_time;

	float environmental_brightness;
};

uniform sampler2D block_texture;
uniform sampler2D shadow_map;

uniform SunLight dir_light;
uniform mat4 view;


// todo make a uniform
const vec4 day_color = vec4(0.17, 0.55, 0.99, 1.0);
const vec4 night_color = vec4(0.17/10, 0.55/10, 0.99/10, 1.0);

const float density = 0.005;
const float gradient = 30.0;

const float shininess = 64.0;

const float gamma = 2.2;

vec3 calcDirectionalLight(SunLight light, vec3 normal, vec3 viewDir, vec3 direction, vec2 texCoords, float shadow);
vec4 addFog(vec3 fragColor, float fragDistance, float radialTime);
float shadowCalculation(vec4 fragPosLightSpace, vec3 normal, vec3 direction);

void main()
{
	vec3 norm = normalize(fs_in.f_normal);
	vec3 fragPos = fs_in.f_pos_vs.xyz;
	vec3 viewDir = normalize(-fragPos);
	float fragDistance = length(vec2(fragPos.xz));
	
	vec3 fragColor = vec3(0.0);

	float radialTime = dir_light.radial_time;
	vec3 direction = normalize(mat3(view) * dir_light.position);

	float shadow = shadowCalculation(fs_in.f_pos_ls, norm, direction);
	fragColor += calcDirectionalLight(dir_light, norm, viewDir, direction, fs_in.f_texCoords, shadow);

	o_Color = addFog(fragColor, fragDistance, radialTime);
	o_Color.rgb = pow(o_Color.rgb, vec3(1.0/gamma));

}

vec4 addFog(vec3 fragColor, float fragDistance, float radialTime)
{
	float visbility = clamp((exp(-pow(fragDistance*density,gradient))), 0.0, 1.0);
	float skyTransition = clamp(0.5 - 2.0 * sin(radialTime), 0.0, 1.0);
	vec4 skyColor = mix(day_color, night_color, skyTransition);
	float sunsetTransition = 1-sqrt(abs(sin(radialTime)));

	skyColor = mix(skyColor, vec4(1.0, 0.4, 0.3, 1.0), sunsetTransition);

	return mix(skyColor, vec4(fragColor, 1.0), visbility);
}

vec3 calcDirectionalLight(SunLight light, vec3 normal, vec3 viewDir, vec3 direction, vec2 texCoords, float shadow)
{	
	vec3 halfwayDir = normalize(direction + viewDir);
	
	// diffuse
	float diff = max(dot(normal, direction), 0.0);
	// specular
	//vec3 reflectDir = reflect(-lightDir, normal);
	float spec = pow(max(dot(viewDir, halfwayDir), 0.0), shininess);
	// combine

	vec3 color = vec3(texture(block_texture, texCoords));

	float brightness = light.environmental_brightness;

	vec3 ambient = light.ambient * brightness * color;
	vec3 diffuse = light.diffuse * diff * brightness * color;
	vec3 specular = light.specular * spec * color;

	return ambient + (1.0 - shadow) * (diffuse + specular);
	//return (1.0 - shadow) * (ambient + diffuse + specular);
}

float shadowCalculation(vec4 fragPosLightSpace, vec3 normal, vec3 direction)
{
	vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
	projCoords = projCoords * 0.5 + 0.5;
	float closestDepth = texture(shadow_map, projCoords.xy).r;
	float currentDepth = projCoords.z;

	float bias = max(0.005 * (1.0 - dot(normal, direction)), 0.001);
	//float shadow = currentDepth - bias > closestDepth ? 1.0 : 0.0;

	float shadow = 0.0;
	vec2 texelSize = 1.0 / textureSize(shadow_map, 0);
	const int sampling = 5;
	const float samplingNumber = (2.0*sampling+1.0);//*(2.0*sampling+1);
	for(int x = -sampling; x <= sampling; ++x)
	{
		//for(int y = -sampling; y <= sampling; ++y)
		//{
			float pcfDepth = texture(shadow_map, projCoords.xy + vec2(0, x) * texelSize).r; 
			shadow += currentDepth - bias > pcfDepth ? 1.0 : 0.0;        
		//}    
	}
	shadow /= samplingNumber;

	if(projCoords.z > 1.0) shadow = 0.0;

	return shadow;
}