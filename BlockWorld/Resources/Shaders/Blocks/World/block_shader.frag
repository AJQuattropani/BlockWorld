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

	float day_time;
};

uniform sampler2D block_texture;
uniform sampler2D shadow_map;

uniform SunLight dir_light;
uniform mat4 view;


// todo make a uniform
const vec4 day_color = vec4(0.17, 0.55, 0.99, 1.0);
const vec4 night_color = vec4(0.17/10, 0.55/10, 0.99/10, 1.0);

const float density = 0.07;
const float gradient = 30.0;
const float shininess = 64.0;

const float daylight_to_night = 3.0;

const float gamma = 2.2;

float calcRadialTime(float dayTime);
vec3 calcDirection(float radialTime);
vec3 calcDirectionalLight(SunLight light, vec3 normal, vec3 viewDir, vec3 direction, float radialTime, vec2 texCoords, float shadow);
vec4 addFog(vec3 fragColor, float fragDistance, float radialTime);
float shadowCalculation(vec4 fragPosLightSpace, vec3 normal, vec3 lightDir);

void main()
{
	vec3 norm = normalize(fs_in.f_normal);
	vec3 fragPos = fs_in.f_pos_vs.xyz;
	vec3 viewDir = normalize(-fragPos);
	float fragDistance = length(fragPos.xz);
	
	vec3 fragColor = vec3(0.0);

	float radialTime = calcRadialTime(dir_light.day_time);
	vec3 direction = calcDirection(mod(radialTime, 3.1416));

	float shadow = shadowCalculation(fs_in.f_pos_ls, norm, direction);
	fragColor += calcDirectionalLight(dir_light, norm, viewDir, direction, radialTime, fs_in.f_texCoords, shadow);

	o_Color = addFog(fragColor, fragDistance, radialTime);
	o_Color.rgb = pow(o_Color.rgb, vec3(1.0/gamma));

}

float calcRadialTime(float dayTime)
{
	return 2.0 * 3.1416 * dayTime;
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

vec3 calcDirection(float radialTime)
{
	return -normalize(vec3(cos(radialTime), sin(radialTime), 0.0));
}

vec3 calcDirectionalLight(SunLight light, vec3 normal, vec3 viewDir, vec3 direction, float radialTime, vec2 texCoords, float shadow)
{
	float amb = (daylight_to_night * sqrt(abs(sin(radialTime))) + sin(radialTime) + pow(cos(radialTime), 3)) / (1.0 + daylight_to_night);
	
	vec3 lightDir = normalize(-(mat3(view) * direction));
	vec3 halfwayDir = normalize(lightDir + viewDir);
	
	// diffuse
	float diff = max(dot(normal, lightDir), 0.0);
	// specular
	//vec3 reflectDir = reflect(-lightDir, normal);
	float spec = pow(max(dot(viewDir, halfwayDir), 0.0), shininess);
	// combine
	vec3 ambient = light.ambient * amb * vec3(texture(block_texture, texCoords));
	vec3 diffuse = light.diffuse * diff * amb * vec3(texture(block_texture, texCoords));
	vec3 specular = light.specular * spec * vec3(texture(block_texture, texCoords));

	return (ambient + (1.0 - shadow) * (diffuse + specular));
}

float shadowCalculation(vec4 fragPosLightSpace, vec3 normal, vec3 lightDir)
{
	vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
	projCoords = projCoords * 0.5 + 0.5;
	float closestDepth = texture(shadow_map, projCoords.xy).r;
	float currentDepth = projCoords.z;

	float bias = max(0.05 * (1.0 - dot(normal, lightDir)), 0.005);
	float shadow = currentDepth - bias > closestDepth ? 1.0 : 0.0;

	return shadow;
}