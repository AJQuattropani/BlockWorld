#version 330 core

in vec2 f_texCoords;
in vec3 f_normal;
in vec4 f_pos;

out vec4 o_Color;

struct SunLight {
	vec3 ambient;
	vec3 diffuse;
	vec3 specular;

	float day_time;
};

uniform sampler2D block_texture;
uniform SunLight dir_light;
uniform mat4 view;


// todo make a uniform
const vec4 day_color = vec4(0.17, 0.55, 0.99, 1.0);
const vec4 night_color = vec4(0.17/10, 0.55/10, 0.99/10, 1.0);

const float density = 0.07;
const float gradient = 30.0;
const float shininess = 128.0;

float calcRadialTime(float dayTime);
vec3 calcDirection(float radialTime);
vec3 calcDirectionalLight(SunLight light, vec3 normal, vec3 viewDir, vec3 direction, float radialTime);
vec4 addFog(vec3 fragColor, float fragDistance, float radialTime);

void main()
{
	vec3 norm = normalize(f_normal);
	vec3 fragPos = f_pos.xyz;
	vec3 viewDir = normalize(-fragPos);
	float fragDistance = length(fragPos.xz);
	
	vec3 fragColor = vec3(0.0);

	float radialTime = calcRadialTime(dir_light.day_time);
	vec3 direction = calcDirection(radialTime);
	fragColor += calcDirectionalLight(dir_light, norm, viewDir, direction, radialTime);

	o_Color = addFog(fragColor, fragDistance, radialTime);
}

float calcRadialTime(float dayTime)
{
	return 2.0 * 3.1416 * dayTime;
}

vec4 addFog(vec3 fragColor, float fragDistance, float radialTime)
{
	float visbility = clamp((exp(-pow(fragDistance*density,gradient))), 0.0, 1.0);
	float skyTransition = sin(radialTime / 2) * sin(radialTime / 2);
	vec4 skyColor = mix(day_color, night_color, skyTransition);

	return mix(skyColor, vec4(fragColor, 1.0), visbility);
}

vec3 calcDirection(float radialTime)
{
	return -normalize(vec3(cos(radialTime), sin(radialTime), 0.0));
}

vec3 calcDirectionalLight(SunLight light, vec3 normal, vec3 viewDir, vec3 direction, float radialTime)
{

	vec3 lightDir = normalize(-(mat3(view) * direction));
	// diffuse
	float diff = max(dot(normal, lightDir), 0.0);
	// specular
	vec3 reflectDir = reflect(-lightDir, normal);
	float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
	// combine
	vec3 ambient = light.ambient * vec3(texture(block_texture, f_texCoords));
	vec3 diffuse = light.diffuse * diff * vec3(texture(block_texture, f_texCoords));
	vec3 specular = light.specular * spec * vec3(texture(block_texture, f_texCoords));
	return (ambient + diffuse + specular);
}
