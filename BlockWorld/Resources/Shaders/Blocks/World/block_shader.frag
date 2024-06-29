#version 330 core

in vec2 f_texCoords;
in vec3 f_normal;
in vec4 f_pos;

out vec4 o_Color;

struct DirectionalLight {
	vec3 ambient;
	vec3 diffuse;
	vec3 specular;
};

uniform sampler2D block_texture;
uniform DirectionalLight dir_light;
uniform mat4 view;

uniform float day_time;

// todo make a uniform
const vec4 day_color = vec4(0.17, 0.55, 0.99, 1.0);
const vec4 night_color = vec4(0.17/10, 0.55/10, 0.99/10, 1.0);

const float density = 0.07;
const float gradient = 30.0;
const float shininess = 32.0;

vec3 calcDirectionalLight(DirectionalLight light, vec3 normal, vec3 viewDir, vec3 direction);

void main()
{
	float radial_time = 2.0 * 3.1416 * day_time;

	vec3 result = vec3(0.0);

	vec3 norm = normalize(f_normal);
	vec3 fragPos = f_pos.xyz;
	vec3 viewDir = normalize(-fragPos);

	vec3 direction = -vec3(cos(radial_time), sin(radial_time), 0.0);

	result += calcDirectionalLight(dir_light, norm, viewDir, direction);

	float f_distance = length(fragPos.xz);
	float visibility = clamp(exp(-pow(f_distance*density,gradient)), 0.0, 1.0);

	float sky_transition = sin(radial_time / 2) * sin(radial_time / 2);
	vec4 skyColor = mix(day_color, night_color, sky_transition);
	o_Color = mix(skyColor, vec4(result, 1.0), visibility);
}

vec3 calcDirectionalLight(DirectionalLight light, vec3 normal, vec3 viewDir, vec3 direction)
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
