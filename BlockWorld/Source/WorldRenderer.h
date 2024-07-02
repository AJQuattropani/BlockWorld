#pragma once

#include <glm/glm.hpp>
#include <glm/vec3.hpp>

#include "DayCycle.h"
#include "Skybox.h"

namespace bwrenderer {

	inline void setDaylightShaderInfo(bwrenderer::Shader& block_shader, const DayLightCycle& dayLightCycle) {
		block_shader.bind();

		//float ambience = 0.2 + 0.1 * glm::sin(dayLightCycle.sun_Angle);

		block_shader.setUniform3f("dir_light.ambient", 0.5f, 0.5f, 0.5f);
		block_shader.setUniform3f("dir_light.diffuse", 0.5f, 0.5f, 0.5f);
		block_shader.setUniform3f("dir_light.specular", 0.2f, 0.2f, 0.2f);

		block_shader.setUniform1f("dir_light.day_time", dayLightCycle.time_game_days);
	}

	inline void setSunShaderInfo(bwrenderer::Shader& sky_shader, const DayLightCycle& dayLightCycle) {
		sky_shader.bind();

		sky_shader.setUniformMat4f("rotation", glm::rotate(glm::mat4(1.0), dayLightCycle.sun_Angle, glm::vec3(0.0, 0.0, 1.0)));


	}


}