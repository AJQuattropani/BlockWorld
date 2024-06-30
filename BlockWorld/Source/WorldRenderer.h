#pragma once

#include <glm/glm.hpp>
#include <glm/vec3.hpp>

#include "DayCycle.h"
#include "Skybox.h"

namespace bwrenderer {

	inline void setSunShaderInfo(bwrenderer::Shader& shader, const DayLightCycle& dayLightCycle) {
		shader.bind();

		//float ambience = 0.2 + 0.1 * glm::sin(dayLightCycle.sun_Angle);

		shader.setUniform3f("dir_light.ambient", 0.3f, 0.3f, 0.3f);
		shader.setUniform3f("dir_light.diffuse", 0.5f, 0.5f, 0.5f);
		shader.setUniform3f("dir_light.specular", 0.2f, 0.2f, 0.2f);

		shader.setUniform1f("dir_light.day_time", dayLightCycle.time_game_days);
	}

	


}