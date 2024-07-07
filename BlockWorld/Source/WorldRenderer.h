#pragma once

#include <glm/glm.hpp>
#include <glm/vec3.hpp>

#include "DayCycle.h"
#include "Skybox.h"

namespace bwrenderer {

	inline void setDaylightShaderInfo(bwrenderer::Shader& block_shader, const bwrenderer::Shader& shadow_shader, const DayLightCycle& dayLightCycle) {
		
		//float ambience = 0.2 + 0.1 * glm::sin(dayLightCycle.sun_Angle);

	}

	inline void setSunShaderInfo(bwrenderer::Shader& sky_shader, const DayLightCycle& dayLightCycle) {
		sky_shader.bind();

		sky_shader.setUniformMat4f("rotation", glm::rotate(glm::mat4(1.0), dayLightCycle.sun_Angle, glm::vec3(0.0, 0.0, 1.0)));


	}


}