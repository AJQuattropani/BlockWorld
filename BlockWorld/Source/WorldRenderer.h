#pragma once

#include <glm/glm.hpp>
#include <glm/vec3.hpp>

#include "DayCycle.h"
#include "Shader.h"
#include "RenderContext.h"

namespace bwrenderer {

	inline void setSunShaderInfo(bwrenderer::RenderContext& context, const DayLightCycle& dayLightCycle) {
		context.shader.bind();

		//float ambience = 0.2 + 0.1 * glm::sin(dayLightCycle.sun_Angle);

		context.shader.setUniform3f("dir_light.ambient", 0.3f, 0.3f, 0.3f);
		context.shader.setUniform3f("dir_light.diffuse", 0.5f, 0.5f, 0.5f);
		context.shader.setUniform3f("dir_light.specular", 0.2f, 0.2f, 0.2f);

		context.shader.setUniform1f("day_time", dayLightCycle.time_game_days);
	}
}