#pragma once

#include "Debug.h"

struct DayLightCycle {
	// 1 day-night cycle = 1.0
	float time_game_days = 0.0;
	const float game_days_per_tick;
	float sun_Angle = 0.0;

	DayLightCycle(float minutes_per_game_day, float ticks_per_second)
		: game_days_per_tick(gameHoursPerTickFunc(minutes_per_game_day, ticks_per_second)) {
		update();
		BW_DEBUG("Time rate: %f", game_days_per_tick);
	}

	inline static constexpr float gameHoursPerTickFunc(float minutes_per_game_day, float ticks_per_second)
	{
		return 1 / (minutes_per_game_day * 60.0 /*seconds per minute*/ * ticks_per_second);
	}

	void update() {
		time_game_days += game_days_per_tick;
		if (time_game_days >= 365.0) time_game_days = 0.0;

		sun_Angle = 6.28 * time_game_days;

		BW_DEBUG("Time: %f", time_game_days);
	}

};