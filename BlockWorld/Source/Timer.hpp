#pragma once
#include "Debug.hpp"

#include <thread>

#ifndef DIST
#define TIME_FUNC(x) Timer bw_timer_macro = Timer(x)
#else
#define TIME_FUNC(x)
#endif

struct Timer
{
	std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
	const char* function_name;
	Timer(void)
	{
		function_name = "";
		startTime = std::chrono::high_resolution_clock::now();
	}

	Timer(const char* name)
	{
		function_name = name;
		startTime = std::chrono::high_resolution_clock::now();
	}

	~Timer()
	{
		Stop();
	}

	inline void Stop()
	{
		using namespace std::chrono;
		int64_t duration = time_point_cast<microseconds>(high_resolution_clock::now()).time_since_epoch().count()
			- time_point_cast<microseconds>(startTime).time_since_epoch().count();
		BW_DEBUG("<%s> took %Lf ms", function_name, duration * 0.001);
	}
};