#pragma once
#include "Debug.h"

#include <thread>

#ifndef DIST
#define TIME_FUNC(x) Timer bw_timer_macro = Timer(x)
#else
#define TIME_FUNC(x)
#endif

struct Timer
{
	std::chrono::time_point<std::chrono::steady_clock> startTime;
	const char* functionName;
	Timer(void)
	{
		functionName = "";
		startTime = std::chrono::high_resolution_clock::now();
	}

	Timer(const char* name)
	{
		functionName = name;
		startTime = std::chrono::high_resolution_clock::now();
	}

	~Timer()
	{
		std::chrono::duration<float> duration = std::chrono::high_resolution_clock::now() - startTime;
		BW_DEBUG("<%s> took %Lf ms", functionName, duration.count() * 1000.0);
	}
};