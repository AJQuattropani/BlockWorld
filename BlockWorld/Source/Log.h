#pragma once

#include <stdio.h>
#include <unordered_map>
#include <memory>
#include <string>

#define BW_DEFAULT_LOG_LEVEL Log_Level::INFO
#define GL_DEFAULT_LOG_LEVEL Log_Level::INFO

#ifdef BW_LOG_OUTPUT
#define BW_INFO(msg, ...) Log::getAppLog()->print(Log_Level::INFO, TextColor::WHITE, msg, ##__VA_ARGS__)
#define BW_WARN(msg, ...) Log::getAppLog()->print(Log_Level::WARN, TextColor::BRIGHT_YELLOW, msg, ##__VA_ARGS__)
#define BW_ERROR(msg, ...) Log::getAppLog()->print(Log_Level::ERROR, TextColor::BRIGHT_RED, msg, ##__VA_ARGS__)
#define BW_FATAL(msg, ...) Log::getAppLog()->print(Log_Level::FATAL, TextColor::RED, msg, ##__VA_ARGS__)
#define BW_LEVEL(x) Log::getAppLog()->setLevel(x);
#ifdef BW_DEBUGGING
#define BW_DEBUG(msg, ...) Log::getAppLog()->print(Log_Level::DEBUG, TextColor::GREEN, msg, ##__VA_ARGS__)
#else
#define BW_DEBUG(x)
#endif
#else
#define BW_INFO(x)
#define BW_WARN(x)
#define BW_ERROR(x)
#define BW_FATAL(x)
#define BW_DEBUG(x)
#define BW_LEVEL(x)
#endif

#ifdef GL_DEBUGGING
#define GL_INFO(msg, ...) Log::getGraphicsLog()->print(Log_Level::INFO, TextColor::WHITE, msg, ##__VA_ARGS__)
#define GL_WARN(msg, ...) Log::getGraphicsLog()->print(Log_Level::WARN, TextColor::BRIGHT_YELLOW, msg, ##__VA_ARGS__)
#define GL_ERROR(msg, ...) Log::getGraphicsLog()->print(Log_Level::ERROR, TextColor::BRIGHT_RED, msg, ##__VA_ARGS__)
#define GL_FATAL(msg, ...) Log::getGraphicsLog()->print(Log_Level::FATAL, TextColor::RED, msg, ##__VA_ARGS__)
#define GL_DEBUG(msg, ...) Log::getGraphicsLog()->print(Log_Level::DEBUG, TextColor::GREEN, msg, ##__VA_ARGS__)
#define GL_LEVEL(x) Log::getGraphicsLog()->setLevel(x)
#else
#define GL_INFO(x) 
#define GL_WARN(x) 
#define GL_ERROR(x)
#define GL_FATAL(x)
#define GL_DEBUG(x)
#define GL_LEVEL(x)
#endif

enum class Log_Level
{
	INFO = 0, WARN, ERROR, FATAL, DEBUG
};

enum class TextColor
{
	BLACK,
	RED,
	GREEN,
	YELLOW,
	BLUE,
	MAGENTA,
	CYAN,
	WHITE,
	BRIGHT_BLACK,
	BRIGHT_RED,
	BRIGHT_GREEN,
	BRIGHT_YELLOW,
	BRIGHT_BLUE,
	BRIGHT_MAGENTA,
	BRIGHT_CYAN,
	BRIGHT_WHITE,
	COUNT
};

class Log
{
public:
	inline static Log const* getAppLog() { return AppLog.get(); }
	inline static Log const* getGraphicsLog() { return GraphicsLog.get(); }
public:
	template <typename ...Args>
	void print(Log_Level msg_level, TextColor textColor, const char* message, Args... args) const
	{
		static const char* TextColorTable[(int)TextColor::COUNT] =
		{
			"\x1b[30m",
			"\x1b[31m",
			"\x1b[32m",
			"\x1b[33m",
			"\x1b[34m",
			"\x1b[35m",
			"\x1b[36m",
			"\x1b[37m",
			"\x1b[90m",
			"\x1b[91m",
			"\x1b[92m",
			"\x1b[93m",
			"\x1b[94m",
			"\x1b[95m",
			"\x1b[96m",
			"\x1b[97m",
		};
		if (msg_level < log_level) return;
		char* formatBuffer = new char[256*3];
		snprintf(formatBuffer, 256*3, "%s[ %s ] %s \033[0m", TextColorTable[(int)textColor], names[msg_level], message);
		char* textBuffer = new char[256*3];
		snprintf(textBuffer, 256*3, formatBuffer, args...);

		puts(textBuffer);

		delete[] formatBuffer;
		delete[] textBuffer;
	}
	inline void setLevel(Log_Level level) { log_level = level;	}
	
	Log(Log_Level level, const std::string& type) : log_level(level), type(type) {	}

private:
	Log_Level log_level;
	const std::string type;
private:
	static std::shared_ptr<Log> AppLog;
	static std::shared_ptr<Log> GraphicsLog;
	static std::unordered_map<Log_Level, const char*> names;

};