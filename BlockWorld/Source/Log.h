#pragma once

#include <iostream>
#include <map>
#include <memory>
#include <string>

#define BW_DEFAULT_LOG_LEVEL Log_Level::INFO
#define GL_DEFAULT_LOG_LEVEL Log_Level::INFO

#ifdef BW_LOG_OUTPUT
#define BW_INFO(x) Log::getAppLog()->print(Log_Level::INFO, x)
#define BW_WARN(x) Log::getAppLog()->print(Log_Level::WARN, x)
#define BW_ERROR(x) Log::getAppLog()->print(Log_Level::ERROR, x)
#define BW_FATAL(x) Log::getAppLog()->print(Log_Level::FATAL, x)
#define BW_LEVEL(x) Log::getAppLog()->setLevel(x);
#ifdef BW_DEBUGGING
#define BW_DEBUG(x) Log::getAppLog()->print(Log_Level::DEBUG, x)
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
#define GL_INFO(x) Log::getGraphicsLog()->print(Log_Level::INFO, x)
#define GL_WARN(x) Log::getGraphicsLog()->print(Log_Level::WARN, x)
#define GL_ERROR(x) Log::getGraphicsLog()->print(Log_Level::ERROR, x)
#define GL_FATAL(x) Log::getGraphicsLog()->print(Log_Level::FATAL, x)
#define GL_DEBUG(x) Log::getGraphicsLog()->print(Log_Level::DEBUG, x)
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

class Log
{
public:
	inline static Log const* getAppLog() { return AppLog.get(); }
	inline static Log const* getGraphicsLog() { return GraphicsLog.get(); }
public:
	inline void print(Log_Level msg_level, const char* message) const {
		if (msg_level >= log_level) std::cout << type << " [ " << names[msg_level] << " ] : " << message << std::endl;
	}
	inline void setLevel(Log_Level level) { log_level = level;	}
	
	Log(Log_Level level, const std::string& type) : log_level(level), type(type) {	}

private:
	Log_Level log_level;
	const std::string type;
private:
	static std::shared_ptr<Log> AppLog;
	static std::shared_ptr<Log> GraphicsLog;
	static std::map<Log_Level, const char*> names;

};