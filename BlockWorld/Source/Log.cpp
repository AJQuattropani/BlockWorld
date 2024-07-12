#include "Log.h"

#ifndef DIST
std::shared_ptr<Log> Log::AppLog = std::make_shared<Log>(BW_DEFAULT_LOG_LEVEL, "Application");;
std::shared_ptr<Log> Log::GraphicsLog = std::make_shared<Log>(GL_DEFAULT_LOG_LEVEL, "OpenGL");
std::mutex Log::log_lock;
#endif

std::unordered_map<Log_Level, const char*> Log::names =
{
	{Log_Level::INFO, "INFO"},
	{Log_Level::WARN, "WARNING"},
	{Log_Level::ERROR, "ERROR"},
	{Log_Level::FATAL, "FATAL"},
	{Log_Level::DEBUG, "DEBUG"}
};;
