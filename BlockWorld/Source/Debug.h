#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include "Log.h"

#ifdef BW_DEBUGGING

#define BW_ASSERT(x) if (!(x)) {__debugbreak();\
	glfwTerminate();}
#else
#define BW_ASSERT(x)


#endif

#ifdef GL_DEBUGGING

#define GL_ASSERT(x) if (!(x)) __debugbreak();
#define GL_TRY(x) GLClearError();\
	x;\
	GL_ASSERT(GLErrorLogCall(#x, __FILE__, __LINE__));
#define GLFW_DEBUG_HINT glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
#define GL_DEBUG_CALLBACK doGLDebugCallback();

void doGLDebugCallback();
void GLClearError();
bool GLErrorLogCall(const char* function, const char* file, int line);
void GLAPIENTRY glDebugOutput(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const char* message, const void* userParam);

#else
#define GL_ASSERT(x)
#define GL_TRY(x) x
#define GLFW_DEBUG_HINT
#define GL_DEBUG_CALLBACK

#endif