#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <map>

#include "Log.h"

#ifdef BW_DEBUGGING

#define BW_ASSERT(x, msg, ...) if (!(x)) {	\
	BW_ERROR(msg, ##__VA_ARGS__);			\
	__debugbreak();							\
	glfwTerminate();}
#else
#define BW_ASSERT(x, msg, ...) x


#endif

#ifdef GL_DEBUGGING

#define GL_ASSERT(x, msg, ...) if (!(x)) {	\
	GL_ERROR(msg, ##__VA_ARGS__);			\
	__debugbreak(); }
#define GL_TRY(x) GLClearError();			\
	x;										\
	GL_ASSERT(GLErrorLogCall(#x, __FILE__, __LINE__), NULL, NULL);
#define GLFW_DEBUG_HINT glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE);
#define GL_DEBUG_CALLBACK doGLDebugCallback();
#define GL_SHADER_COMPILE_STATUS(id, shader) shaderCompileStatus(id, shader);
#define GL_SHADER_LINK_STATUS(program) shaderLinkStatus(program);

void doGLDebugCallback();
void GLClearError();
bool GLErrorLogCall(const char* function, const char* file, int line);
void GLAPIENTRY glDebugOutput(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const char* message, const void* userParam);

void shaderCompileStatus(GLuint id, const char* shader);

void shaderLinkStatus(GLuint program);


#else
#define GL_ASSERT(x, msg, ...) x
#define GL_TRY(x) x
#define GLFW_DEBUG_HINT
#define GL_DEBUG_CALLBACK
#define GL_SHADER_COMPILE_STATUS(id, shader)
#define GL_SHADER_LINK_STATUS(program)

#endif