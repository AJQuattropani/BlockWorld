#pragma once
#pragma once

#define GLEW_STATIC
#include <GL/glew.h>

#include "Debug.h"
#include "Input.h"
#include "Shader.h"
#include "Renderer.h"

#include <string>

class Application
{
public:
	Application(unsigned int screen_width = 1080, unsigned int screen_height = 720, float fps = 60.0f, float ups = 60.0f);
	int run();
	~Application();
private:
	GLFWwindow* window;
	double frameRateSeconds, gameUpdateRateSeconds;
	InputContext inputContext;
	RenderContext* renderContext;
	Camera camera;
	struct {
		double lastTimeSeconds, deltaTimeSeconds;
		double lastFrameTimeSeconds, lastUpdateTimeSeconds;
	} timer;



private:
	GLFWwindow* glfwWindowInit(const std::string& name);
private:
	void update();

	void render();

	void handleInput();

	void handleRenderContext();
private:
	static void loadCallbacks(GLFWwindow* window);
	static void updateTime(double& lastTimeSeconds, double& deltaTimeSeconds);


};