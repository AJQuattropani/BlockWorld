#pragma once
#pragma once

#define GLEW_STATIC
#include <GL/glew.h>

#include "Debug.h"
#include "Input.h"

#include <string>

class Application
{
public:
	Application(unsigned int screen_width = 1080, unsigned int screen_height = 720, float fps = 1.0f, float ups = 1.0f);
	int run();
	~Application();
private:
	GLFWwindow* window;
	InputContext context;
	struct {
		double lastTimeSeconds, deltaTimeSeconds;
		double lastFrameTimeSeconds, lastUpdateTimeSeconds;
	} timer;
	double frameRateSeconds, gameUpdateRateSeconds;

	unsigned int screen_width_px, screen_height_px;

private:
	GLFWwindow* glfwWindowInit(const std::string& name);
private:
	void update();

	void render();

	void handleInput();

private:
	static void loadCallbacks(GLFWwindow* window);
	static void updateTime(double& lastTimeSeconds, double& deltaTimeSeconds);


};