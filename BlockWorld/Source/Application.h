#pragma once

#define GLEW_STATIC
#include <GL/glew.h>

#include "Debug.h"
#include "Input.h"
#include "Shader.h"
#include "RenderContext.h"
#include "Camera.h"
#include "Chunk.h"
#include "World.h"

#include <memory>
#include <string>

class Application
{
public:
	Application(unsigned int screen_width = 1080, unsigned int screen_height = 720, float fps = 60.0f, float ups = 60.0f);
	int run();
	~Application();
private:
	GLFWwindow* window;
	InputContext inputContext;
	std::shared_ptr<bwrenderer::RenderContext> renderContext;
	bwrenderer::Camera camera;
	double frameRateSeconds, gameUpdateRateSeconds;
	struct {
		double lastTimeSeconds, deltaTimeSeconds;
		double lastFrameTimeSeconds, lastUpdateTimeSeconds;
	} timer;
	std::shared_ptr<bwgame::BlockRegister> blocks;
	bwgame::World world;

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