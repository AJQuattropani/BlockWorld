#pragma once

#define GLEW_STATIC
#include <GL/glew.h>

#include "Debug.hpp"
#include "Input.hpp"
#include "Shader.hpp"
#include "Context.hpp"
#include "Camera.hpp"
#include "Chunk.hpp"
#include "World.hpp"
#include "WorldRenderer.hpp"

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
	// REMOVE
	InputContext input_context;
	std::shared_ptr<bwrenderer::RenderContext> render_context;
	std::unique_ptr<bwrenderer::WorldRenderer> world_renderer;
	std::unique_ptr<bwrenderer::Camera> camera;

	std::shared_ptr<bwgame::UserContext> user_context;
	std::shared_ptr<bwgame::BlockRegister> blocks;
	std::shared_ptr<bwgame::World> world;

private:
	GLFWwindow* glfwWindowInit(const std::string& name);
private:
	void update();

	void render();

	void handleInput();

	void handleContext();
private:
	static void loadCallbacks(GLFWwindow* window);
	static void updateTime(double& lastTimeSeconds, double& deltaTimeSeconds);


};