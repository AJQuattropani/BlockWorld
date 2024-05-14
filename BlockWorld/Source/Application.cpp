#include "Application.h"

Application::Application(unsigned int screen_width, unsigned int screen_height, float fps, float ups) :
	screen_width_px(screen_width),
	screen_height_px(screen_height),
	context{
		.screen_handler{screen_width_px, screen_height_px},
		.scroll_handler = {0},
		.key_handler{ .keyCache = std::map<unsigned int, bool>()},
		.mouse_handler{screen_width_px/2.0,screen_height_px/2.0,0,0}
	},
	timer{ 0, 0, 0, 0 },
	frameRateSeconds(1 / fps),
	gameUpdateRateSeconds(1 / ups)
{
	window = glfwWindowInit("Block World");
	BW_ASSERT(window != nullptr);

	glfwMakeContextCurrent(window);
	glfwSetWindowUserPointer(window, &context);
	if (glewInit() != GLEW_OK) {
		GL_FATAL("Glew failed to initialize.");
		GL_ASSERT(false);
	}
	loadCallbacks(window);

	const char* version(reinterpret_cast<const char*>(glGetString(GL_VERSION)));
	GL_INFO(version);
}

int Application::run()
{
	while (!glfwWindowShouldClose(window))
	{
		if (timer.lastTimeSeconds - timer.lastFrameTimeSeconds >= frameRateSeconds)
		{
			render();
			timer.lastFrameTimeSeconds = timer.lastTimeSeconds;

		}
		
		if (timer.lastTimeSeconds - timer.lastUpdateTimeSeconds >= gameUpdateRateSeconds)
		{
			handleInput();
			update();
			timer.lastUpdateTimeSeconds = timer.lastTimeSeconds;
		}
		
		updateTime(timer.lastTimeSeconds, timer.deltaTimeSeconds);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	return 0;
}

Application::~Application()
{
	glfwTerminate();
}

GLFWwindow* Application::glfwWindowInit(const std::string& name)
{
	if (!glfwInit())
	{
		BW_FATAL("Failed to initialize GLFW.");
		return nullptr;
	}

	// TODO: look up what this does
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // macOS
	glfwWindowHint(GLFW_DOUBLEBUFFER, GL_TRUE);
	glfwWindowHint(GLFW_SAMPLES, 4);
	GLFW_DEBUG_HINT;
	return glfwCreateWindow(screen_width_px, screen_height_px, name.c_str(), NULL, NULL);
}

void Application::update()
{
	BW_WARN("Updated!");
}

void Application::render()
{
	GL_ERROR("Rendered!");
}

void Application::handleInput()
{
	// Framebuffer Input
	screen_width_px = context.screen_handler.screen_width_px;
	screen_height_px = context.screen_handler.screen_height_px;

	// Key Input
	for (const auto& key : context.key_handler.keyCache)
	{
		switch (key.first)
		{
		case GLFW_KEY_ESCAPE:
			glfwSetWindowShouldClose(window, 1);
			break;

		}
	}

	// Cursor Input
	if (context.mouse_handler.resetFlag)
	{
		context.mouse_handler.last_x = screen_width_px / 2;
		context.mouse_handler.last_y = screen_height_px / 2;
	}

	// Mouse Button Input



	// Scroll Input



}

void Application::loadCallbacks(GLFWwindow* window)
{
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetCursorEnterCallback(window, cursor_enter_callback);
	glfwSetCursorPosCallback(window, mouse_cursor_input_callback);
	glfwSetMouseButtonCallback(window, mouse_button_input_callback);
	glfwSetScrollCallback(window, scroll_callback);
	GL_DEBUG_CALLBACK;
}

void Application::updateTime(double& lastFrameTimeSeconds, double& deltaTimeSeconds)
{
	double currentFrameTimeSeconds = glfwGetTime();
	deltaTimeSeconds = currentFrameTimeSeconds - lastFrameTimeSeconds;
	lastFrameTimeSeconds = currentFrameTimeSeconds;
}

