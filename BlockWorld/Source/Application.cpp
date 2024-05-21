#include "Application.h"
#include "Vendor/stb_image.h"

Application::Application(unsigned int screen_width, unsigned int screen_height, float fps, float ups) :
	inputContext{
		.screen_handler{screen_width, screen_height},
		.scroll_handler = {0},
		.key_handler{.keyCache = std::map<unsigned int, bool>()},
		.mouse_handler{screen_width / 2.0, screen_height / 2.0,0,0}
},
timer{ 0, 0, 0, 0 },
frameRateSeconds(1 / fps),
gameUpdateRateSeconds(1 / ups),
renderContext(nullptr)
{
	window = glfwWindowInit("Block World");
	BW_ASSERT(window != nullptr, "Window failed to initialize.");

	glfwMakeContextCurrent(window);
	glfwSetWindowUserPointer(window, &inputContext);

	GL_ASSERT(glewInit() == GLEW_OK, "Glew failed to initialize.");

	loadCallbacks(window);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	renderContext = new bwrenderer::RenderContext{screen_width, screen_height, Shader("Blocks", "block_shader"), glm::mat4(1.0), glm::mat4(1.0f)};

	camera.attachContext(renderContext);

	glClearColor(0.3f, 0.4f, 1.0f, 1.0f);
	glEnable(GL_DEPTH_TEST);

	const char* version(reinterpret_cast<const char*>(glGetString(GL_VERSION)));
	GL_INFO(version);
}

int Application::run() {


	render();
	glfwSwapBuffers(window);

	while (!glfwWindowShouldClose(window))
	{

		if (timer.lastTimeSeconds - timer.lastUpdateTimeSeconds >= gameUpdateRateSeconds)
		{
			handleInput();
			update();
			timer.lastUpdateTimeSeconds = timer.lastTimeSeconds;
		}

		if (timer.lastTimeSeconds - timer.lastFrameTimeSeconds >= frameRateSeconds)
		{
			handleRenderContext();
			render();
			timer.lastFrameTimeSeconds = timer.lastTimeSeconds;
			glfwSwapBuffers(window);

		}
		
		updateTime(timer.lastTimeSeconds, timer.deltaTimeSeconds);

		glfwPollEvents();
	}

	return 0;
}

Application::~Application() {
	delete renderContext;
	glfwTerminate();
}

GLFWwindow* Application::glfwWindowInit(const std::string& name) {
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
	return glfwCreateWindow(inputContext.screen_handler.screen_width_px, 
		inputContext.screen_handler.screen_height_px, name.c_str(), NULL, NULL);
}

void Application::update() {
}

void Application::render() {
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	static bwrenderer::TexturedCubeMesh texturedCube(&texture_cache.findOrLoad("Blocks", "blockmap.jpeg"), glm::vec3(0.0f, 0.0f, 0.0f));

	renderContext->shader.bind();
	renderContext->shader.setUniform3f("inColor", 0.5, 1.0f, 0.7f);
	renderContext->shader.setUniformMat4f("view", renderContext->viewMatrix);
	renderContext->shader.setUniformMat4f("projection", renderContext->projectionMatrix);
	renderContext->shader.setUniform1i("block_texture", 0);

	texturedCube.render(renderContext->shader);
}

void Application::handleInput() {
	// Framebuffer Input
	renderContext->screen_width_px = inputContext.screen_handler.screen_width_px;
	renderContext->screen_height_px = inputContext.screen_handler.screen_height_px;

	glViewport(0, 0, renderContext->screen_width_px, renderContext->screen_height_px);

	// Key Input
	for (const auto& key : inputContext.key_handler.keyCache)
	{
		if (key.second)
		{
			switch (key.first)
			{
			case GLFW_KEY_ESCAPE:
				glfwSetWindowShouldClose(window, 1);
				break;
			case GLFW_KEY_W:
				camera.move(Camera_Controls::FORWARD, 1.0f);
				break;
			case GLFW_KEY_S:
				camera.move(Camera_Controls::BACKWARD, 1.0f);
				break;
			case GLFW_KEY_D:
				camera.move(Camera_Controls::RIGHT, 1.0f);
				break;
			case GLFW_KEY_A:
				camera.move(Camera_Controls::LEFT, 1.0f);
				break;
			case GLFW_KEY_SPACE:
				camera.move(Camera_Controls::UP, 1.0f);
				break;
			case GLFW_KEY_LEFT_SHIFT:
				camera.move(Camera_Controls::DOWN, 1.0f);
				break;
			}
		}
	}

	camera.turn(static_cast<float>(inputContext.mouse_handler.cached_x_offset), static_cast<float>(-inputContext.mouse_handler.cached_y_offset));
	inputContext.mouse_handler.cached_x_offset = 0;
	inputContext.mouse_handler.cached_y_offset = 0;

	// Cursor Input
	if (inputContext.mouse_handler.resetFlag) {
		GL_INFO("Reset mouse.");
		glfwGetCursorPos(window, &inputContext.mouse_handler.last_x, &inputContext.mouse_handler.last_y);
		
		inputContext.mouse_handler.resetFlag = false;
	}

	// Mouse Button Input



	// Scroll Input
	camera.zoom(static_cast<float>(inputContext.scroll_handler.cache_amount));
	inputContext.scroll_handler.cache_amount = 0;
}

void Application::handleRenderContext()
{
	camera.updateContext();
}

void Application::loadCallbacks(GLFWwindow* window)
{
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetCursorEnterCallback(window, cursor_enter_callback);
	glfwSetCursorPosCallback(window, mouse_cursor_input_callback);
	glfwSetMouseButtonCallback(window, mouse_button_input_callback);
	glfwSetKeyCallback(window, key_input_callback);
	glfwSetScrollCallback(window, scroll_callback);
	GL_DEBUG_CALLBACK;
}

void Application::updateTime(double& lastFrameTimeSeconds, double& deltaTimeSeconds)
{
	double currentFrameTimeSeconds = glfwGetTime();
	deltaTimeSeconds = currentFrameTimeSeconds - lastFrameTimeSeconds;
	lastFrameTimeSeconds = currentFrameTimeSeconds;
}

