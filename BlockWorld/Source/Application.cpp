#include "Application.hpp"
#include "Vendor/stb_image.h"

Application::Application(unsigned int screen_width, unsigned int screen_height, float fps, float ups) :
	input_context{
		.screen_handler{screen_width, screen_height},
		.key_handler{.key_cache = std::map<unsigned int, bool>()},
		.mouse_handler{screen_width / 2.0, screen_height / 2.0,0,0}
	},
	blocks(std::make_shared<bwgame::BlockRegister>())
{
	window = glfwWindowInit("Block World");
	BW_ASSERT(window != nullptr, "Window failed to initialize.");

	glfwMakeContextCurrent(window);
	glfwSetWindowUserPointer(window, &input_context);

	GL_ASSERT(glewInit() == GLEW_OK, "Glew failed to initialize.");

	loadCallbacks(window);
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	render_context = std::make_shared<bwrenderer::RenderContext>
		(bwrenderer::RenderContext{ .screen_width_px = screen_width, .screen_height_px = screen_height,
			.ch_shadow_window_distance = 8});

	user_context = std::make_shared<bwgame::UserContext>
		(bwgame::UserContext{ .player_position_x = 0.0, .player_position_y = 0.0, .player_position_z = 0.0, .ch_render_load_distance = 12,
			.Timer{.frame_rate_seconds = 1.0/fps, .game_update_rate_seconds = 1.0/ups} });

	camera = std::make_unique<bwrenderer::Camera>(user_context, render_context);

	world = std::make_unique<bwgame::World>(blocks, user_context, ups, 10.0, 1);

	world_renderer = std::make_unique<bwrenderer::WorldRenderer>(world, user_context, render_context);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glDisable(GL_MULTISAMPLE);

	const char* version(reinterpret_cast<const char*>(glGetString(GL_VERSION)));
	GL_INFO(version);
}

int Application::run() {

	render();
	glfwSwapBuffers(window);

	auto& Timer = user_context->Timer;

	while (!glfwWindowShouldClose(window))
	{

		if (Timer.last_time_seconds - Timer.last_update_time_seconds >= Timer.game_update_rate_seconds)
		{
			Timer.last_update_time_seconds = Timer.last_time_seconds;
			handleInput();
			update();
		}

		if (Timer.last_time_seconds - Timer.last_frame_time_seconds >= Timer.frame_rate_seconds)
		{
			Timer.last_frame_time_seconds = Timer.last_time_seconds;
			handleContext();
			render();
			glfwSwapBuffers(window);

		}
		
		updateTime(Timer.last_time_seconds, Timer.delta_time_seconds);

		glfwPollEvents();
	}

	return 0;
}

Application::~Application() {
	GL_INFO("Application terminated.");
	glfwTerminate();
}

GLFWwindow* Application::glfwWindowInit(const std::string& name) {
	if (!glfwInit())
	{
		BW_FATAL("Failed to initialize GLFW.");
		return nullptr;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
	glfwWindowHint(GLFW_DOUBLEBUFFER, GL_TRUE);
	glfwWindowHint(GLFW_SAMPLES, 4);
	GLFW_DEBUG_HINT;
	return glfwCreateWindow(input_context.screen_handler.screen_width_px, 
		input_context.screen_handler.screen_height_px, name.c_str(), NULL, NULL);
}

void Application::update() {
	world->update();
	
	BW_DEBUG("Player coords: { %f, %f, %f }", user_context->player_position_x,
		user_context->player_position_y, user_context->player_position_z);
}

void Application::render() {
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, render_context->screen_width_px, render_context->screen_height_px);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	world_renderer->render();

}

void Application::handleInput() {

	// Framebuffer Input
	render_context->screen_width_px = input_context.screen_handler.screen_width_px;
	render_context->screen_height_px = input_context.screen_handler.screen_height_px;


	// Key Input
	for (const auto& key : input_context.key_handler.key_cache)
	{
		if (key.second)
		{
			switch (key.first)
			{
			case GLFW_KEY_ESCAPE:
				glfwSetWindowShouldClose(window, 1);
				break;
			case GLFW_KEY_W:
				camera->move(bwgame::Controls::FORWARD, 1.0f);
				break;
			case GLFW_KEY_S:
				camera->move(bwgame::Controls::BACKWARD, 1.0f);
				break;
			case GLFW_KEY_D:
				camera->move(bwgame::Controls::RIGHT, 1.0f);
				break;
			case GLFW_KEY_A:
				camera->move(bwgame::Controls::LEFT, 1.0f);
				break;
			case GLFW_KEY_SPACE:
				camera->move(bwgame::Controls::UP, 1.0f);
				break;
			case GLFW_KEY_LEFT_SHIFT:
				camera->move(bwgame::Controls::DOWN, 1.0f);
				break;
			}
		}
	}

	camera->turn(static_cast<float>(input_context.mouse_handler.cached_x_offset), static_cast<float>(-input_context.mouse_handler.cached_y_offset));
	input_context.mouse_handler.cached_x_offset = 0;
	input_context.mouse_handler.cached_y_offset = 0;

	// Cursor Input
	if (input_context.mouse_handler.reset_flag) {
		GL_INFO("Reset mouse.");
		glfwGetCursorPos(window, &input_context.mouse_handler.last_x, &input_context.mouse_handler.last_y);
		
		input_context.mouse_handler.reset_flag = false;
	}

	if (user_context->player_position_y > 0.0f && user_context->player_position_y < 256.0f)
	{	// Mouse Button Input
		if (input_context.click_handler.right_click)
		{
			world->setBlock(blocks->logs, bwgame::WorldBlockCoords{
				.x = static_cast<int64_t>(floor(user_context->player_position_x + 0.5f)),
				.z = static_cast<int64_t>(floor(user_context->player_position_z + 0.5f)),
				.y = static_cast<uint8_t>(floor(user_context->player_position_y + 0.5f)) });
		}
		if (input_context.click_handler.left_click)
		{
			world->destroyBlock(bwgame::WorldBlockCoords{
				.x = static_cast<int64_t>(floor(user_context->player_position_x + 0.5f)),
				.z = static_cast<int64_t>(floor(user_context->player_position_z + 0.5f)),
				.y = static_cast<uint8_t>(floor(user_context->player_position_y + 0.5f)) });
		}
	}


	// Scroll Input
	camera->zoom(static_cast<float>(input_context.scroll_handler.cache_amount));
	input_context.scroll_handler.cache_amount = 0;
}

void Application::handleContext()
{
	camera->updateContext();
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

