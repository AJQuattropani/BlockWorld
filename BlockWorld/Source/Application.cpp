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

	context = std::make_shared<bwgame::Context>
		(bwgame::Context{ .screen_width_px = screen_width, .screen_height_px = screen_height,
			.ch_shadow_window_distance = 8, .ch_render_load_distance = 12,
			.Timer{.frame_rate_seconds = 1.0 / fps, .game_update_rate_seconds = 1.0 / ups}, .player_position = glm::vec3(0.0f) });

	
	world = std::make_shared<bwgame::World>(blocks, context, ups, 10.0, 1);
	world_renderer = std::make_unique<bwrenderer::WorldRenderer>(world, context);

	player = std::make_shared<bwgame::Player>(world, context);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glDisable(GL_MULTISAMPLE);

	const char* version(reinterpret_cast<const char*>(glGetString(GL_VERSION)));
	GL_INFO(version);
}

int Application::run() {

	render();
	glfwSwapBuffers(window);

	auto& Timer = context->Timer;

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
	
	BW_DEBUG("Player coords: { %f, %f, %f }", context->player_position.x,
		context->player_position.y, context->player_position.z);
}

void Application::render() {
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, context->screen_width_px, context->screen_height_px);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	world_renderer->render();

}

void Application::handleInput() {

	// Framebuffer Input
	context->screen_width_px = input_context.screen_handler.screen_width_px;
	context->screen_height_px = input_context.screen_handler.screen_height_px;


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
				player->move(bwgame::Controls::FORWARD);
				break;
			case GLFW_KEY_S:
				player->move(bwgame::Controls::BACKWARD);
				break;
			case GLFW_KEY_D:
				player->move(bwgame::Controls::RIGHT);
				break;
			case GLFW_KEY_A:
				player->move(bwgame::Controls::LEFT);
				break;
			case GLFW_KEY_SPACE:
				player->move(bwgame::Controls::UP);
				break;
			case GLFW_KEY_LEFT_SHIFT:
				player->move(bwgame::Controls::DOWN);
				break;
			}
		}
	}

	if (input_context.mouse_handler.cached_x_offset != 0 || input_context.mouse_handler.cached_y_offset != 0)
	{
		player->turn(static_cast<float>(input_context.mouse_handler.cached_x_offset), static_cast<float>(-input_context.mouse_handler.cached_y_offset));
		input_context.mouse_handler.cached_x_offset = 0;
		input_context.mouse_handler.cached_y_offset = 0;
	}

	// Cursor Input
	if (input_context.mouse_handler.reset_flag) {
		GL_INFO("Reset mouse.");
		glfwGetCursorPos(window, &input_context.mouse_handler.last_x, &input_context.mouse_handler.last_y);
		
		input_context.mouse_handler.reset_flag = false;
	}

	if (input_context.click_handler.right_click_press)
	{
		player->placeBlock();
	}
	if (input_context.click_handler.left_click_press)
	{
		player->breakBlock();
	}
	

	if (input_context.click_handler.left_click_release)
	{
		input_context.click_handler.left_click_press = false;
		input_context.click_handler.left_click_release = false;
	}

	if (input_context.click_handler.right_click_release)
	{
		input_context.click_handler.right_click_press = false;
		input_context.click_handler.right_click_release = false;
	}

	if (input_context.click_handler.middle_click_release)
	{
		input_context.click_handler.middle_click_press = false;
		input_context.click_handler.middle_click_release = false;
	}

	// Scroll Input
	player->zoom(static_cast<float>(input_context.scroll_handler.cache_amount));
	input_context.scroll_handler.cache_amount = 0;
}

void Application::handleContext()
{
	player->update();
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

