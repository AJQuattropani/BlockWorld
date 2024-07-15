#pragma once

#include "Debug.hpp"

#include "GLFW/glfw3.h"

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include <map>

struct InputContext
{
	struct {
		unsigned int screen_width_px, screen_height_px;
	} screen_handler;

	struct {
		double cache_amount;
	} scroll_handler;
	
	struct {
		std::map<unsigned int, bool> key_cache;
	} key_handler;

	struct {
		double last_x, last_y;
		double cached_x_offset, cached_y_offset;
		bool reset_flag;
	} mouse_handler;
};

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void cursor_enter_callback(GLFWwindow* window, int entered);
void mouse_cursor_input_callback(GLFWwindow* window, double xoffset, double yoffset);
void mouse_button_input_callback(GLFWwindow* window, int button, int action, int mods);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void key_input_callback(GLFWwindow* window, int key, int scancode, int action, int mods);