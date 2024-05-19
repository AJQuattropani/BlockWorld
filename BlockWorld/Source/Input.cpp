#include "Input.h"

void mouse_cursor_input_callback(GLFWwindow* window, double xpos, double ypos)
{
	static InputContext* const context = static_cast<InputContext*>(glfwGetWindowUserPointer(window));
	auto& handler = context->mouse_handler;

	//GL_INFO("Offsets: %f %f | Last: %f %f | Current Read: %f %f", handler.cached_x_offset, handler.cached_y_offset, handler.last_x, handler.last_y, xpos, ypos);

	handler.cached_x_offset += xpos - handler.last_x;
	handler.cached_y_offset += ypos - handler.last_y;
	handler.last_x = xpos;
	handler.last_y = ypos;

}

void mouse_button_input_callback(GLFWwindow* window, int button, int action, int mods)
{
	static InputContext* const context = static_cast<InputContext*>(glfwGetWindowUserPointer(window));
}

void key_input_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	static InputContext* const context = static_cast<InputContext*>(glfwGetWindowUserPointer(window));
	auto& keys = context->key_handler.keyCache;
	if (action == GLFW_PRESS)
	{
		keys[key] = true;
	}
	else if (action == GLFW_RELEASE)
	{
		keys[key] = false;
	}
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
	static InputContext* const context = static_cast<InputContext*>(glfwGetWindowUserPointer(window));
	context->scroll_handler.cache_amount += yoffset;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	static InputContext* const context = static_cast<InputContext*>(glfwGetWindowUserPointer(window));
	
	context->screen_handler.screen_width_px = width;
	context->screen_handler.screen_height_px = height;
}

void cursor_enter_callback(GLFWwindow* window, int entered)
{
	static InputContext* const context = static_cast<InputContext*>(glfwGetWindowUserPointer(window));

	if (entered)
	{
		context->mouse_handler.resetFlag = true;
	}
	else
	{

	}
}
