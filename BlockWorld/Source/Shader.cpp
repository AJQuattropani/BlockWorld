#include "Shader.h"

#include <map>
#include <fstream>
#include <filesystem>
#include <sstream>

const std::string Shader::SHADER_PATH = "Resources/Shaders/";
const std::unordered_map<Shader_Type, Shader_Info> Shader::SHADERS = {
	{Shader_Type::VERTEX_SHADER, Shader_Info{"VERT", ".vert"} },
	{Shader_Type::FRAGMENT_SHADER, Shader_Info{"FRAG", ".frag"} },
	{Shader_Type::GEOMETRY_SHADER, Shader_Info{"GEOM", ".geom"} },
	{Shader_Type::TESS_CONTROL_SHADER, Shader_Info{"TESC", ".tesc"} },
	{Shader_Type::TESS_EVALUATION_SHADER, Shader_Info{"TESE", ".tese"} }
};

Shader::Shader(const std::string& file) : filePath(SHADER_PATH + file), shaderID(init(SHADER_PATH + file))
{
	BW_INFO("SHADER [%x] has been created.", shaderID);
	controlHead = new unsigned int(1);
}

GLuint Shader::init(const std::string& filePath) 
{
	std::unordered_map<Shader_Type, std::string>* shaders = ParseShaders(filePath);
	GLuint new_program = CreateShaders(*shaders);
	delete shaders;
	return new_program;
}

Shader::~Shader() {
	if (controlHead) {
		BW_INFO("1 instance of SHADER [%x] is destroyed of [%d] shared ptrs.", shaderID, *controlHead);
		if (--(*controlHead))
		{
			BW_INFO("SHADER [%x] is deleted.", shaderID);
			glDeleteProgram(shaderID);
		}
	}
}


Shader::Shader(const Shader& other)
	: shaderID(other.shaderID), filePath(other.filePath), uniformLocationCache(other.uniformLocationCache)
{
	controlHead = other.controlHead;
	(*controlHead)++;
	BW_INFO("New copy of SHADER [%x] totals [%d] shared ptrs.", shaderID, *controlHead);
}

Shader::Shader(Shader&& other) noexcept : shaderID(other.shaderID),
	filePath(other.filePath), uniformLocationCache(std::move(other.uniformLocationCache)), controlHead(other.controlHead)
{ 
	other.controlHead = nullptr;
	BW_INFO("SHADER [%x] instance moved. [%d] total shared ptrs.", shaderID, *controlHead);
}

Shader& Shader::operator=(const Shader& other)
{
	this->~Shader();
	new (this) Shader(other);
	return *this;
}

Shader& Shader::operator=(Shader&& other) noexcept
{
	this->~Shader();
	new (this) Shader(std::move(other));
	return *this;
}

// TODO make it move rather than copy or heap alloca
std::unordered_map<Shader_Type, std::string>* Shader::ParseShaders(const std::string& filePath) {
	std::unordered_map<Shader_Type, std::string> *shaders = new std::unordered_map<Shader_Type, std::string>();
	for (const auto& shader : SHADERS) {
		std::ifstream stream(filePath + shader.second.ext);
		if (stream.fail()) {
			switch (shader.first)
			{
			case Shader_Type::FRAGMENT_SHADER:
			case Shader_Type::VERTEX_SHADER:
				GL_ERROR("File not found: \"%s/%s%s\"",
					std::filesystem::current_path().string().data(), filePath.data(), shader.second.ext.data());
				break;
			default:
				GL_INFO("File not found: \"%s/%s%s\"", 
					std::filesystem::current_path().string().data(), filePath.data(), shader.second.ext.data());
				break;
			} 
			continue;
		}
		std::string line;
		std::stringstream ss;
		while (getline(stream, line)) {	//pushes line into the shader
			ss << line << '\n';
		}
		(*shaders)[shader.first] = ss.str();
	}
	return shaders;
}

GLuint Shader::CreateShaders(const std::unordered_map<Shader_Type, std::string>& shaderSources) {
	GLuint program = glCreateProgram();

	std::vector<GLuint> shaders(5);
	for (const auto& source : shaderSources) {
		GLuint shader = CompileShader(source.first, source.second);
		if (shader != -1) {
			glAttachShader(program, shader);
			shaders.push_back(shader);
		}
	}
	glLinkProgram(program);
	glValidateProgram(program);

	for (GLuint& shader : shaders) glDeleteShader(shader);
	
	int result;
	glGetProgramiv(program, GL_LINK_STATUS, &result);
	if (result == GL_FALSE)
	{
		shaderLinkStatus(program);
		return -1;
	}
	return program;
}

GLuint Shader::CompileShader(Shader_Type type, const std::string& source)
{
	GLuint id = glCreateShader(static_cast<GLenum>(type));
	const char* source_code = source.c_str();
	glShaderSource(id, 1, &source_code, nullptr);
	glCompileShader(id);

	// GL_INFO(source.c_str());

	// TODO move debugging info to debug.h
	int result;
	glGetShaderiv(id, GL_COMPILE_STATUS, &result);
	if (result == GL_FALSE)
	{
		shaderCompileStatus(id, SHADERS.at(type).name.data());
		return -1;
	}

	return id;
}

