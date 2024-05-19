#pragma once
#include <GL/glew.h>
#include <glm/glm.hpp>
#include "Debug.h"

#include <string>
#include <unordered_map>

enum class Shader_Type
{
	VERTEX_SHADER = GL_VERTEX_SHADER,
	FRAGMENT_SHADER = GL_FRAGMENT_SHADER,
	GEOMETRY_SHADER = GL_GEOMETRY_SHADER,
	TESS_CONTROL_SHADER = GL_TESS_CONTROL_SHADER,
	TESS_EVALUATION_SHADER = GL_TESS_EVALUATION_SHADER
};

struct Shader_Info
{
	std::string name;
	std::string ext;
};

// Structured based on shared_ptr
class Shader
{
public:
	explicit Shader(const std::string& path);

	~Shader();

	Shader(const Shader& other);
	Shader(Shader&& other) noexcept;
	Shader& operator=(const Shader& other);
	Shader& operator=(Shader&& other) noexcept;

	inline void bind() const
	{
		glUseProgram(shaderID);
	}

	inline void unbind() const
	{
		glUseProgram(0);
	}

	// TODO abstract shader uniform system
	void setUniform1f(const std::string& name, GLfloat value)
	{
		glUniform1f(getUniformLocation(name), value);
	}

	void setUniform2f(const std::string& name, GLfloat v0, GLfloat v1)
	{
		glUniform2f(getUniformLocation(name), v0, v1);
	}

	void setUniform3f(const std::string& name, GLfloat v0, GLfloat v1, GLfloat v2)
	{
		glUniform3f(getUniformLocation(name), v0, v1, v2);
	}

	void setUniform4f(const std::string& name, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3)
	{
		glUniform4f(getUniformLocation(name), v0, v1, v2, v3);
	}

	void setUniform1i(const std::string& name, GLint value)
	{
		glUniform1i(getUniformLocation(name), value);
	}
	
	void setUniformMat3f(const std::string& name, const glm::mat3& matrix)
	{
		glUniformMatrix3fv(getUniformLocation(name), 1, GL_FALSE, &matrix[0][0]);
	}

	void setUniformMat4f(const std::string& name, const glm::mat4& matrix)
	{
		glUniformMatrix4fv(getUniformLocation(name), 1, GL_FALSE, &matrix[0][0]);
	}

public:
	static const std::string SHADER_PATH;
	static const std::unordered_map<Shader_Type, Shader_Info> SHADERS;
private:
	const std::string filePath;
	const GLuint shaderID;
	mutable std::unordered_map<std::string, GLint> uniformLocationCache;

	unsigned int* controlHead;
private:
	GLuint getUniformLocation(const std::string& name)
	{
		if (const auto& f = uniformLocationCache.find(name); f != uniformLocationCache.end()) return f->second;
		GLint location = glGetUniformLocation(shaderID, name.c_str());
		if (location == -1) GL_WARN( " Uniform '%s' not found.", name.c_str());
		uniformLocationCache[name] = location;
		return location;
	}
private:
	static GLuint init(const std::string& filePath);

	static std::unordered_map<Shader_Type, std::string>* ParseShaders(const std::string& filePath);
	static GLuint CreateShaders(const std::unordered_map<Shader_Type, std::string>& shaderSources);
	static GLuint CompileShader(Shader_Type type, const std::string& source);


};