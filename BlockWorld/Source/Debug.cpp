#include "Debug.hpp"

#ifdef BW_DEBUGGING



#endif

#ifdef GL_DEBUGGING

void doGLDebugCallback()
{
    glEnable(GL_DEBUG_OUTPUT);
    int flags; glGetIntegerv(GL_CONTEXT_FLAGS, &flags);
    if (flags & GL_CONTEXT_FLAG_DEBUG_BIT)
    {
        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        glDebugMessageCallback(glDebugOutput, nullptr);
        glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);
    }
}

void GLAPIENTRY glDebugOutput(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const char* message, const void* userParam)
{

    // ignore non-significant error/warning codes
    //if (id == 131169 || id == 131185 || id == 131218 || id == 131204) return;

    const static std::map<GLenum, const char*> sources =
    {
       { GL_DEBUG_SOURCE_API,                   "API" },
       { GL_DEBUG_SOURCE_WINDOW_SYSTEM,         "Window System" },
       { GL_DEBUG_SOURCE_SHADER_COMPILER,       "Shader Compiler" },
       { GL_DEBUG_SOURCE_THIRD_PARTY,           "Third Party" },
       { GL_DEBUG_SOURCE_APPLICATION,           "Application" },
       { GL_DEBUG_SOURCE_OTHER,                 "Other" }
    };

    const static std::map<GLenum, const char*> types =
    {
        { GL_DEBUG_TYPE_ERROR,                  "Error" },
        { GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR,    "Deprecated Behavior"},
        { GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR,     "Undefined Behavior"},
        { GL_DEBUG_TYPE_PORTABILITY,            "Portability"},
        { GL_DEBUG_TYPE_PERFORMANCE,            "Performance"},
        { GL_DEBUG_TYPE_MARKER,                 "Marker"},
        { GL_DEBUG_TYPE_PUSH_GROUP,             "Push Group"},
        { GL_DEBUG_TYPE_POP_GROUP,              "Pop Group"},
        { GL_DEBUG_TYPE_OTHER,                  "Other"}
    };

    //static std::string log_message = " [ " + std::to_string(id) + " ]: " + message + "\n"
    //    + "Source: " + sources.find(source)->second + " | Type: " + types.find(type)->second;

    switch (severity)
    {
    case GL_DEBUG_SEVERITY_HIGH:         GL_FATAL("[%d]: %s\n Source: %s | Type: %s", id, message, 
        sources.find(source)->second, types.find(type)->second); break;

    case GL_DEBUG_SEVERITY_MEDIUM:       GL_ERROR("[%d]: %s\n Source: %s | Type: %s", id, message,
        sources.find(source)->second, types.find(type)->second); break;

    case GL_DEBUG_SEVERITY_LOW:          GL_WARN("[%d]: %s\n Source: %s | Type: %s", id, message,
        sources.find(source)->second, types.find(type)->second); break;

    case GL_DEBUG_SEVERITY_NOTIFICATION: GL_INFO("[%d]: %s\n Source: %s | Type: %s", id, message,
        sources.find(source)->second, types.find(type)->second); break;
    }
}

void shaderCompileStatus(GLuint id, const char* shader)
{
    int length;
    glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
    char* message = (char*)_malloca(length * sizeof(char));
    glGetShaderInfoLog(id, length, &length, message);
    GL_ERROR("COMPILER: %s | %s", shader, message);
}

void shaderLinkStatus(GLuint program)
{
    int length;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
    char* message = (char*)_malloca(length * sizeof(char));
    glGetShaderInfoLog(program, length, &length, message);
    GL_ERROR("LINKER: %s ", message);
}

void GLClearError()
{
    while (glGetError() != GL_NO_ERROR);
}

bool GLErrorLogCall(const char* function, const char* file, int line)
{
    while (GLenum errorCode = glGetError())
    {
        static std::string error;
        switch (errorCode)
        {
        case GL_INVALID_ENUM:	error = "INVALID_ENUM"; break;
        case GL_INVALID_VALUE:	error = "INVALID_VALUE"; break;
        case GL_INVALID_OPERATION:	error = "INVALID_OPERATION"; break;
        case GL_STACK_OVERFLOW:	error = "STACK_OVERFLOW"; break;
        case GL_STACK_UNDERFLOW:	error = "STACK_UNDERFLOW"; break;
        case GL_OUT_OF_MEMORY:	error = "OUT_OF_MEMORY"; break;
        }
        error += std::string(" Error located in ") + function + " " + file + ": " + std::to_string(line);
        GL_ERROR( error.c_str());
        return false;
    }
    return true;
}

#endif