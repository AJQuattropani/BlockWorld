#include "Texture.h"

const std::string TextureRegister::TEXTURE_PATH = "Resources/Textures/";

TextureRegister::TextureRegister() : type("") {}

TextureRegister::TextureRegister(const std::string& type) : type(type) {
    GL_INFO("Texture register of type %s created.", type.c_str());
}

TextureRegister::~TextureRegister() {
    GL_INFO("Texture register for %s with %d texture buffers deleted.", type.c_str(), textureBuffers.size());
    glDeleteTextures(textureBuffers.size(), textureBuffers.data());
}

TextureRegister::TextureRegister(TextureRegister&& other) noexcept
    : type(other.type), cached_names(other.cached_names), textureBuffers(other.textureBuffers) {
    GL_INFO("Moved.");
}

TextureRegister& TextureRegister::operator=(TextureRegister&& other) noexcept {
    this->~TextureRegister();
    new (this) TextureRegister(std::move(other));
    return *this;
}

GLuint TextureRegister::findOrLoad(const std::string& name) {
    // 1. Get all keys in my map
    // 2. Filter keys by the size of the string
    // 2. Filter remaining keys by recursively comparing first, second... characters
    namespace rng = std::ranges::views;
    //auto compare_by_length = [&name](std::pair<const std::string, const size_t>& element) -> size_t {
    //    return element.first.length() == name.length();
    //    };
    //auto filter_by_length = [&name, &compare_by_length](std::unordered_map<const std::string&, size_t>& key_map) {
    //    return rng::filter(key_map, compare_by_length);
    //    };
    //auto filter_by_length = [&name](std::unordered_map<const std::string, const size_t>& key_map) {
    //    return rng::filter(key_map, [&name](auto& element) { return element.first.length() == name.length(); });
    //    };
    //auto filter_by_value = [&name, &compare_by_char](auto& key_map) {
    //    return rng::filter(key_map, compare_by_char);
    //    };
    ////auto result = filter_by_value(cached_names);
    ////auto result = filter_by_length(cached_names);
    ////auto result2 = filter_by_value(result);
    auto compare_by_char_impl = [&name](auto& self, auto& element, size_t index = 0) -> bool {
        if (index == name.length()) return true;
        if (element.first[index] != name[index]) return false;
        self(self, element, index + 1);
        };

    auto compare_by_char = [&name, &compare_by_char_impl](auto& element) -> bool {
        return compare_by_char_impl(compare_by_char_impl, element, 0);
        };

    auto f = rng::filter(cached_names, [&name](auto& element) { return element.first.length() == name.length(); });
    auto f1 = rng::filter(cached_names, compare_by_char);

    auto search_filter = [&name, &compare_by_char](std::unordered_map<std::string, size_t>& key_map) {
        return rng::filter(rng::filter(key_map, [&name](auto& element) { return element.first.length() == name.length(); }), compare_by_char);
        };
    if (auto result = search_filter(cached_names); !result.empty()) return textureBuffers[result.front().second];

    GLuint textureID = loadTexture(name);
    textureBuffers.push_back(textureID);
    cached_names.insert({ name,  textureBuffers.size() - 1 });

    GL_INFO("New member: %x in %s", textureID, type.c_str());

    return textureID;
}

GLuint TextureRegister::findOrLoadLazy(const std::string& name) {
    if (const auto& f = cached_names.find(name); f != cached_names.end()) return textureBuffers[f->second];

    GLuint textureID = loadTexture(name);
    textureBuffers.push_back(textureID);
    cached_names.insert({ name, textureBuffers.size() - 1 });

    return textureID;
}

GLuint TextureRegister::loadTexture(const std::string& name, GLint wrap_s, GLint wrap_t, GLint min_filter, GLint mag_filter) {
    stbi_set_flip_vertically_on_load(true);

    int width;
    int height;
    int nrChannels;

    unsigned char* data = stbi_load((TEXTURE_PATH + type + "/" + name).c_str(), &width, &height, &nrChannels, 0);

    GLenum format;
    switch (nrChannels)
    {
    case 1:
        format = GL_RED;
        break;
    case 3:
        format = GL_RGB;
        break;
    case 4:
        format = GL_RGBA;
        break;
    default:
        BW_ASSERT(false);
    }

    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap_s);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap_t);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, min_filter);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, mag_filter);

    GL_INFO((TEXTURE_PATH + type + "/" + name).c_str());

    BW_ASSERT(data != nullptr);

    glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
    glGenerateMipmap(GL_TEXTURE_2D);

    stbi_image_free(data);

    glBindTexture(GL_TEXTURE_2D, 0);

    return textureID;
}


