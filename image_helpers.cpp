#include "image_helpers.hpp"

#include <iostream>
#include <fstream>
#include <assert.h>


//size with be the side length of a cube in the cubemap(how)
bool loadTextureFile(const std::string& filename, uint32_t &_width, uint32_t &_height, std::vector<char> &data) {
    int width,height,channels;
    uint32_t desired_channels = 4;//rgba
    stbi_set_flip_vertically_on_load(0);
    unsigned char* loaded_data = stbi_load(filename.c_str(), &width, &height, &channels, desired_channels);
    if (!loaded_data) {
        std::cerr << "Error: Failed to load image." << std::endl;
        std::cerr << "Reason: " << stbi_failure_reason() << std::endl; // Optional: print failure reason
        return false;
    }
    
    //Calculate the total size of the image data
    size_t data_size = width * height * desired_channels;
    _width = (uint32_t)width;
    _height = (uint32_t)height;
    //Copy the raw data into a the struct's data field
    data.resize(data_size);
    std::memcpy(data.data(), loaded_data, data_size);

    //Free the memory allocated by stb_image immediately after copying
    stbi_image_free(loaded_data);

    std::cout << "Width: " << width << ", Height: " << height << ", Channels: " << desired_channels << std::endl;
    std::cout << "Data size in vector: " << data.size() << " bytes\n" << std::endl;
    return true;
}



bool loadBinaryFile(const std::string& filename, uint32_t &size, std::vector<char> &data) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "Error opening file: " << filename << '\n';
        return false;
    }

    const std::streamsize fileSize = file.tellg();
    if (fileSize < 0) {
        std::cerr << "Invalid file size: " << filename << '\n';
        return false;
    }

    file.seekg(0, std::ios::beg);

    data.resize(static_cast<size_t>(fileSize));
    if (!file.read(data.data(), fileSize)) {
        std::cerr << "Error reading file, only read "
                << file.gcount() << " bytes\n";
        data.clear();
        size = 0;
        return false;
    }

    size = static_cast<uint32_t>(fileSize);
    return true;
}


//expects a cubemap with pre-resized rgba_float_vector
void rgbe_to_rgba_float(std::vector<char> const &rgbe_vector, std::vector<float> &rgba_float_vector, uint32_t width){
    assert(rgbe_vector.size() == 4 * width * width * 6);
    assert(rgba_float_vector.size() == 4 * width * width * 6);
   
	for(uint32_t i = 0; i < width * width * 6; i++){
		uint8_t exponent = rgbe_vector[4*i+3];
		if (exponent == 0) {
			rgba_float_vector[i*4 + 0] = 0.0f;
			rgba_float_vector[i*4 + 1] = 0.0f;
			rgba_float_vector[i*4 + 2] = 0.0f;
			rgba_float_vector[i*4 + 3] = 1.0f;
			continue;
		}
		uint8_t r = rgbe_vector[4*i+0];
		uint8_t g = rgbe_vector[4*i+1];
		uint8_t b = rgbe_vector[4*i+2];

		float r_rad = std::ldexp(1.0f, exponent - 128) * (r + 0.5f)/256;
		float g_rad = std::ldexp(1.0f, exponent - 128) * (g + 0.5f)/256;
		float b_rad = std::ldexp(1.0f, exponent - 128) * (b + 0.5f)/256;
		rgba_float_vector[i*4 + 0] = r_rad;
		rgba_float_vector[i*4 + 1] = g_rad;
		rgba_float_vector[i*4 + 2] = b_rad;
	}
}

//rgba_floats is four floats per pixel, the radiance_values parameter should be pre-resized
//size is the number of pixels
void rgba_float_to_radiance_values(std::vector<float> const &rgba_floats, std::vector<vec3> &radiance_values, uint32_t size){
    assert(rgba_floats.size() == size * 4);
    assert(radiance_values.size() == size);
    for(uint32_t i = 0; i < size; i++){
        radiance_values[i] = vec3(rgba_floats[i * 4 + 0], rgba_floats[i * 4 + 1], rgba_floats[i * 4 + 2]);
    }
}

void rgbe_to_radiance_values(std::vector<char> const &rgbe_values, std::vector<vec3> &radiance_values, uint32_t width){
    assert(rgbe_values.size() == 4 * width * width * 6);
    assert(radiance_values.size() == width * width * 6);
   
	for(uint32_t i = 0; i < width * width * 6; i++){
		uint8_t exponent = rgbe_values[4*i+3];
		if (exponent == 0) {
			radiance_values[i].x = 0.0f;
			radiance_values[i].y = 0.0f;
			radiance_values[i].z = 0.0f;
			continue;
		}
		uint8_t r = rgbe_values[4*i+0];
		uint8_t g = rgbe_values[4*i+1];
		uint8_t b = rgbe_values[4*i+2];

		float r_rad = std::ldexp(1.0f, exponent - 128) * (r + 0.5f)/256;
		float g_rad = std::ldexp(1.0f, exponent - 128) * (g + 0.5f)/256;
		float b_rad = std::ldexp(1.0f, exponent - 128) * (b + 0.5f)/256;
		radiance_values[i].x = r_rad;
		radiance_values[i].y = g_rad;
		radiance_values[i].z = b_rad;
	}

}


void radiance_values_to_rgbe(std::vector<vec3> const &radiance_values, std::vector<char> &rgbe_vector, uint32_t size){
    assert(radiance_values.size() == size);
    assert(rgbe_vector.size() == size * 4);

    for(uint32_t i = 0; i < size; i++){

        float r = radiance_values[i].x;
        float g = radiance_values[i].y;
        float b = radiance_values[i].z;

        float max_channel = std::max({r, g, b});

        if (max_channel < 1e-32f) {
            // Black pixel
            rgbe_vector[4*i+0] = 0;
            rgbe_vector[4*i+1] = 0;
            rgbe_vector[4*i+2] = 0;
            rgbe_vector[4*i+3] = 0;
            continue;
        }

        int exponent;
        float mantissa = std::frexp(max_channel, &exponent);
        // max_channel = mantissa * 2^exponent
        // mantissa in [0.5, 1)

        float scale = mantissa * 256.0f / max_channel;

        rgbe_vector[4*i+0] = (uint8_t)(r * scale);
        rgbe_vector[4*i+1] = (uint8_t)(g * scale);
        rgbe_vector[4*i+2] = (uint8_t)(b * scale);
        rgbe_vector[4*i+3] = (uint8_t)(exponent + 128);
    }

}
