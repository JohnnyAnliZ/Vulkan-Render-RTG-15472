#pragma once

#include "stb_image.h"
#include "mat4.hpp"
#include <string>
#include <vector>


bool loadTextureFile(const std::string &filename, uint32_t &_width, uint32_t &_height, std::vector<char> &data);

bool loadBinaryFile(const std::string &filename, uint32_t &size, std::vector<char> &data);

//expect cubemap
void rgbe_to_rgba_float(std::vector<char> const &rgbe_vector, std::vector<float> &rgba_float_vector, uint32_t width);

//could be any image
void rgba_float_to_radiance_values(std::vector<float> const &rgba_floats, std::vector<vec3> &radiance_values, uint32_t size);

//expect cubemap
void rgbe_to_radiance_values(std::vector<char> const &rgbe_values, std::vector<vec3> &radiance_values, uint32_t width);

void radiance_values_to_rgbe(std::vector<vec3> const &radiance_values, std::vector<char> &rgbe_vector, uint32_t size);