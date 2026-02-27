#pragma once

#include "image_helpers.hpp"
#include "stb_image_write.h"

#include "mat4.hpp"

#include <iostream>
#include <cmath>
#include <algorithm>
#include <string>
#include <assert.h>


struct Sample {
    vec3 direction;
    float solid_angle;
    vec3 radiance;
};

//helpers
float cube_texel_solid_angle(uint32_t x, uint32_t y, uint32_t width);
vec3 cube_texel_direction(uint32_t face, uint32_t x, uint32_t y, uint32_t width);
void direction_to_cube_texel(vec3 d, uint32_t width, uint32_t &face, uint32_t &x, uint32_t &y);

//for diffuse shading
void convolve_cubemap_diffuse(uint32_t in_width, std::vector<char> const &in_cube, uint32_t out_width, std::vector<char> &out_cube);


//for pbr shading
void convolve_cubemap_ggx(uint32_t in_width, std::vector<vec3> const &env, uint32_t out_width, std::vector<vec3> &output, float roughness);

//for the brdf lut
vec2 integrate_brdf(float roughness, float NoV);

