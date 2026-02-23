#include "image_helpers.hpp"
#include "stb_image_write.h"

#include "mat4.hpp"

#include <iostream>
#include <string>
#include <assert.h>


struct Sample {
    vec3 direction;
    float solid_angle;
    vec3 radiance;
};
float cube_texel_solid_angle(uint32_t x, uint32_t y, uint32_t width){
    float inv = 1.0f / width;

    float u = (2.0f * (x + 0.5f) * inv) - 1.0f;
    float v = (2.0f * (y + 0.5f) * inv) - 1.0f;

    float texel_area = 4.0f * inv * inv;  // = 4 / width^2

    float jacobian = 1.0f / pow(1.0f + u*u + v*v, 1.5f);

    return texel_area * jacobian;
}

vec3 cube_texel_direction(uint32_t face, uint32_t x, uint32_t y, uint32_t width){
    vec3 dir;
    float u = (2.0f * (x + 0.5f) / width) - 1.0f;
    float v = (2.0f * (y + 0.5f) / width) - 1.0f;
    v = -v;//following assignments assume uv is bottom-left origin per cube map face
    assert(face < 6);
    switch(face){
        case 0: // +X (right)
            dir = normalized(vec3( 1.0f,  v, -u));
            break;

        case 1: // -X (left)
            dir = normalized(vec3(-1.0f,  v,  u));
            break;

        case 2: // +Y (front)
            dir = normalized(vec3( u, 1.0f, -v));
            break;

        case 3: // -Y (back)
            dir = normalized(vec3( u, -1.0f,  v));
            break;

        case 4: // +Z (top)
            dir = normalized(vec3( u,  v, 1.0f));
            break;

        case 5: // -Z (bottom)
            dir = normalized(vec3( u, -v, -1.0f));
            break;
    }
    return dir;
}

int main(int argc, char **argv) {
    assert(argc == 4);
    std::string in_cube_path = argv[1];
    std::string utility_type = argv[2];//for A2, it's either --lambertian or --ggx 
    std::string out_cube_path = argv[3];

    //get all the input cube map
    uint32_t width;
    uint32_t _;//no need height cuz it's cubemap
    std::vector<char> in_data;
    loadTextureFile(in_cube_path, width, _, in_data);

    //convert to rgba floats
    std::vector<float> in_cube_rgba_float_data;//r32g32b32a32 sfloat (a is unused)
    in_cube_rgba_float_data.resize(6 * width * width * 4);
    rgbe_to_rgba_float(in_data, in_cube_rgba_float_data, width);

    //convert to radiance values
    std::vector<vec3> in_cube_radiance_values;
    in_cube_radiance_values.resize(width * width * 6);
    rgba_float_to_radiance_values(in_cube_rgba_float_data, in_cube_radiance_values, width * width * 6);

    //integrate over whole input cubemap for each output texel

    //precompute input texel directions
    std::vector<Sample> input_samples;
    input_samples.reserve(width * width * 6);

    for(uint32_t face = 0; face < 6; face++){
        for(uint32_t y = 0; y < width; y++){
            for(uint32_t x = 0; x < width; x++){
                vec3 dir = cube_texel_direction(face, x, y, width);
                float solid_angle = cube_texel_solid_angle(x,y, width);
                uint32_t index = face * width * width + y * width + x;
                vec3 L = in_cube_radiance_values[index];
                input_samples.emplace_back(Sample{.direction = dir, .solid_angle = solid_angle, .radiance = L});
            }
        }
    }

    //integrate over all the input samples for each out texel
    uint32_t out_width = 16;
    std::vector<vec3> out_radiance(6 * out_width * out_width);
    for(uint32_t face = 0; face < 6; ++face){
        for(uint32_t y = 0; y < out_width; ++y){
            for(uint32_t x = 0; x < out_width; ++x){

                vec3 n = cube_texel_direction(face, x, y, out_width);

                vec3 irradiance(0.0f);
                
                for(const auto& s : input_samples){

                    float cos_theta = dot(n, s.direction);

                    if(cos_theta > 0.0f){
                        irradiance = irradiance + s.radiance * cos_theta * s.solid_angle / 3.14159265f;
                    }
                }
                uint32_t index = face * out_width * out_width + y * out_width + x;
                out_radiance[index] = irradiance;
            }
        }
    }

    //output the image
    std::vector<char> out_image;
    out_image.resize(out_width * out_width * 6 * 4);
    radiance_values_to_rgbe(out_radiance, out_image, out_width * out_width * 6);

    stbi_write_png(out_cube_path.data(), out_width, out_width * 6, 4, out_image.data(), out_width * 4);

}

