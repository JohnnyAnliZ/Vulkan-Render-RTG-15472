#include "cube_utilities.hpp"
#include "stb_image_write.h"

#include "mat4.hpp"

#include <iostream>
#include <filesystem>
#include <string>
#include <assert.h>
#include <array>

int main(int argc, char **argv) {
    assert(argc == 4);
    std::string in_cube_path = argv[1];
    std::string utility_type = argv[2];//for A2, it's either --lambertian or --ggx 
    std::string out_cube_path = argv[3];

    std::filesystem::path p(out_cube_path);
    

    {//do the brdf LUT
        std::vector<vec3> outputs;
        uint32_t const width = 512;
        outputs.reserve(width * width);
        for(uint32_t x = 0; x < width; x++){
            for(uint32_t y = 0; y < width; y++){
                float roughness = (float)y / (float)(width - 1);
                float cos_theta = (float)x / (float)(width - 1);
                vec2 out = integrate_brdf(roughness, cos_theta);
                outputs.emplace_back(vec3(out.x, out.y, 0.0f));
            }
        }

        std::vector<unsigned char> image(width * width * 3);

        for (uint32_t i = 0; i < width * width; ++i) {
            image[3*i + 0] = (unsigned char)(outputs[i].x * 255.0f);
            image[3*i + 1] = (unsigned char)(outputs[i].y * 255.0f);
            image[3*i + 2] = 0;
        }

        std::string lut_path = (p.parent_path() / "brdf_lut.png").string();

        std::cout<<"writing: "<<lut_path<<std::endl;
        stbi_flip_vertically_on_write(1);
        int success = stbi_write_png(lut_path.data(), width, width, 3, image.data(), width * 3);
        if (!success) {
            fprintf(stderr, "Error: Failed to write image to brdf_lut.png\n");
            // Handle the error (e.g., free memory, return error code)
        } else {
            printf("Image written successfully.\n");
        }
        
    }
    

    
    if(utility_type == "--lambertian"){
        //get the input cube map
        uint32_t width;
        uint32_t _;//no need height cuz it's a cubemap
        std::vector<char> in_data;
        loadTextureFile(in_cube_path, width, _, in_data);
        
        //allocate space for out cube map
        uint32_t out_width = 16;
        std::vector<char> out_image;
        out_image.resize(out_width * out_width * 6 * 4);

        convolve_cubemap_diffuse(width, in_data, out_width, out_image);

        stbi_write_png(out_cube_path.data(), out_width, out_width * 6, 4, out_image.data(), out_width * 4);
    }


    
    else if(utility_type == "--ggx"){
        //get the input cube map
        uint32_t width;
        uint32_t _;//no need height cuz it's a cubemap
        std::vector<char> in_data;


        loadTextureFile(in_cube_path, width, _, in_data);


        //convert to radiance values
        std::vector<vec3> in_rads(width * width * 6);
        rgbe_to_radiance_values(in_data, in_rads, width);

        //decalre the mip images
        const uint32_t mip_levels = 6;
        std::array<std::vector<vec3>, mip_levels> mip_images;
        for(uint32_t level = 1; level < mip_levels; level++){
            float roughness = (float)level / (float)(mip_levels - 1);//roughness from zero (original image) to one, smallest mip map
            uint32_t cur_width = width / (1<<level);//if input is 1024, then width goes 512, 256, 128, 64, 32
            //allocate space
            mip_images[level].resize(cur_width * cur_width * 6);
            //convolve with ggx
            convolve_cubemap_ggx(width, in_rads, cur_width, mip_images[level], roughness);
            //convert back to rgbe
            std::vector<char> out_image(4 * cur_width * cur_width * 6);
            radiance_values_to_rgbe(mip_images[level], out_image, cur_width * cur_width * 6);
            //store

            // Construct new filename: parent_path / filename_without_ext + "." + level + ext
            std::string mip_image_name = (p.parent_path() / (p.stem().string() + "." + std::to_string(level) + p.extension().string())).string();

            std::cout << mip_image_name << std::endl;

            std::cout<<"outputting: "<<mip_image_name<<std::endl;
            stbi_flip_vertically_on_write(0);
            stbi_write_png(mip_image_name.data(), cur_width, cur_width * 6, 4, out_image.data(), cur_width * 4);
        }
    }

    
}

