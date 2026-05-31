
#include "VK.hpp"
#include "TextureSystem.hpp"
#include "image_helpers.hpp"

#include <GLFW/glfw3.h>

#include <array>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <corecrt_math_defines.h>

#include <random>
#include <chrono>

TextureSystem::TextureSystem(RTG &rtg_, S72 const &scene72_)
	: rtg(rtg_), scene72(scene72_){}

TextureSystem::~TextureSystem() {
	if (texture_sampler) {
		vkDestroySampler(rtg.device, texture_sampler, nullptr);
		texture_sampler = VK_NULL_HANDLE;
	}
	if (depth_texture_sampler) {
		vkDestroySampler(rtg.device, depth_texture_sampler, nullptr);
		depth_texture_sampler = VK_NULL_HANDLE;
	}
	for (VkImageView &view : texture_views) {
		vkDestroyImageView(rtg.device, view, nullptr);
		view = VK_NULL_HANDLE;
	}
	texture_views.clear();
	for (auto &tex : textures) {
		rtg.helpers.destroy_image(std::move(tex));
	}
	textures.clear();
}


void TextureSystem::make_flat_image(S72::Texture const &texture){
	assert(texture.type == S72::Texture::Type::flat);
	//make a place for the texture to live on the GPU:
	textures.emplace_back(rtg.helpers.create_image(
		VkExtent2D{ .width = texture.width , .height = texture.height }, //size of image, we know this is not a cubemap, could be unsquare
		texture.format == S72::Texture::Format::srgb ? VK_FORMAT_R8G8B8A8_SRGB: VK_FORMAT_R8G8B8A8_UNORM, //how to interpret image data 
		VK_IMAGE_TILING_OPTIMAL,
		VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, //will sample and upload
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, //should be device-local
		Helpers::Unmapped, false
	));
	
	//transfer data:
	rtg.helpers.transfer_to_image(texture.data.data(), texture.data.size(), textures.back());//here size is in bytes
}

void TextureSystem::make_cube_image(S72::Texture const &texture){
	assert(texture.type == S72::Texture::Type::cube);
	//make a place for the texture to live on the GPU:
	textures.emplace_back(rtg.helpers.create_image(
		VkExtent2D{ .width = texture.width , .height = texture.width }, //size of image, cubemap should have it square
		VK_FORMAT_R32G32B32A32_SFLOAT, //how to interpret image data (in this case, float rgba)
		VK_IMAGE_TILING_OPTIMAL,
		VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, //will sample and upload
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, //should be device-local
		Helpers::Unmapped, true, (uint32_t)texture.mip_data.size() + 1 // base level plus mip levels
	));
	//TODO: also get the mip map images chained to the same image

	//transfer data
	//convert from rgbe to rgbafloat
	std::vector<float> original_rgba(texture.width * texture.width * 6 * 4);
	rgbe_to_rgba_float(texture.data, original_rgba, texture.width);

	uint32_t const mip_levels = (uint32_t)texture.mip_data.size() + 1;
	std::vector<std::vector<float>> mip_rgbas(mip_levels);
	for(uint32_t i = 0; i < mip_levels; i++){
		if(i == 0){
			mip_rgbas[i] = original_rgba;
		}
		else{
			uint32_t cur_width = texture.width >> i;
			mip_rgbas[i].resize(cur_width * cur_width * 6 * 4);
			rgbe_to_rgba_float(texture.mip_data[i - 1], mip_rgbas[i], cur_width);
		}
	}
	//get the prefiltered levels into the mip levels
	rtg.helpers.transfer_to_image(mip_rgbas, mip_rgbas[0].size() * sizeof(float), textures.back());//here size is in bytes
}

void TextureSystem::make_image_view(VkImageView &image_view, Helpers::AllocatedImage const & image){
	VkImageViewCreateInfo create_info{
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.flags = 0,
		.image = image.handle,
		.viewType = image.is_cubemap ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D,
		.format = image.format,
		// .components sets swizzling and is fine when zero-initialized
		.subresourceRange{
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = 0,
			.levelCount = image.mip_levels,
			.baseArrayLayer = 0,
			.layerCount = image.is_cubemap ? 6u : 1u,
		},
	};
	VK( vkCreateImageView(rtg.device, &create_info, nullptr, &image_view) );
}

void TextureSystem::make_image_view_3D(VkImageView &image_view, Helpers::AllocatedImage3D const & image){
	VkImageViewCreateInfo create_info{
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.flags = 0,
		.image = image.handle,
		.viewType = VK_IMAGE_VIEW_TYPE_3D,
		.format = image.format,
		// .components sets swizzling and is fine when zero-initialized
		.subresourceRange{
			.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
			.baseMipLevel = 0,
			.levelCount = 1u,
			.baseArrayLayer = 0,
			.layerCount = 1u,
		},
	};
	VK( vkCreateImageView(rtg.device, &create_info, nullptr, &image_view));
}

void TextureSystem::make_one_off_texture(TextureType t_type, std::variant<vec3, float> value){
	{//actually make the one-off texture:
		//make a place for the texture to live on the GPU:
		auto create_and_emplace_image = [&](VkFormat format, bool is_cube = false, uint32_t mip_levels = 1) {
			textures.emplace_back(rtg.helpers.create_image(
				VkExtent2D{ .width = 1 , .height = 1 }, //size of image
				format, //how to interpret image data (in this case, SRGB-encoded 8-bit RGBA)
				VK_IMAGE_TILING_OPTIMAL,
				VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, //will sample and upload
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, //should be device-local
				Helpers::Unmapped,
				is_cube,
				mip_levels
			));
		};
		

		if(t_type == TextureType::ALBEDO){
			S72::color c = std::get<S72::color>(value);
			uint8_t r = uint8_t(std::round(255 * S72::sRGB(c.x)));
			uint8_t g = uint8_t(std::round(255 * S72::sRGB(c.y)));
			uint8_t b = uint8_t(std::round(255 * S72::sRGB(c.z)));
			uint8_t a = 0xff;
			uint32_t data = uint32_t(r) | (uint32_t(g) << 8) | (uint32_t(b) << 16) | (uint32_t(a) << 24);

			create_and_emplace_image(VK_FORMAT_R8G8B8A8_SRGB);
			rtg.helpers.transfer_to_image(&data, sizeof(data), textures.back());//literally 4 bytes
		}
		else if(t_type == TextureType::ROUGHNESS || t_type == TextureType::METALNESS){
			float v = std::get<float>(value);
			float data[4] = {v, v, v, 1.0f};
			create_and_emplace_image(VK_FORMAT_R32G32B32A32_SFLOAT);
			rtg.helpers.transfer_to_image(&data, sizeof(float) * 4, textures.back());
		}
		else if(t_type == TextureType::NORMAL){
			vec3 normal = std::get<vec3>(value);
			float data[4] = {normal.x, normal.y,normal.z, 1.0f};
			create_and_emplace_image(VK_FORMAT_R32G32B32A32_SFLOAT);
			rtg.helpers.transfer_to_image(&data, sizeof(float) * 4, textures.back());
		}
		else if(t_type == TextureType::ENV){//black, should be unused
			float data[6 * 4] = {0.0f, 0.0f, 0.0f, 0.0f};
			create_and_emplace_image(VK_FORMAT_R32G32B32A32_SFLOAT, true, 1);
			rtg.helpers.transfer_to_image(&data, sizeof(float) * 4 * 6, textures.back());
		}
		else if(t_type == TextureType::ENV_MIP){//black, should be unused
			float data[6 * 4] = {0.0f, 0.0f, 0.0f, 0.0f};
			create_and_emplace_image(VK_FORMAT_R32G32B32A32_SFLOAT, true, 1);
			rtg.helpers.transfer_to_image(&data, sizeof(float) * 4 * 6, textures.back());
		}
		else if(t_type == TextureType::BRDF_LUT){//black, should be unused
			uint32_t data = 0;
			create_and_emplace_image(VK_FORMAT_R8G8B8A8_UNORM);
			rtg.helpers.transfer_to_image(&data, sizeof(uint32_t), textures.back());
		}
	}

	//also make image view for it
	VkImageView image_view = VK_NULL_HANDLE;
	make_image_view(image_view, textures.back());
	texture_views.emplace_back(image_view);

	assert(texture_views.size() == textures.size());
}


void TextureSystem::load_textures(){

	//create a map from texture name to texture number

	textures.reserve(scene72.textures.size());
	texture_index = 0;
	{ //texture 0 will be a dark grey / light grey checkerboard with a red square at the origin.

		//actually make the texture:
		uint32_t size = 128;
		std::vector< uint32_t > data;
		data.reserve(size * size);
		for (uint32_t y = 0; y < size; ++y) {
			float fy = (y + 0.5f) / float(size);
			for (uint32_t x = 0; x < size; ++x) {
				float fx = (x + 0.5f) / float(size);
				//highlight the origin:
				if      (fx < 0.05f && fy < 0.05f) data.emplace_back(0xff0000ff); //red
				else if ( (fx < 0.5f) == (fy < 0.5f)) data.emplace_back(0xff444444); //dark grey
				else data.emplace_back(0xffbbbbbb); //light grey
			}
		}
		assert(data.size() == size*size);

		//make a place for the texture to live on the GPU
		textures.emplace_back(rtg.helpers.create_image(
			VkExtent2D{ .width = size , .height = size }, //size of image
			VK_FORMAT_R8G8B8A8_UNORM, //how to interpret image data (in this case, linearly-encoded 8-bit RGBA)
			VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, //will sample and upload
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, //should be device-local
			Helpers::Unmapped
		));
		//transfer data:
		rtg.helpers.transfer_to_image(data.data(), sizeof(data[0]) * data.size(), textures.back());
		texture_index++;
	}
	{ //texture 1 will be a classic 'xor' texture:

		//actually make the texture:
		uint32_t size = 256;
		std::vector< uint32_t > data;
		data.reserve(size * size);
		for (uint32_t y = 0; y < size; ++y) {
			for (uint32_t x = 0; x < size; ++x) {
				uint8_t r = uint8_t(x) ^ uint8_t(y);
				uint8_t g = uint8_t(x + 128) ^ uint8_t(y);
				uint8_t b = uint8_t(x) ^ uint8_t(y + 27);
				uint8_t a = 0xff;
				data.emplace_back( uint32_t(r) | (uint32_t(g) << 8) | (uint32_t(b) << 16) | (uint32_t(a) << 24) );
			}
		}
		assert(data.size() == size*size);

		//make a place for the texture to live on the GPU:
		textures.emplace_back(rtg.helpers.create_image(
			VkExtent2D{ .width = size , .height = size }, //size of image
			VK_FORMAT_R8G8B8A8_SRGB, //how to interpret image data (in this case, SRGB-encoded 8-bit RGBA)
			VK_IMAGE_TILING_OPTIMAL,
			VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, //will sample and upload
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, //should be device-local
			Helpers::Unmapped
		));
		
		//transfer data:
		rtg.helpers.transfer_to_image(data.data(), sizeof(data[0]) * data.size(), textures.back());
		texture_index++;
	}

	{//make some default textures for normal, displacement
		//TODO
	}

	{//go through the textures to get the loaded textures
		for(auto const &tex: scene72.textures){
			S72::Texture texture = tex.second;
			std::cout<<"loading texture "<<texture.src<<std::endl;
			//check for multiple references of the same texture
			std::string texture_unique_key = texture.src;

			if(textures_name_to_index.find(texture_unique_key) != textures_name_to_index.end()){
				std::cerr<<"duplicate texture in the list of textures "<<texture_unique_key<<std::endl;
			}

			assert(texture_index == textures.size());
			textures_name_to_index[texture_unique_key] = texture_index;//record this loaded texture
			texture.type == S72::Texture::Type::cube ? make_cube_image(texture): make_flat_image(texture);//make texture and put into textures
			texture_index++;//now texture_index is the next texture
		}		
	}


	{//assign the environment name if it's already loaded in the previous all-textures pass
		assert(scene72.environments.size() <= 1);
		if(scene72.environments.size() == 1){
			environment_name = scene72.environments.begin()->second.name;
			assert(textures_name_to_index.find(environment_name) != textures_name_to_index.end());
			std::cout<<"assigning environment to be "<< environment_name <<std::endl;
		}
	}

	{//make texture for pre_computed LUTS
		//get the path to the LUTs 
		if(scene72.environments.size() > 0){//lambertian irradiance
			//load		
			std::filesystem::path p(scene72.environments.begin()->second.radiance->path);//random texture file
			std::string filepath = (p.parent_path() / (p.stem().string() + "_mine.lambertian" + p.extension().string())).string();
			lambertian_irradiance_lut_name= p.stem().string() + "_mine.lambertian" + p.extension().string();

			std::cout<<"loading "<< lambertian_irradiance_lut_name <<" ...";
			S72::Texture lambertian_irradiance_cubemap{
				.src = lambertian_irradiance_lut_name, 
				.type = S72::Texture::Type::cube, 
				.format = S72::Texture::Format::rgbe,
				.path = filepath,
			};
			
			loadTextureFile(filepath, lambertian_irradiance_cubemap.width, lambertian_irradiance_cubemap.height, lambertian_irradiance_cubemap.data);
			std::cout<<"success: width "<< lambertian_irradiance_cubemap.width << "| height "<<lambertian_irradiance_cubemap.height<<std::endl;
			//put into textures
			assert(texture_index == textures.size());
			textures_name_to_index[lambertian_irradiance_lut_name] = texture_index;//record this loaded texture
			assert(lambertian_irradiance_cubemap.mip_data.size() == 0);//no mip maps
			make_cube_image(lambertian_irradiance_cubemap);//make texture and put into textures
			texture_index++;//now texture_index is the next texture
		}
		

		if(scene72.environments.size() > 0){//brdf lut
			//load		
			std::filesystem::path p(scene72.textures.begin()->second.path);//random file from the 
			std::string filepath = (p.parent_path() / ("brdf_lut" + p.extension().string())).string();
			brdf_lut_name = "brdf_lut" + p.extension().string();
			if(textures_name_to_index.find(brdf_lut_name) == textures_name_to_index.end()){
				std::cerr<<"can't find "<<brdf_lut_name<<std::endl;
			}
			std::cout<<"loading "<< filepath <<" ...";
			S72::Texture brdf_lut{
				.src = brdf_lut_name, 
				.type = S72::Texture::Type::flat, 
				.format = S72::Texture::Format::linear,
				.path = filepath,
			};
			loadTextureFile(filepath, brdf_lut.width, brdf_lut.height, brdf_lut.data);
			std::cout<<"success: width "<< brdf_lut.width << "| height "<<brdf_lut.height<<std::endl;
			//put into textures
			assert(texture_index == textures.size());
			textures_name_to_index[brdf_lut_name] = texture_index;//record this loaded texture
			make_flat_image(brdf_lut);//make texture and put into textures
			texture_index++;//now texture_index is the next texture
			assert(texture_index == textures.size());
		}
		
	}

	{ //make image views for the textures
		texture_views.reserve(textures.size());
		for (Helpers::AllocatedImage const &image : textures) {
			VkImageView image_view = VK_NULL_HANDLE;
			make_image_view(image_view, image);
			texture_views.emplace_back(image_view);
		}
		assert(texture_views.size() == textures.size());
	}

	{//make one off dummy textures for environment, lambertian environment and brdf lut
		if(scene72.environments.size() == 0){

			//environment
			assert(texture_index == textures.size());
			make_one_off_texture(TextureType::ENV_MIP, vec3(1.0f,1.0f,1.0f));		
			environment_name = "ONE_OFF_ENVIRONMENT";
			textures_name_to_index[environment_name] = texture_index;
			texture_index++;//now texture_index is the next texture	

			//lambertian environment
			assert(texture_index == textures.size());
			make_one_off_texture(TextureType::ENV, 1.0f);//make texture and put into textures
			lambertian_irradiance_lut_name = "ONE_OFF_LAMBERTIAN_LUT";
			textures_name_to_index[lambertian_irradiance_lut_name] = texture_index;//record this loaded texture
			texture_index++;//now texture_index is the next texture


			//brdf lut
			assert(texture_index == textures.size());
			make_one_off_texture(TextureType::BRDF_LUT, 0.0f);//make texture and put into textures		
			brdf_lut_name = "ONE_OFF_BRDF_LUT";
			textures_name_to_index[brdf_lut_name] = texture_index;//record this loaded texture		
			texture_index++;//now texture_index is the next texture
		}
	}

	{ //make a sampler for the 2D textures
		VkSamplerCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.flags = 0,
			.magFilter = VK_FILTER_LINEAR,
			.minFilter = VK_FILTER_LINEAR,
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.mipLodBias = 0.0f,
			.anisotropyEnable = VK_FALSE,
			.maxAnisotropy = 0.0f, //doesn't matter if anisotropy isn't enabled
			.compareEnable = VK_FALSE,
			.compareOp = VK_COMPARE_OP_ALWAYS, //doesn't matter if compare isn't enabled
			.minLod = 0.0f,
			.maxLod = VK_LOD_CLAMP_NONE,
			.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
			.unnormalizedCoordinates = VK_FALSE,
		};
		VK( vkCreateSampler(rtg.device, &create_info, nullptr, &texture_sampler) );
	}


	
	
}


