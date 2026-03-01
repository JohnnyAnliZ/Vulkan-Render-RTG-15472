// Tutorial_scene.cpp
// This file contains functions that concerns the 
#include "Tutorial.hpp"
#include "VK.hpp"

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

void Tutorial::make_flat_image(S72::Texture const &texture){
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

void Tutorial::make_cube_image(S72::Texture const &texture){
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

void Tutorial::make_image_view(VkImageView &image_view, Helpers::AllocatedImage const & image){
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

void Tutorial::make_one_off_texture(TextureType t_type, std::variant<vec3, float> value){
	{//actually make the one-off texture:
		//make a place for the texture to live on the GPU:
		auto create_and_emplace_image = [&](VkFormat format) {
			textures.emplace_back(rtg.helpers.create_image(
				VkExtent2D{ .width = 1 , .height = 1 }, //size of image
				format, //how to interpret image data (in this case, SRGB-encoded 8-bit RGBA)
				VK_IMAGE_TILING_OPTIMAL,
				VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, //will sample and upload
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, //should be device-local
				Helpers::Unmapped
			));
		};
		

		if(t_type == TextureType::ALBEDO){
			S72::color c = get<S72::color>(value);
			uint8_t r = uint8_t(std::round(255 * S72::sRGB(c.x)));
			uint8_t g = uint8_t(std::round(255 * S72::sRGB(c.y)));
			uint8_t b = uint8_t(std::round(255 * S72::sRGB(c.z)));
			uint8_t a = 0xff;
			uint32_t data = uint32_t(r) | (uint32_t(g) << 8) | (uint32_t(b) << 16) | (uint32_t(a) << 24);

			create_and_emplace_image(VK_FORMAT_R8G8B8A8_SRGB);
			rtg.helpers.transfer_to_image(&data, sizeof(data), textures.back());//literally 4 bytes
		}
		else if(t_type == TextureType::ROUGHNESS || t_type == TextureType::METALNESS){
			float v = get<float>(value);
			float data[4] = {v, v, v, 1.0f};
			create_and_emplace_image(VK_FORMAT_R32G32B32A32_SFLOAT);
			rtg.helpers.transfer_to_image(&data, sizeof(float) * 4, textures.back());
		}
		else if(t_type == TextureType::NORMAL){
			vec3 normal = get<vec3>(value);
			create_and_emplace_image(VK_FORMAT_R32G32B32A32_SFLOAT);
			rtg.helpers.transfer_to_image(&normal, sizeof(float) * 3, textures.back());
		}
	}

	//also make image view for it
	VkImageView image_view = VK_NULL_HANDLE;
	make_image_view(image_view, textures.back());
	texture_views.emplace_back(image_view);

	assert(texture_views.size() == textures.size());

}


void Tutorial::load_textures(){

	//create a map from texture name to texture number

	textures.reserve(scene72.textures.size());
	uint32_t texture_index = 0;
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

		assert(scene72.environments.size() == 1);
		environment_name = scene72.environments.begin()->second.name;
		assert(textures_name_to_index.find(environment_name) != textures_name_to_index.end());
		std::cout<<"assigning environment to be "<< environment_name <<std::endl;
	}

	{//make texture for pre_computed LUTS
		//get the path to the LUTs 
		{//lambertian irradiance
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

		{//brdf lut
			//load		
			std::filesystem::path p(scene72.textures.begin()->second.path);//random texture file
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

	{ //make a sampler for the 2D textures
		VkSamplerCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.flags = 0,
			.magFilter = VK_FILTER_NEAREST,
			.minFilter = VK_FILTER_NEAREST,
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

	{ //create the texture descriptor pool
		uint32_t num_lambertian = 0;
		uint32_t num_envmirror= 0;
		uint32_t num_pbr = 0;
		for(auto const& mat : scene72.materials) {
			auto const& v = mat.second.brdf;
			if(std::holds_alternative<S72::Material::Lambertian>(v))
				num_lambertian++;
			else if(std::holds_alternative<S72::Material::Mirror>(v) ||
					std::holds_alternative<S72::Material::Environment>(v))
				num_envmirror++;
			else if(std::holds_alternative<S72::Material::PBR>(v))
				num_pbr++;
		}
		std::array< VkDescriptorPoolSize, 1> pool_sizes{
			VkDescriptorPoolSize{
				.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.descriptorCount = num_lambertian + num_envmirror + num_pbr * 6, //one descriptor per set, one set per texture
			},
		};
		
		VkDescriptorPoolCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.flags = 0, //because CREATE_FREE_DESCRIPTOR_SET_BIT isn't included, *can't* free individual descriptors allocated from this pool
			.maxSets = num_lambertian + num_envmirror + num_pbr, //one set per texture
			.poolSizeCount = uint32_t(pool_sizes.size()),
			.pPoolSizes = pool_sizes.data(),
		};

		VK( vkCreateDescriptorPool(rtg.device, &create_info, nullptr, &texture_descriptor_pool) );	
	}

	//the rest of the textures are loaded into memory from image files
	for(auto const &mat: scene72.materials){

		std::cout<<"loading material "<<mat.first<<std::endl;
		
		//----actually load the texture
		//resolve the brdf
		std::variant<S72::Material::PBR, S72::Material::Lambertian, S72::Material::Mirror, S72::Material::Environment> const &v = mat.second.brdf;
		if(std::holds_alternative<S72::Material::PBR>(v)){
			//TODO:
			//for each texture to bind, either located using the texture-name-to-index table, or generate one
			//then make descriptor_set mapping the name of the material (mat.first) to the index into the vector of descriptor sets
			S72::Material::PBR pbr = get<S72::Material::PBR>(v);
			int albedo_ind = -1;
			int roughness_ind = -1;
			int metalness_ind = -1;
			int brdf_lut_ind = -1;
			int diffuse_irradiance_ind = -1;
			int environment_ind = -1;
			
			//albedo
			if(std::holds_alternative<S72::color>(pbr.albedo)){
				//make a one off texture
				make_one_off_texture(TextureType::ALBEDO, get<vec3>(pbr.albedo));
				//put into descriptor set
				assert(texture_index == textures.size() - 1);
				albedo_ind = (int)texture_index;//store index for albedo
				texture_index++;//now texture_index is the next texture		
			}
			else if(std::holds_alternative<S72::Texture *>(pbr.albedo)){
				S72::Texture * alb = get<S72::Texture *>(pbr.albedo);
				std::string texture_unique_key = alb->src;
				if(textures_name_to_index.find(texture_unique_key) != textures_name_to_index.end()){//the texture has already been emplaced into the texture vector
					std::cout<<"making pbr material with loaded texture "<<texture_unique_key<<std::endl;
					albedo_ind = (int)textures_name_to_index.at(texture_unique_key);//store index for albedo
				}
				else{
					std::cerr<<"No texture ?"<<texture_unique_key<<std::endl;
				}
			}

			//roughness
			if(std::holds_alternative<float>(pbr.roughness)){
				//make a one off texture
				make_one_off_texture(TextureType::ROUGHNESS, get<float>(pbr.roughness));
				//put into descriptor set
				assert(texture_index == textures.size() - 1);
				roughness_ind = (int)texture_index;//store index for albedo
				texture_index++;//now texture_index is the next texture		
			}
			else if(std::holds_alternative<S72::Texture *>(pbr.roughness)){
				S72::Texture * rough = get<S72::Texture *>(pbr.roughness);
				std::string texture_unique_key = rough->src;
				if(textures_name_to_index.find(texture_unique_key) != textures_name_to_index.end()){//the texture has already been emplaced into the texture vector
					std::cout<<"making pbr material with loaded texture "<<texture_unique_key<<std::endl;
					roughness_ind = (int)textures_name_to_index.at(texture_unique_key);//store index for albedo
				}
				else{
					std::cerr<<"No texture ?"<<texture_unique_key<<std::endl;
				}
			}

			//metalness
			if(std::holds_alternative<float>(pbr.metalness)){
				//make a one off texture
				make_one_off_texture(TextureType::METALNESS, get<float>(pbr.metalness));
				//put into descriptor set
				assert(texture_index == textures.size() - 1);
				metalness_ind = (int)texture_index;//store index for albedo
				texture_index++;//now texture_index is the next texture		
			}
			else if(std::holds_alternative<S72::Texture *>(pbr.metalness)){
				S72::Texture * metal = get<S72::Texture *>(pbr.metalness);
				std::string texture_unique_key = metal->src;
				if(textures_name_to_index.find(texture_unique_key) != textures_name_to_index.end()){//the texture has already been emplaced into the texture vector
					std::cout<<"making pbr material with loaded texture "<<texture_unique_key<<std::endl;
					metalness_ind = (int)textures_name_to_index.at(texture_unique_key);//store index for albedo
				}
				else{
					std::cerr<<"No texture ?"<<texture_unique_key<<std::endl;
				}
			}
			
			//brdf_lut
			if(textures_name_to_index.find(brdf_lut_name) != textures_name_to_index.end()){//the texture has already been emplaced into the texture vector
				std::cout<<"making pbr material with loaded texture "<<brdf_lut_name<<std::endl;
				brdf_lut_ind = (int)textures_name_to_index.at(brdf_lut_name);//store index for albedo
			}
			else{
				std::cerr<<"No texture ?"<<brdf_lut_name<<std::endl;
			}

			//lambertian irradiance 
			if(textures_name_to_index.find(lambertian_irradiance_lut_name) != textures_name_to_index.end()){//the texture has already been emplaced into the texture vector
				std::cout<<"making pbr material with loaded texture "<<lambertian_irradiance_lut_name<<std::endl;
				diffuse_irradiance_ind = (int)textures_name_to_index.at(lambertian_irradiance_lut_name);//store index for albedo
			}
			else{
				std::cerr<<"No texture ?"<<lambertian_irradiance_lut_name<<std::endl;
			}

			//environment with it's mip levels
			if(textures_name_to_index.find(environment_name) != textures_name_to_index.end()){//the texture has already been emplaced into the texture vector
				std::cout<<"making pbr material with loaded texture "<<environment_name<<std::endl;
				environment_ind = (int)textures_name_to_index.at(environment_name);//store index for albedo
			}
			else{
				std::cerr<<"No texture ?"<<environment_name<<std::endl;
			}

			//make the descriptor set with all the texture indices
			make_descriptor_set(mat.first, MaterialType::PBR, Texture_Indices{
				.albedo_index = albedo_ind,
				.roughness_index = roughness_ind,
				.metalness_index = metalness_ind,
				.environment_index = environment_ind,
				.brdf_lut_index = brdf_lut_ind,
				.diffuse_irradiance_index = diffuse_irradiance_ind,
			});
		}
		else if(std::holds_alternative<S72::Material::Lambertian>(v)){
			S72::Material::Lambertian const &lamb = get<S72::Material::Lambertian>(v);
			if(std::holds_alternative<S72::color>(lamb.albedo)){
				//make a one off texture
				make_one_off_texture(TextureType::ALBEDO, get<vec3>(lamb.albedo));
				//put into descriptor_set
				assert(texture_index == textures.size() - 1);
				make_descriptor_set(mat.first, MaterialType::LAMBERTIAN, Texture_Indices{.albedo_index = (int)texture_index});
				texture_index++;//now texture_index is the next texture		
			}
			else if(std::holds_alternative<S72::Texture*>(lamb.albedo)){
				S72::Texture * tex_ptr = get<S72::Texture*>(lamb.albedo);

				//check for multiple references of the same texture
				std::string texture_unique_key = tex_ptr->src;
				if(textures_name_to_index.find(texture_unique_key) != textures_name_to_index.end()){//the texture has already been emplaced into the texture vector
					std::cout<<"making lambertian material with loaded texture "<<texture_unique_key<<std::endl;
					make_descriptor_set(mat.first, MaterialType::LAMBERTIAN, Texture_Indices{.albedo_index = (int)textures_name_to_index.at(texture_unique_key)});
				}
				else{
					std::cerr<<"No texture ?"<<texture_unique_key<<std::endl;
				}
			}	
		}
		else if(std::holds_alternative<S72::Material::Mirror>(v)||std::holds_alternative<S72::Material::Environment>(v)){			
			if(textures_name_to_index.find(environment_name) != textures_name_to_index.end()){
				std::cout<<"making envmirror material with loaded texture "<<environment_name<<std::endl;
				make_descriptor_set(mat.first, MaterialType::ENVMIRROR, Texture_Indices{.environment_index = (int)textures_name_to_index.at(environment_name)});
			}
			else{
				std::cerr<<"can't find texture: "<<environment_name<<std::endl;
			}			
		}
	}	
}




void Tutorial::make_descriptor_set(std::string mat_name, MaterialType mat_type, Texture_Indices tex_inds){//allocate and write the texture descriptor sets
	//allocate descriptor set differing in layout
	VkDescriptorSetLayout *pDesSetLayout = nullptr;
	if(mat_type == MaterialType::ENVMIRROR) pDesSetLayout = &env_mirror_objects_pipeline.set2_TEXTURE;
	else if(mat_type == MaterialType::LAMBERTIAN) pDesSetLayout = &lambertian_objects_pipeline.set2_TEXTURE;
	else if(mat_type == MaterialType::PBR) pDesSetLayout = &pbr_objects_pipeline.set2_TEXTURE;

	VkDescriptorSetAllocateInfo alloc_info{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = texture_descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = pDesSetLayout,
	};
	
	uint32_t ind = (uint32_t)texture_descriptor_sets.size();
	texture_descriptor_sets.emplace_back(VK_NULL_HANDLE);
	vkAllocateDescriptorSets(rtg.device, &alloc_info, &texture_descriptor_sets[ind]);
	material_texture_descriptor_set_table[mat_name] = ind;

	//write to the descriptor set based on the material type
	
	if(mat_type == MaterialType::LAMBERTIAN){
		assert(tex_inds.albedo_index != -1);
		std::array<VkDescriptorImageInfo, 1> infos{
			VkDescriptorImageInfo{
				.sampler = texture_sampler,
				.imageView = texture_views[tex_inds.albedo_index],
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			},
			//TODO, add normal map
		};
		
		VkWriteDescriptorSet write{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = texture_descriptor_sets[ind],//texture descriptors have the same index as textures
			.dstBinding = 0,
			.dstArrayElement = 0,
			.descriptorCount = (uint32_t)infos.size(),
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.pImageInfo = &infos[0],
		};
		vkUpdateDescriptorSets( rtg.device, 1, &write, 0, nullptr );
	}
	else if(mat_type == MaterialType::ENVMIRROR){
		assert(tex_inds.environment_index != -1);
		std::array<VkDescriptorImageInfo, 1> infos{
			VkDescriptorImageInfo{
				.sampler = texture_sampler,
				.imageView = texture_views[tex_inds.environment_index],
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			},
		};
		
		VkWriteDescriptorSet write{
			.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
			.dstSet = texture_descriptor_sets[ind],//texture descriptors have the same index as textures
			.dstBinding = 0,
			.dstArrayElement = 0,
			.descriptorCount = (uint32_t)infos.size(),
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.pImageInfo = &infos[0],
		};
		vkUpdateDescriptorSets( rtg.device, 1, &write, 0, nullptr );
	}
	else if(mat_type == MaterialType::PBR){
		assert(tex_inds.albedo_index != -1);
		assert(tex_inds.roughness_index != -1);
		assert(tex_inds.metalness_index != -1);
		assert(tex_inds.brdf_lut_index != -1);
		assert(tex_inds.diffuse_irradiance_index != -1);

		uint32_t const num_of_textures = 6;

		std::array<VkDescriptorImageInfo, num_of_textures> infos{
			VkDescriptorImageInfo{
				.sampler = texture_sampler,
				.imageView = texture_views[tex_inds.albedo_index],
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			},
			VkDescriptorImageInfo{
				.sampler = texture_sampler,
				.imageView = texture_views[tex_inds.roughness_index],
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			},
			VkDescriptorImageInfo{
				.sampler = texture_sampler,
				.imageView = texture_views[tex_inds.metalness_index],
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			},
			VkDescriptorImageInfo{
				.sampler = texture_sampler,
				.imageView = texture_views[tex_inds.environment_index],
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			},
			VkDescriptorImageInfo{
				.sampler = texture_sampler,
				.imageView = texture_views[tex_inds.brdf_lut_index],
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			},
			VkDescriptorImageInfo{
				.sampler = texture_sampler,
				.imageView = texture_views[tex_inds.diffuse_irradiance_index],
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			},
		};
		
		std::array<VkWriteDescriptorSet, num_of_textures>writes; 
		for(uint32_t i = 0; i < num_of_textures; i++){
			writes[i] = VkWriteDescriptorSet{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = texture_descriptor_sets[ind],//texture descriptors have the same index as textures
				.dstBinding = i,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &infos[i],
			};
		}
		vkUpdateDescriptorSets( rtg.device, num_of_textures, writes.data(), 0, nullptr );
	}
	
}