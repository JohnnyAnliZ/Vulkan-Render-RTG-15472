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


void Tutorial::load_textures(){
    { // make some textures 
		//create a map from texture name to texture number
		uint32_t reserve_size = 5;
		material_textures_table.reserve(reserve_size);
		textures.reserve(reserve_size);
		uint32_t texture_index = 0;
		{ //texture 0 will be a dark grey / light grey checkerboard with a red square at the origin.
			//insert_into lookup table
			material_textures_table["light_grey_checkerboard_with_red_square_origin"] = Tutorial::Texture_Indices{
				.albedo_index = (int)texture_index,
			};
			texture_index++;//now texture index is the next texture to be loaded
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
		}
		{ //texture 1 will be a classic 'xor' texture:
			//insert into color lookup table
			material_textures_table["classic_xor_texture"] = Tutorial::Texture_Indices{
				.albedo_index = (int)texture_index,
			};
			texture_index++;//now texture index is the next texture to be loaded
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
		}
		//the rest of the textures are loaded into memory from image files
		for(auto const &mat: scene72.materials){
			assert(material_textures_table.find(mat.first) == material_textures_table.end());
			std::cout<<"loading material "<<mat.first<<std::endl;
			
			//----actually load the texture
			//resolve the brdf
			std::variant<S72::Material::PBR, S72::Material::Lambertian, S72::Material::Mirror, S72::Material::Environment> const &v = mat.second.brdf;
			if(std::holds_alternative<S72::Material::PBR>(v)){

			}
			else if(std::holds_alternative<S72::Material::Lambertian>(v)){
				S72::Material::Lambertian const &lamb = get<S72::Material::Lambertian>(v);
				//----insert into material_textures_table
				assert(textures.size() == texture_index);//this index should be the newly inserted texture
				material_textures_table[mat.first] = Tutorial::Texture_Indices{
					.albedo_index = (int)texture_index,//albedo texture is the only texture descriptor it needs
				};
				// ^ here we assign it assuming textures at texture_index will be populated with a new texture, 
				//but if the texture is already loaded, overwrite with index to it kept track of by textures_name_to_index

				if(std::holds_alternative<S72::color>(lamb.albedo)){
					//actually make the texture:
					uint32_t size = 1;
					std::vector< uint32_t > data;
					data.reserve(size * size);

					S72::color c = get<S72::color>(lamb.albedo);
					uint8_t r = uint8_t(std::round(255 * S72::sRGB(c.x)));
					uint8_t g = uint8_t(std::round(255 * S72::sRGB(c.y)));
					uint8_t b = uint8_t(std::round(255 * S72::sRGB(c.z)));
					uint8_t a = 0xff;
					data.emplace_back( uint32_t(r) | (uint32_t(g) << 8) | (uint32_t(b) << 16) | (uint32_t(a) << 24) );
					assert(data.size() == size*size);

					std::cout<<"color of the lambertian material "<< std::setfill('0') << std::setw(8) << std::hex<<data.back()<<std::dec<<std::endl;
					//make a place for the texture to live on the GPU:
					textures.emplace_back(rtg.helpers.create_image(
						VkExtent2D{ .width = size , .height = size }, //size of image
						VK_FORMAT_R8G8B8A8_SRGB, //how to interpret image data (in this case, SRGB-encoded 8-bit RGBA)
						VK_IMAGE_TILING_OPTIMAL,
						VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, //will sample and upload
						VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, //should be device-local
						Helpers::Unmapped
					));
					texture_index++;//now texture_index is the next texture

					//transfer data:
					rtg.helpers.transfer_to_image(data.data(), sizeof(data[0]) * data.size(), textures.back());
				}
				else if(std::holds_alternative<S72::Texture*>(lamb.albedo)){
					S72::Texture * tex_ptr = get<S72::Texture*>(lamb.albedo);

					//check for multiple references of the same texture
					std::string texture_unique_key = tex_ptr->src;
					if(textures_name_to_index.find(texture_unique_key) == textures_name_to_index.end()){
						//record this loaded texture
						textures_name_to_index[texture_unique_key] = texture_index;
						
						// std::cout<<"texture name: "<<tex_ptr->path<<std::endl;
						// std::cout<<"size of the loaded texture "<<size<<std::endl;

                        assert(tex_ptr->type == S72::Texture::Type::flat);
						//make a place for the texture to live on the GPU:
						textures.emplace_back(rtg.helpers.create_image(
							VkExtent2D{ .width = tex_ptr->width , .height = tex_ptr->height }, //size of image, we know this is not a cubemap, could be unsquare
							tex_ptr->format == S72::Texture::Format::srgb ? VK_FORMAT_R8G8B8A8_SRGB: VK_FORMAT_R32G32B32A32_SFLOAT, //how to interpret image data (in this case, SRGB-encoded 8-bit RGBA)
							VK_IMAGE_TILING_OPTIMAL,
							VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, //will sample and upload
							VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, //should be device-local
							Helpers::Unmapped
						));
						texture_index++;//now texture_index is the next texture

						//transfer data:
						rtg.helpers.transfer_to_image(tex_ptr->data.data(), tex_ptr->data.size(), textures.back());//here size is in bytes
					}
					else{//the texture has already been emplaced into the texture vector
						material_textures_table[mat.first] = Tutorial::Texture_Indices{
							.albedo_index = (int)textures_name_to_index.at(texture_unique_key),
						};
						std::cout<<"Already emplaced into textures: "<<texture_unique_key<<std::endl;
						//no need to emplace new texture nor increment texture_index
					}			
				}	
			}
			else if(std::holds_alternative<S72::Material::Mirror>(v)||std::holds_alternative<S72::Material::Environment>(v)){
				std::string environment_name;
				if(std::holds_alternative<S72::Material::Environment>(v)){
					S72::Material::Environment const &env = get<S72::Material::Environment>(v);
					environment_name = env.name;
				}
				else if (std::holds_alternative<S72::Material::Mirror>(v)){
					S72::Material::Mirror const &mir = get<S72::Material::Mirror>(v);
					environment_name = mir.env_name;
				}
				
				if(textures_name_to_index.find(environment_name) == textures_name_to_index.end()){
					textures_name_to_index[environment_name] = texture_index;
					//----insert into material_textures_table
					assert(textures.size() == texture_index);//this index should be the newly inserted texture
					material_textures_table[mat.first] = Tutorial::Texture_Indices{
						.environment_index = (int)texture_index,//only need environment texture
					};
					texture_index++;
					
					std::cout<<"accessimg scene72.environments at: "<<environment_name<<std::endl;
					S72::Texture * tex_ptr = scene72.environments.at(environment_name).radiance;//this is supposed to be a rgbe cubemap
					assert(tex_ptr->type == S72::Texture::Type::cube && tex_ptr->format == S72::Texture::Format::rgbe);

					//decode rgbe to radiance value stored in r32g32b32a32 with a channel unused
					uint32_t width = tex_ptr->width;//knowing it's a cube, take only it's width
					std::vector<float> converted_data;
					converted_data.resize(4 * width * width * 6);//4 float channels, 6 cube faces
					rgbe_to_rgba_float(tex_ptr->data, converted_data, width);

					std::cout<<"texture name: "<<tex_ptr->path<<std::endl;
					std::cout<<"width of the loaded rgbe cubemap image "<<width<<std::endl;
					//make a place for the texture to live on the GPU:
					textures.emplace_back(rtg.helpers.create_image(
						VkExtent2D{ .width = width , .height = width }, //size of image
						VK_FORMAT_R32G32B32A32_SFLOAT, //Convert to RGB float, A is unused
						VK_IMAGE_TILING_OPTIMAL,
						VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, //will sample and upload
						VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, //should be device-local
						Helpers::Unmapped,
						true
					));

					//transfer data:
					assert(sizeof(converted_data[0]) == 4u);
					rtg.helpers.transfer_to_image(converted_data.data(), converted_data.size() * sizeof(converted_data[0]), textures.back(), true);//here size is in bytes
				}
				else{
					std::cout<<"already has "<<environment_name<<std::endl;
					material_textures_table[mat.first] = Tutorial::Texture_Indices{
						.environment_index = (int)textures_name_to_index.at(environment_name),//only need environment texture
					};
				}			
			}
		}
	}

	{ //make image views for the textures
		texture_views.reserve(textures.size());
		for (Helpers::AllocatedImage const &image : textures) {
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
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = image.is_cubemap ? 6u : 1u,
				},
			};

			VkImageView image_view = VK_NULL_HANDLE;
			VK( vkCreateImageView(rtg.device, &create_info, nullptr, &image_view) );

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
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.mipLodBias = 0.0f,
			.anisotropyEnable = VK_FALSE,
			.maxAnisotropy = 0.0f, //doesn't matter if anisotropy isn't enabled
			.compareEnable = VK_FALSE,
			.compareOp = VK_COMPARE_OP_ALWAYS, //doesn't matter if compare isn't enabled
			.minLod = 0.0f,
			.maxLod = 0.0f,
			.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
			.unnormalizedCoordinates = VK_FALSE,
		};
		VK( vkCreateSampler(rtg.device, &create_info, nullptr, &texture_sampler) );
	}
		
	{ //create the texture descriptor pool
		uint32_t per_texture = uint32_t(textures.size());
		std::array< VkDescriptorPoolSize, 1> pool_sizes{
			VkDescriptorPoolSize{
				.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.descriptorCount = 1 * 1 * per_texture, //one descriptor per set, one set per texture
			},
		};
		
		VkDescriptorPoolCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.flags = 0, //because CREATE_FREE_DESCRIPTOR_SET_BIT isn't included, *can't* free individual descriptors allocated from this pool
			.maxSets = 1 * per_texture, //one set per texture
			.poolSizeCount = uint32_t(pool_sizes.size()),
			.pPoolSizes = pool_sizes.data(),
		};

		VK( vkCreateDescriptorPool(rtg.device, &create_info, nullptr, &texture_descriptor_pool) );	
	}

	{ //allocate and write the texture descriptor sets
		//allocate the descriptors (using the same alloc_info):
		VkDescriptorSetAllocateInfo alloc_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = texture_descriptor_pool,
			.descriptorSetCount = 1,
			.pSetLayouts = &lambertian_objects_pipeline.set2_TEXTURE,
		};
		std::cout<<"allocating and writing "<<textures.size()<<" textures"<<std::endl;

		texture_descriptors.assign(textures.size(), VK_NULL_HANDLE);
		for (VkDescriptorSet &descriptor_set : texture_descriptors) {
			VK( vkAllocateDescriptorSets(rtg.device, &alloc_info, &descriptor_set) );
		}
		
		//write descriptors for textures:
		std::vector<VkDescriptorImageInfo> infos(textures.size());//presize so that the address doesn't change
		std::vector<VkWriteDescriptorSet> writes(textures.size());

		for (Helpers::AllocatedImage const &image : textures) {
			size_t i = &image - &textures[0];
			
			infos[i] = VkDescriptorImageInfo{
				.sampler = texture_sampler,
				.imageView = texture_views[i],
				.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			};
			writes[i] = VkWriteDescriptorSet{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = texture_descriptors[i],//texture descriptors have the same index as textures
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.pImageInfo = &infos[i],
			};
		}
		vkUpdateDescriptorSets( rtg.device, uint32_t(writes.size()), writes.data(), 0, nullptr );
	}
}




