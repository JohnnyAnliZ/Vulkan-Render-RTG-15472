#include "VK.hpp"
#include "MaterialSystem.hpp"
#include "image_helpers.hpp"

#include <array>
#include <cassert>
#include <iostream>


MaterialSystem::MaterialSystem(RTG &rtg_, S72 const &scene72_,
		LambertianObjectsPipeline &lambertian_objects_pipeline_,
		EnvMirrorObjectsPipeline &env_mirror_objects_pipeline_,
		PbrObjectsPipeline &pbr_objects_pipeline_)
	: TextureSystem(rtg_, scene72_),
	  lambertian_objects_pipeline(lambertian_objects_pipeline_),
	  env_mirror_objects_pipeline(env_mirror_objects_pipeline_),
	  pbr_objects_pipeline(pbr_objects_pipeline_) {}

MaterialSystem::~MaterialSystem() {
	if (texture_descriptor_pool) {
		vkDestroyDescriptorPool(rtg.device, texture_descriptor_pool, nullptr);
		texture_descriptor_pool = VK_NULL_HANDLE;
		texture_descriptor_sets.clear();
	}
}

void MaterialSystem::make_descriptor_set(std::string mat_name, MaterialType mat_type, Texture_Indices tex_inds){//allocate and write the texture descriptor sets
	//allocate descriptor set differing in layout
	VkDescriptorSetLayout *pDesSetLayout = nullptr;
	if(mat_type == MaterialType::ENVMIRROR) pDesSetLayout = &env_mirror_objects_pipeline.set2_TEXTURE;
	else if(mat_type == MaterialType::LAMBERTIAN) pDesSetLayout = &lambertian_objects_pipeline.set3_Texture;
	else if(mat_type == MaterialType::PBR) pDesSetLayout = &pbr_objects_pipeline.set2_TEXTURE;


	std::cout<<"making descriptor set"<<std::endl;

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
	
	//infos collects all the image views a material needs
	std::vector<VkDescriptorImageInfo> infos;
	auto add_info = [&](int index){
		infos.emplace_back(VkDescriptorImageInfo{
			.sampler = texture_sampler,
			.imageView = texture_views[index],
			.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
		});
	};
	uint32_t num_of_textures = 0;
	if(mat_type == MaterialType::LAMBERTIAN){
		assert(tex_inds.normal_index != -1);
		assert(tex_inds.albedo_index != -1);
		
		assert(textures[tex_inds.diffuse_irradiance_index].is_cubemap);
		num_of_textures = 3;
		infos.reserve(num_of_textures);
		add_info(tex_inds.normal_index);
		add_info(tex_inds.albedo_index);
		add_info(tex_inds.diffuse_irradiance_index);

	}
	else if(mat_type == MaterialType::ENVMIRROR){
		assert(tex_inds.normal_index != -1);
		assert(tex_inds.environment_index != -1);
		num_of_textures = 2;
		add_info(tex_inds.normal_index);
		add_info(tex_inds.environment_index);
	}
	else if(mat_type == MaterialType::PBR){
		assert(tex_inds.normal_index != -1);
		assert(tex_inds.albedo_index != -1);
		assert(tex_inds.roughness_index != -1);
		assert(tex_inds.metalness_index != -1);
		assert(tex_inds.environment_index != -1);
		assert(tex_inds.brdf_lut_index != -1);
		assert(tex_inds.diffuse_irradiance_index != -1);

		num_of_textures = 7;

		infos.reserve(num_of_textures);
		add_info(tex_inds.normal_index);
		add_info(tex_inds.albedo_index);
		add_info(tex_inds.roughness_index);
		add_info(tex_inds.metalness_index);
		add_info(tex_inds.environment_index);
		add_info(tex_inds.brdf_lut_index);
		add_info(tex_inds.diffuse_irradiance_index);

	}

	std::vector<VkWriteDescriptorSet>writes;
	writes.resize(num_of_textures); 
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


void MaterialSystem::load_materials(){
    	{ //create the texture descriptor pool
		uint32_t per_workspace = uint32_t(rtg.workspaces.size());

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
		//descriptors each material uses including the shadow atlas
		uint32_t lambertian_material_descriptors = 4;
		uint32_t env_mirror_material_descriptors = 2;
		uint32_t pbr_material_descriptors = 8;
		
		uint32_t volume_sampler_descriptors = 2;
		std::array< VkDescriptorPoolSize, 1> pool_sizes{
			VkDescriptorPoolSize{
				.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.descriptorCount = 
					num_lambertian * lambertian_material_descriptors + 
					num_envmirror * env_mirror_material_descriptors + 
					num_pbr * pbr_material_descriptors +
					volume_sampler_descriptors
			}
		};

		VkDescriptorPoolCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.flags = 0, //because CREATE_FREE_DESCRIPTOR_SET_BIT isn't included, *can't* free individual descriptors allocated from this pool
			.maxSets = num_lambertian * (1+per_workspace)   + num_envmirror + num_pbr * (1+per_workspace) + volume_sampler_descriptors, //lambertian and pbr have shadow map
			.poolSizeCount = uint32_t(pool_sizes.size()),
			.pPoolSizes = pool_sizes.data(),
		};
		std::cout<<"texture descriptor pool has max count"<<num_lambertian * (1+per_workspace)   + num_envmirror + num_pbr * (1+per_workspace)<<std::endl;
		VK( vkCreateDescriptorPool(rtg.device, &create_info, nullptr, &texture_descriptor_pool) );	
	}

	//the rest of the textures are loaded into memory from image files
	for(auto const &mat: scene72.materials){

		std::cout<<"loading material "<<mat.first<<std::endl;

		//normal map 
		int normal_ind = -1;
		{//get index to the material's normal map
			if(mat.second.normal_map == nullptr){
				make_one_off_texture(TextureType::NORMAL, vec3(0.5f,0.5f,1.0f));
				assert(texture_index == textures.size() - 1);
				normal_ind = (int)texture_index;//store index for albedo
				texture_index++;//now texture_index is the next texture		
			}
			else{
				std::string normal_map_name = mat.second.normal_map->src;
				if(textures_name_to_index.find(normal_map_name) != textures_name_to_index.end()){//the texture has already been emplaced into the texture vector
					std::cout<<"adding normal map "<<normal_map_name<<" to material "<<mat.first<<std::endl;
					normal_ind = (int)textures_name_to_index.at(normal_map_name);//store index for albedo
				}
				else{
					std::cerr<<"No texture ?"<<normal_map_name<<std::endl;
				}
			}
		}
		std::cout<<"made normal map yipee"<<std::endl;
		//----actually load the texture
		//resolve the brdf
		std::variant<S72::Material::PBR, S72::Material::Lambertian, S72::Material::Mirror, S72::Material::Environment> const &v = mat.second.brdf;
		if(std::holds_alternative<S72::Material::PBR>(v)){
			//for each texture to bind, either located using the texture-name-to-index table, or generate one
			//then make descriptor_set mapping the name of the material (mat.first) to the index into the vector of descriptor sets
			S72::Material::PBR pbr = std::get<S72::Material::PBR>(v);
			//collect the texture indices, gotta collect them all ~
			int albedo_ind = -1;
			int roughness_ind = -1;
			int metalness_ind = -1;
			int brdf_lut_ind = -1;
			int diffuse_irradiance_ind = -1;
			int environment_ind = -1;
			
			//albedo
			if(std::holds_alternative<S72::color>(pbr.albedo)){
				//make a one off texture
				make_one_off_texture(TextureType::ALBEDO, std::get<vec3>(pbr.albedo));
				//put into descriptor set
				assert(texture_index == textures.size() - 1);
				albedo_ind = (int)texture_index;//store index for albedo
				texture_index++;//now texture_index is the next texture		
			}
			else if(std::holds_alternative<S72::Texture *>(pbr.albedo)){
				S72::Texture * alb = std::get<S72::Texture *>(pbr.albedo);
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
				make_one_off_texture(TextureType::ROUGHNESS, std::get<float>(pbr.roughness));
				//put into descriptor set
				assert(texture_index == textures.size() - 1);
				roughness_ind = (int)texture_index;//store index for albedo
				texture_index++;//now texture_index is the next texture		
			}
			else if(std::holds_alternative<S72::Texture *>(pbr.roughness)){
				S72::Texture * rough = std::get<S72::Texture *>(pbr.roughness);
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
				make_one_off_texture(TextureType::METALNESS, std::get<float>(pbr.metalness));
				//put into descriptor set
				assert(texture_index == textures.size() - 1);
				metalness_ind = (int)texture_index;//store index for albedo
				texture_index++;//now texture_index is the next texture		
			}
			else if(std::holds_alternative<S72::Texture *>(pbr.metalness)){
				S72::Texture * metal = std::get<S72::Texture *>(pbr.metalness);
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
				.normal_index = normal_ind,
				.albedo_index = albedo_ind,
				.roughness_index = roughness_ind,
				.metalness_index = metalness_ind,
				.environment_index = environment_ind,
				.brdf_lut_index = brdf_lut_ind,
				.diffuse_irradiance_index = diffuse_irradiance_ind,
			});
		}
		else if(std::holds_alternative<S72::Material::Lambertian>(v)){
			S72::Material::Lambertian const &lamb = std::get<S72::Material::Lambertian>(v);

			//collect the texture indices
			int albedo_ind = -1;
			if(std::holds_alternative<S72::color>(lamb.albedo)){
				//make a one off texture
				make_one_off_texture(TextureType::ALBEDO, std::get<vec3>(lamb.albedo));
				//put into descriptor_set
				assert(texture_index == textures.size() - 1);
				albedo_ind = (int) texture_index;
				texture_index++;//now texture_index is the next texture		
			}
			else if(std::holds_alternative<S72::Texture*>(lamb.albedo)){
				S72::Texture * tex_ptr = std::get<S72::Texture*>(lamb.albedo);

				//check for multiple references of the same texture
				std::string texture_unique_key = tex_ptr->src;
				if(textures_name_to_index.find(texture_unique_key) != textures_name_to_index.end()){//the texture has already been emplaced into the texture vector
					std::cout<<"making lambertian material with loaded texture "<<texture_unique_key<<std::endl;
					albedo_ind = textures_name_to_index.at(texture_unique_key);
				}
				else{
					std::cerr<<"No texture ?"<<texture_unique_key<<std::endl;
				}
			}

			//make the descriptor
			make_descriptor_set(mat.first, MaterialType::LAMBERTIAN, 
				Texture_Indices{
					.normal_index = normal_ind,
					.albedo_index = (int)albedo_ind, 
					.diffuse_irradiance_index = (int)textures_name_to_index[lambertian_irradiance_lut_name],
				}
			);	
		}
		else if(std::holds_alternative<S72::Material::Mirror>(v)||std::holds_alternative<S72::Material::Environment>(v)){			
			if(textures_name_to_index.find(environment_name) != textures_name_to_index.end()){
				std::cout<<"making envmirror material with loaded texture "<<environment_name<<std::endl;
				make_descriptor_set(
					mat.first, 
					MaterialType::ENVMIRROR, 
					Texture_Indices{
						.normal_index = normal_ind,
						.environment_index = (int)textures_name_to_index.at(environment_name)
					}
				);
			}
			else{
				std::cerr<<"can't find texture: "<<environment_name<<std::endl;
			}			
		}
	}
}