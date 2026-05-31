#pragma once
#include "TextureSystem.hpp"
#include "ObjectPipelines.hpp"
#include <string>
#include <unordered_map>
#include <vector>

struct MaterialSystem : TextureSystem {
	MaterialSystem(RTG &rtg, S72 const &scene72,
	               LambertianObjectsPipeline &lambertian_objects_pipeline,
	               EnvMirrorObjectsPipeline &env_mirror_objects_pipeline,
	               PbrObjectsPipeline &pbr_objects_pipeline);
	~MaterialSystem();

	// rtg and scene72 inherited from TextureSystem

	LambertianObjectsPipeline &lambertian_objects_pipeline;
	EnvMirrorObjectsPipeline &env_mirror_objects_pipeline;
	PbrObjectsPipeline &pbr_objects_pipeline;

	enum class MaterialType {
		LAMBERTIAN = 1,
		ENVMIRROR = 2,
		PBR = 3,
	};

	std::unordered_map<std::string, uint32_t> material_texture_descriptor_set_table;
	VkDescriptorPool texture_descriptor_pool = VK_NULL_HANDLE;
	std::vector<VkDescriptorSet> texture_descriptor_sets;

	void load_materials();
	void make_descriptor_set(std::string mat_name, MaterialType mat_type, Texture_Indices tex_inds);
};
