#pragma once
#include <string>
#include "S72.hpp"
#include <unordered_map>
#include <variant>
#include <vector>
#include "mat4.hpp"
#include "RTG.hpp"
#include "Helpers.hpp"

struct TextureSystem {
	TextureSystem(RTG &rtg, S72 const &scene72);
	~TextureSystem();

	RTG &rtg;
	S72 const &scene72;

	struct Texture_Indices {
		int normal_index = -1;
		int displacement_index = -1;
		int albedo_index = -1;
		int roughness_index = -1;
		int metalness_index = -1;
		int environment_index = -1;
		int brdf_lut_index = -1;
		int diffuse_irradiance_index = -1;
	};

	int normal_default = -1;
	int disp_default = -1;

	std::string brdf_lut_name;
	std::string lambertian_irradiance_lut_name;
	std::string environment_name;

	std::unordered_map<std::string, uint32_t> textures_name_to_index;

	//tracks how many textures have been allocated; shared between load_textures and load_materials
	uint32_t texture_index = 0;

	void load_textures();
	void make_flat_image(S72::Texture const &texture);
	void make_cube_image(S72::Texture const &texture);
	void make_image_view(VkImageView &image_view, Helpers::AllocatedImage const &image);
	void make_image_view_3D(VkImageView &image_view, Helpers::AllocatedImage3D const &image);

	enum class TextureType {
		ALBEDO = 1,
		ROUGHNESS = 2,
		METALNESS = 3,
		NORMAL = 4,
		ENV = 5,
		ENV_MIP = 6,
		BRDF_LUT = 7,
	};
	void make_one_off_texture(TextureType t_type, std::variant<vec3, float> value);

	//textures and texture_views are indexed the same
	std::vector<Helpers::AllocatedImage> textures;
	std::vector<VkImageView> texture_views;
	VkSampler texture_sampler = VK_NULL_HANDLE;
	VkSampler depth_texture_sampler = VK_NULL_HANDLE;
};
