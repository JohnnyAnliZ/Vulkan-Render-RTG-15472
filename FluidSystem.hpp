#pragma once
#include "ComputeSystem.hpp"
#include "MaterialSystem.hpp"
#include "SceneSystem.hpp"
#include "AddVectorSourcesPipeline.hpp"
#include "DiffuseVectorPipeline.hpp"
#include "AdvectVectorPipeline.hpp"
#include "ProjectionPipelines.hpp"
#include "DensityPipelines.hpp" 

struct FluidSystem {

    void init_fluid(RTG &rtg, ComputeContext &comp_context, MaterialSystem &material_system, SceneSystem &scene_system);
    void destroy(RTG &rtg);

	bool fluid_unpaused = false;
	//scalar(for fluid density)
	AddScalarSourcesPipeline add_scalar_sources_pipeline;
	DiffuseScalarPipeline diffuse_scalar_pipeline;
	AdvectDensityPipeline advect_density_pipeline;
	//vectors(for fluid velocity)
	AddVectorSourcesPipeline add_vector_sources_pipeline;
	DiffuseVectorPipeline diffuse_vector_pipeline;
	AdvectVectorPipeline advect_vector_pipeline;
	//projection pipelines
	DivergencePipeline divergence_pipeline;
	PressureSolvePipeline pressure_solve_pipeline;
	GradientSubtractPipeline gradient_subtract_pipeline;

	const uint32_t v_volume_side_length = 128;
	const uint32_t groupCounts[3] = {8,8,8};
	float cell_size_ws = 100.0f;

	VkSampler volume_sampler = VK_NULL_HANDLE;

	void make_ping_pong_descriptor_sets(RTG &rtg, ComputeContext comp_context, VkDescriptorSet *pressure_sets, VkDescriptorSetLayout const &layout, VkImageView *image_views);

	void add_sources_density(float dt, ComputeContext comp_context);
	void diffuse_density(float dt, ComputeContext comp_context);
	void advect_density(float dt, ComputeContext comp_context);

	void add_sources_velocity(float dt, ComputeContext comp_context, SceneSystem &scene_system);
	void diffuse_velocity(float dt, ComputeContext comp_context);
	void advect_velocity(float dt, ComputeContext comp_context);
	void project_velocity(ComputeContext comp_context);

	void ping_pong_barrier(VkCommandBuffer cmd, VkImage &img);

	//velocity
	uint32_t velocity_ind = 0;
	VkDescriptorSet velocity_volumes[2];
	Helpers::AllocatedImage3D velocity_3D_textures[2];
	VkImageView velocity_3D_views[2];
	//density
	uint32_t density_ind = 0;
	VkDescriptorSet density_volumes[2];
	Helpers::AllocatedImage3D density_3D_textures[2];
	VkImageView density_3D_views[2];
	//pressure
	uint32_t pressure_ind = 0;
	VkDescriptorSet pressure_volumes[2];
	Helpers::AllocatedImage3D pressure_3D_textures[2];
	VkImageView pressure_3D_views[2];
	//divergence
	VkDescriptorSet divergence_volume = VK_NULL_HANDLE;
	Helpers::AllocatedImage3D divergence_3D_texture;
	VkImageView divergence_3D_view;

	void update_fluid(float dt, ComputeContext comp_context, SceneSystem &scene_system);
};