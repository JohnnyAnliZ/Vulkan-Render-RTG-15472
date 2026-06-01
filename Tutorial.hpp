#pragma once

#include "RenderWorkspace.hpp"
#include "SceneSystem.hpp"

#include "PosNorTexVertex.hpp"
#include "PosNorTanTexVertex.hpp"
#include "InputEvent.hpp"
#include "stb_image_write.h"
#include <iostream>

#include "BackgroundPipeline.hpp"
#include "LinesPipeline.hpp"
#include "TextureDebugPipeline.hpp"
#include "ObjectPipelines.hpp"
#include "Shadow2DPipeline.hpp"
#include "AddVectorSourcesPipeline.hpp"
#include "DiffuseVectorPipeline.hpp"
#include "AdvectVectorPipeline.hpp"
#include "ProjectionPipelines.hpp"
#include "DensityPipelines.hpp"
#include "RayMarchSmokeVolume.hpp"

#include "RTG.hpp"
#include <corecrt_math_defines.h>

struct Tutorial : RTG::Application {

	Tutorial(RTG &);
	Tutorial(Tutorial const &) = delete;
	~Tutorial();

	//kept for use in destructor:
	RTG &rtg;

	//--------------------------------------------------------------------
	//Resources that last the lifetime of the application:

	VkFormat depth_format{};
	VkRenderPass render_pass = VK_NULL_HANDLE;
	VkRenderPass shadow_pass = VK_NULL_HANDLE;

	//Pipelines:
	BackgroundPipeline background_pipeline;
	TextureDebugPipeline texture_debug_pipeline;
	RayMarchSmokeVolumePipeline ray_march_smoke_volume_pipeline;
	LinesPipeline lines_pipeline;
	LambertianObjectsPipeline lambertian_objects_pipeline;
	EnvMirrorObjectsPipeline env_mirror_objects_pipeline;
	PbrObjectsPipeline pbr_objects_pipeline;
	Shadow2DPipeline shadow_2D_pipeline;

	void init_tutorial();
	void init_shadow_mapping();
	VkCommandPool command_pool = VK_NULL_HANDLE;
	VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
	std::vector<Workspace> workspaces;

	//-------------------------------------------------------------------
	// scene72 is owned here so both systems can reference it by address
	S72 scene72;

	// material_system uses scene72 for texture loading;
	MaterialSystem material_system;


	// scene_system uses scene72 for graph traversal and material_system for descriptor lookup
	// camera, debug lines, and view state live in scene_system
	SceneSystem scene_system;

	//------Fluid Simulation stuff-----
	bool fluid_unpaused = false;
	VkDescriptorPool storage_descriptor_pool = VK_NULL_HANDLE;
	VkCommandPool compute_command_pool = VK_NULL_HANDLE;
	VkCommandBuffer compute_cmd_buf = VK_NULL_HANDLE;
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

	void make_ping_pong_descriptor_sets(VkDescriptorSet *pressure_sets, VkDescriptorSetLayout const &layout, VkImageView *image_views);

	void add_sources_density(float dt);
	void diffuse_density(float dt);
	void advect_density(float dt);

	void add_sources_velocity(float dt);
	void diffuse_velocity(float dt);
	void advect_velocity(float dt);
	void project_velocity();

	void ping_pong_barrier(VkCommandBuffer cmd, VkImage &img);

	//velocity
	uint32_t velocity_ind = 0;
	VkDescriptorSet velocity_volumes[2];
	VkDescriptorSet velocity_tex = VK_NULL_HANDLE;
	Helpers::AllocatedImage3D velocity_3D_textures[2];
	VkImageView velocity_3D_views[2];
	//density
	uint32_t density_ind = 0;
	VkDescriptorSet density_volumes[2];
	VkDescriptorSet density_tex = VK_NULL_HANDLE;
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
	
	void init_compute();
	void init_fluid();
	void update_fluid(float dt);

	//--------------------------------------------------------------------
	//Resources that change when the swapchain is resized:

	virtual void on_swapchain(RTG &, RTG::SwapchainEvent const &) override;

	Helpers::AllocatedImage swapchain_depth_image;
	VkImageView swapchain_depth_image_view = VK_NULL_HANDLE;
	std::vector<VkFramebuffer> swapchain_framebuffers;
	void destroy_framebuffers();

	//--------------------------------------------------------------------
	//Resources that change when time passes or the user interacts:

	virtual void update(float dt) override;
	virtual void on_input(InputEvent const &) override;

	std::function<void(InputEvent const &)> action;

	float time = 0.0f;

	//animation controls
	uint32_t frame_number = 0;
	float frame_time = 0.0f;
	bool animating = true;


	//--------------------------------------------------------------------
	virtual void render(RTG &, RTG::RenderParams const &) override;
};
