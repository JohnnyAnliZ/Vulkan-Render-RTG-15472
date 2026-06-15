#pragma once

#include "RenderWorkspace.hpp"
#include "SceneSystem.hpp"
#include "ComputeSystem.hpp"
#include "FluidSystem.hpp"
#include "TextureDebugSystem.hpp"

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
#include "RayMarchSmokeVolume.hpp"
#include "ParticleSystem.hpp"

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

	//compute system that handles GPU compute shader work 
	ComputeSystem compute_system;

	//fluid system that uses computes system to do voxel fluid simulation
	FluidSystem fluid_system;

	//TextureDebugSystem
	TextureDebugSystem texture_debug_system;

	ParticleSystem particle_system;
	VkDescriptorPool particle_descriptor_pool = VK_NULL_HANDLE;

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

	std::function<void(InputEvent const &)> action;\

	//animation
	float time = 0.0f;
	uint32_t frame_number = 0;
	float frame_time = 0.0f;
	bool animating = true;

	//--------------------------------------------------------------------
	virtual void render(RTG &, RTG::RenderParams const &) override;
};
