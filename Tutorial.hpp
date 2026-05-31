#pragma once

#include "mat4.hpp"
#include "GeometryGen.hpp"
#include "Light.hpp"
#include "Camera.hpp"
#include "Transform.hpp"
#include "RenderWorkspace.hpp"
#include "MaterialSystem.hpp"

#include "PosNorTexVertex.hpp"
#include "PosNorTanTexVertex.hpp"
#include "InputEvent.hpp"
#include "S72.hpp"
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
	Tutorial(Tutorial const &) = delete; //you shouldn't be copying this object
	~Tutorial();

	//kept for use in destructor:
	RTG &rtg;

	//--------------------------------------------------------------------
	//Resources that last the lifetime of the application:

	//chosen format for depth buffer:
	VkFormat depth_format{};
	//Render passes describe how pipelines write to images:
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

	void init_tutorial();
	//pools from which per-workspace things are allocated:
	VkCommandPool command_pool = VK_NULL_HANDLE;
	VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
	std::vector<Workspace> workspaces;

	//-------------------------------------------------------------------

	//loading s72
	S72 scene72 = S72();
	MaterialSystem material_system;

	struct Item {
		S72::Node* node;
		mat4 parentWorld;
	};

	//static scene resources:
	Helpers::AllocatedBuffer object_vertices;
	struct ObjectVertices{//vertex indecies for unique meshes
		uint32_t first = 0;
		uint32_t count = 0;
	};
	ObjectVertices fruit_vertices;//these vertices are hard coded

	//Mesh contains stats of a unique mesh asset
	struct Mesh{
		ObjectVertices verts;
		AABB bbox;
	};
	std::map<std::string, Mesh> meshes;

	void load_scene();
	void update_scene(float dt);

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

	const uint32_t v_volume_side_length = 128; //side length of the velocity volume
	const uint32_t groupCounts[3] = {8,8,8};
	float cell_size_ws = 100.0f;//size of a cell in world space, used for calculating sampling positions in the shaders

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
	uint32_t velocity_ind = 0; //this tracks which velocity volume descriptor set to use. When velocity_ind == 0, binding 0 is velocity_3D[0 , 1]
	VkDescriptorSet velocity_volumes[2]; //these two buffers each contain two bindings of the same image views in opposite order
	VkDescriptorSet velocity_tex = VK_NULL_HANDLE; //this one is just for sampling;
	Helpers::AllocatedImage3D velocity_3D_textures[2]; //two for ping-ponging
	VkImageView velocity_3D_views[2];
	//density
	uint32_t density_ind = 0;
	VkDescriptorSet density_volumes[2];
	VkDescriptorSet density_tex = VK_NULL_HANDLE;//for sampling
	Helpers::AllocatedImage3D density_3D_textures[2];
	VkImageView density_3D_views[2];
	//pressure
	uint32_t pressure_ind = 0;
	VkDescriptorSet pressure_volumes[2];
	Helpers::AllocatedImage3D pressure_3D_textures[2];
	VkImageView pressure_3D_views[2];
	//divergence (only need one since it's only written in the divergence pass and only read in the pressure solve pass, so no ping-ponging needed)
	VkDescriptorSet divergence_volume = VK_NULL_HANDLE;
	Helpers::AllocatedImage3D divergence_3D_texture;
	VkImageView divergence_3D_view;

	void init_compute();
	void init_fluid();
	void update_fluid(float dt);

	//------shadow stuff-------
	bool shadows_on = true;
	bool shadow_dump = false;
	Shadow2DPipeline shadow_2D_pipeline;
	void init_shadow_mapping();
	//draw all objects with shadow2DPipeline from light's perspective into shadow_atlas
	void draw_all_objects(VkCommandBuffer const &cmd, mat4 const &LIGHTS_CLIP_FROM_WORLD, vec4 const &_shadow_atlas);

	//--------------------------------------------------------------------
	//Resources that change when the swapchain is resized:

	virtual void on_swapchain(RTG &, RTG::SwapchainEvent const &) override;

	Helpers::AllocatedImage swapchain_depth_image;
	VkImageView swapchain_depth_image_view = VK_NULL_HANDLE;
	std::vector< VkFramebuffer > swapchain_framebuffers;
	//used from on_swapchain and the destructor: (framebuffers are created in on_swapchain)
	void destroy_framebuffers();

	//--------------------------------------------------------------------
	//Resources that change when time passes or the user interacts:

	virtual void update(float dt) override;
	virtual void on_input(InputEvent const &) override;

	//modal action, intercepts inputs:
	std::function< void(InputEvent const &) > action;

	float time = 0.0f;
	float time_elapsed = 0.0f;

	//animation controls
	uint32_t frame_number = 0;
	float frame_time = 0.0f;

	bool animating = true;



	//-----camera stuff-----
	CameraMode camera_mode     = CameraMode::Scene;
	CameraMode prev_camera_mode = CameraMode::Debug;

	OrbitCamera free_camera;


	//----Culling----

	//takes in corners of both boxes in world space, returns true if the box could be culled
	bool do_cull(std::array<vec3, 8> const &frustrum_corners,
				std::array<vec3, 8> const &box_corners
	);


	//----Debug Camera----
	OrbitCamera debug_camera;//used when camera_mode == CameraMode::Debug:
	void add_debug_lines_frustrum();
	void add_debug_lines_bbox(AABB &bbox, mat4 WORLD_FROM_LOCAL);
	void add_cuboid_from_corners(std::array<vec3, 8> const &box_corners, uint8_t r, uint8_t g, uint8_t b);

	mat4 CLIP_FROM_WORLD;//The CLIP_FROM_WORLD matrix for the camera currently used
	vec3 EYE = vec3(0,0,0);
	vec3 CAM_DIR = vec3(0,0,-1);

	bool wind_motor_active = false;

	//camera loaded in from s72 files
	std::unordered_map<std::string, BasicCamera> loaded_cameras;
	//the camera currently in use
	std::unordered_map<std::string, BasicCamera>::iterator current_camera;

	std::vector<LinesPipeline::Vertex>lines_vertices;

	//world has two lights env and sun
	std::vector<Light> lights;
	std::vector<uint32_t> light_shadow_map_sizes;//this should be aligned with lights
	uint64_t total_shadow_map_size = 0;
	uint32_t atlas_size = 0;//side length of the big square shadow atlas

	struct point
	{
		uint32_t x, y;
	};

	struct box
	{
		point topleft;
		point bottomright;
	};
	void allocate_texture_atlas(
		point const & atlas_size,
		std::vector<uint32_t> const & texture_sizes
	);
	bool default_world_lights = true;//if this is true that means there hasn't been any lights loaded


	//----objects of different materials
	struct LambertianObjectInstance{
		ObjectVertices vertices;
		Transform transform;
		uint32_t texture = 0;
	};
	struct EnvMirrorObjectInstance{
		ObjectVertices vertices;
		Transform transform;
		uint32_t texture = 0;
		int is_env = 1;//default to env
	};
	struct PbrObjectInstance{
		ObjectVertices vertices;
		Transform transform;
		uint32_t texture = 0;
	};

	std::vector<LambertianObjectInstance> lambertian_object_instances;
	std::vector<EnvMirrorObjectInstance> env_mirror_object_instances;
	std::vector<PbrObjectInstance> pbr_object_instances;


	//stuff for tree generation
	vec3 emplace_random_line(vec3 start, uint32_t iter);
	bool growing = true;
	std::vector<vec3> starts;
	uint32_t iters = 0;

	//--------------------------------------------------------------------
	//Rendering function, uses all the resources above to queue work to draw a frame:

	virtual void render(RTG &, RTG::RenderParams const &) override;
};
