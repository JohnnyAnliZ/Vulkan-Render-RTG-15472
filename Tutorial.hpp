#pragma once

#include "mat4.hpp"
#include "GeometryGen.hpp"

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
	LinesPipeline lines_pipeline;

	LambertianObjectsPipeline lambertian_objects_pipeline;
	EnvMirrorObjectsPipeline env_mirror_objects_pipeline;
	PbrObjectsPipeline pbr_objects_pipeline;


	struct Transform{
		mat4 CLIP_FROM_LOCAL;
		mat4 WORLD_FROM_LOCAL;
		mat4 WORLD_FROM_LOCAL_NORMAL;
	};
	static_assert(sizeof(Transform) == 16*4 + 16*4 + 16*4 , "Transform structure is packed");

	struct Light {
		vec4 color;
		vec4 position;
		vec4 direction;
		int type;//0 - sun ; 1 - sphere ; 2 - spot
		float limit;
		//sphere 
		float radius;
		float power;
		//sun
		float angle;
		float strength;
		//spot light only
		float fov;
		float blend;
		vec4 shadow_atlases[6];// (offset.x, offset.y, scale.x, scale.y)
		mat4 CLIP_FROM_WORLD[6];

		void compute_clip_from_world_spot(){
			assert(type == 2);
			vec3 up;
			if (abs(dot(direction.xyz(), vec3(0,1,0))) > 0.99f) {
				// too parallel to Y axis → switch
				up = vec3(0,0,1);
			} else {
				up = vec3(0,1,0);
			}
			up = vec3(0,1,0);
			
			CLIP_FROM_WORLD[0] = perspective(fov, 1.0f, 1.0f, limit) * look_at_free(position.xyz(), vec3(0.0f)+direction.xyz(), up);
		}
		
		void compute_clip_from_world_sphere(){
			assert(type == 1);
			mat4 views[6];
			views[0] = look_at_free(position.xyz(), vec3(1,0,0), vec3(0,-1,0));
			views[1] = look_at_free(position.xyz(), vec3(-1,0,0), vec3(0,-1,0));
			views[2] = look_at_free(position.xyz(), vec3(0,1,0), vec3(0,0,1));
			views[3] = look_at_free(position.xyz(), vec3(0,-1,0), vec3(0,0,-1));
			views[4] = look_at_free(position.xyz(), vec3(0,0,1), vec3(0,-1,0));
			views[5] = look_at_free(position.xyz(), vec3(0,0,-1), vec3(0,-1,0));
			for(uint32_t i = 0; i < 6; i++)
			CLIP_FROM_WORLD[i] = perspective((float)M_PI / 2.0f, 1.0f, 0.1f, limit) * views[i];
		}

		std::array<vec3, 8> get_corners() const;
		std::array<vec3, 8> get_frustum_corners() const;
	};
	static_assert(sizeof(Light) == 3*4*4 + 8*4 + 4 * 4 * 6 + 6 * 16 * 4, "Light structure is packed");    
	

	void init_tutorial();
	//pools from which per-workspace things are allocated:
	VkCommandPool command_pool = VK_NULL_HANDLE;
	VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;

	//workspaces hold per-render resources:
	struct Workspace {
		VkCommandBuffer command_buffer = VK_NULL_HANDLE; //from the command pool above; reset at the start of every render.

		//location for lines data
		Helpers::AllocatedBuffer lines_vertices_src;//host coherent, mapped
		Helpers::AllocatedBuffer lines_vertices;//device-local

		//location for LinesPipeline::Camera data: (streamed to GPU per-frame)
		Helpers::AllocatedBuffer Camera_src; //host coherent; mapped
		Helpers::AllocatedBuffer Camera; //device-local
		VkDescriptorSet Camera_descriptors; //references Camera

		//location for Eye data: (streamed to GPU per-frame)
		Helpers::AllocatedBuffer Eye_src; //host coherent; mapped
		Helpers::AllocatedBuffer Eye; //device-local
		VkDescriptorSet Eye_descriptors; //references Camera
		
		//location for Lights data: (streamed to GPU per-frame)
		Helpers::AllocatedBuffer Lights_src; //host coherent; mapped
		Helpers::AllocatedBuffer Lights; //device-local
		VkDescriptorSet Lights_descriptors; //references World

		//location for Transform data: (streamed to GPU per-frame)
		Helpers::AllocatedBuffer Transforms_src;
		Helpers::AllocatedBuffer Transforms;
		VkDescriptorSet Transforms_descriptors;

		//location for Shadow_Atlas data
		Helpers::AllocatedImage Shadow_Atlas;
		VkImageView Shadow_Atlas_view;
		VkFramebuffer Shadow_Atlas_FB = VK_NULL_HANDLE;
		VkDescriptorSet Shadow_Atlas_descriptors;
		Helpers::AllocatedBuffer debug_buffer;
	};
	std::vector< Workspace > workspaces;

	//-------------------------------------------------------------------
	//loading s72
	S72 scene72 = S72();
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

	struct AABB{
		vec3 max = vec3{-INFINITY,-INFINITY,-INFINITY};
		vec3 min = vec3{INFINITY,INFINITY,INFINITY};
		void get_box_corners(mat4 WORLD_FROM_LOCAL, std::array<vec3, 8> &box_corners);
	};
	struct Mesh{
		ObjectVertices verts;
		AABB bbox;
	};
	std::map<std::string, Mesh> meshes;

	void load_scene();
	void update_scene(float dt);

	//------Fluid Simulation stuff-----
	VkCommandPool compute_command_pool = VK_NULL_HANDLE;
	VkCommandBuffer compute_cmd_buf = VK_NULL_HANDLE;
	AddVectorSourcesPipeline add_vector_sources_pipeline;

	const uint32_t v_volume_side_length = 64; //side length of the velocity volume
	const uint32_t groupCounts[3] = {8,8,8};
	uint32_t velocity_ind = 0; //this tracks which velocity volume descriptor set to use. When velocity_ind == 0, binding 0 is velocity_3D[0 , 1] 
	VkDescriptorSet velocity_volume[2]; //these two buffers each contain two bindings of the same image views in opposite order
	VkDescriptorSet velocity_tex = VK_NULL_HANDLE; //this one is just for sampling;
	VkDescriptorSet density_volume = VK_NULL_HANDLE;
	Helpers::AllocatedImage3D velocity_3D_texture[2]; //two for ping-ponging
	VkImageView velocity_3D_views[2];

	void init_compute_pipeline();
	void init_fluid();
	void update_fluid(float dt);

	//------Texture stuff------
	struct Texture_Indices{
		int normal_index = -1;
		int displacement_index = -1;
		int albedo_index = -1;
		int roughness_index = -1;
		int metalness_index = -1;
		int environment_index = -1;	
		int brdf_lut_index = -1;
		int diffuse_irradiance_index = -1;
	};

	//default texture indices 
	int normal_default = -1;
	int disp_default = -1;

	//name for luts
	std::string brdf_lut_name;
	std::string lambertian_irradiance_lut_name;
	std::string environment_name;

	//maps name of the loaded texture to indices into the texture_descriptor_sets array
	std::unordered_map<std::string, uint32_t> material_texture_descriptor_set_table;

	//memorization so that materials that use the same texture again don't remake the same image
	std::unordered_map<std::string, uint32_t> textures_name_to_index;

	void load_textures();
	//helpers 
	void make_flat_image(S72::Texture const &texture);
	void make_cube_image(S72::Texture const &texture);
	void make_image_view(VkImageView &image_view, Helpers::AllocatedImage const & image);
	void make_image_view_3D(VkImageView &image_view, Helpers::AllocatedImage3D const & image);

	enum class TextureType{
		ALBEDO = 1,
		ROUGHNESS = 2,
		METALNESS = 3,
		NORMAL = 4,
		ENV = 5,//cubemap with one mip level
		ENV_MIP = 6,//cubemap with multiple mip levels
		BRDF_LUT = 7,
	};
	void make_one_off_texture(TextureType t_type, std::variant<vec3, float> value);


	enum class MaterialType{
		LAMBERTIAN = 1,
		ENVMIRROR = 2,
		PBR = 3,
	};
	void make_descriptor_set(std::string mat_name, MaterialType mat_type, Texture_Indices tex_inds);
	void make_velocity_descriptor_sets(VkDescriptorSet *velocity_sets, VkDescriptorSetLayout const &layout);

	

	//textures, texture_views, and texture_descriptors are all indexed the same
	std::vector<Helpers::AllocatedImage> textures;
	std::vector<VkImageView> texture_views;
	VkSampler texture_sampler = VK_NULL_HANDLE;
	VkSampler depth_texture_sampler = VK_NULL_HANDLE;
	VkDescriptorPool texture_descriptor_pool = VK_NULL_HANDLE;
	std::vector<VkDescriptorSet> texture_descriptor_sets;//allocated from texture descriptor pool


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
	//for selecting between cameras:
	enum class CameraMode {
		Scene = 0,
		Free = 1,
		Debug = 2,
	} camera_mode = CameraMode::Scene;

	CameraMode prev_camera_mode = CameraMode::Debug;

	//generic camera type with the bare minimum to construct a CLIP from World
	struct BasicCamera{
		vec3 eye;//translation of the world transform
		vec3 dir;//local -z axis in world space
		vec3 up;//local +y axis in world space
		float aspect;
		float vfov;
		float near;
		float far = std::numeric_limits< float >::infinity(); //optional, if not specified will be set to infinity
		
		mat4 clip_from_world(){
			return perspective(vfov,aspect,near,far) * look_at_free(eye,dir,up);
		}
		// Returns the 8 corners of the view frustum in world space
		// Order: near plane (bottom-left, bottom-right, top-right, top-left),
		//        far plane (bottom-left, bottom-right, top-right, top-left)
		std::array<vec3, 8> get_frustum_corners()const;
	};

	//used when camera_mode == CameraMode::Free:
	struct OrbitCamera{
		vec3 target = vec3();
		float radius = 5.0f;//distance from camera to target
		float azimuth = 0.0f;//counterclockwise angle around z axis between x axis and camera direction (radians)
		float elevation = 0.25 * float(M_PI);//angle up from xy plane to camera direction (radians)
		float fov = 60.0f / 180.0f * float(M_PI); //vertical field of view (radians)
		float near = 0.1f; //near clipping plane
		float far = 1000.0f; //far clipping plane

		mat4 clip_from_world(float aspect){
			return perspective(
			60.0f * float(M_PI) / 180.0f, //vfov
			aspect, //aspect
			near, //near
			far //far
			) * orbit(target, azimuth, elevation, radius);
		}

		vec3 get_eye()const;
		std::array<vec3, 8> get_frustum_corners(float aspect)const;
		
	} free_camera;


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
