#pragma once

#include "mat4.hpp"
#include "GeometryGen.hpp"
#include "PosColVertex.hpp"
#include "PosNorTexVertex.hpp"
#include "PosNorTanTexVertex.hpp"
#include "InputEvent.hpp" 
#include "S72.hpp"




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

	//Pipelines:
	struct BackgroundPipeline{
		//descriptor set layouts

		struct Push{
			float time;
		};

		VkPipelineLayout layout = VK_NULL_HANDLE;

		//vertex bindings
		
		VkPipeline handle = VK_NULL_HANDLE;

		void create(RTG &, VkRenderPass render_pass, uint32_t subpass);
		void destroy(RTG &);
	}background_pipeline;

	struct LinesPipeline{

		//descriptor set layouts
		VkDescriptorSetLayout set0_Camera = VK_NULL_HANDLE;
        //types for descriptors
        struct Camera{
            mat4 CLIP_FROM_WORLD;
        };
        static_assert(sizeof(Camera) == 16*4, "camera buffer structure is packed");

		//pipeline layout
		VkPipelineLayout layout = VK_NULL_HANDLE;
		using Vertex = PosColVertex;
		VkPipeline handle = VK_NULL_HANDLE;
		void create(RTG &,VkRenderPass render_pass, uint32_t subpass);
		void destroy(RTG &);
	}lines_pipeline;




	struct Transform{
		mat4 CLIP_FROM_LOCAL;
		mat4 WORLD_FROM_LOCAL;
		mat4 WORLD_FROM_LOCAL_NORMAL;
	};

	struct LambertianObjectsPipeline{
		//descriptor set layouts
		VkDescriptorSetLayout set0_World = VK_NULL_HANDLE;
		VkDescriptorSetLayout set1_Transforms = VK_NULL_HANDLE;
		VkDescriptorSetLayout set2_TEXTURE = VK_NULL_HANDLE;

        //types for descriptors
		struct World {
			struct { float x, y, z, padding_; } SKY_DIRECTION;
			struct { float r, g, b, padding_; } SKY_ENERGY;
			struct { float x, y, z, padding_; } SUN_DIRECTION;
			struct { float r, g, b, padding_; } SUN_ENERGY;
		};
		static_assert(sizeof(World) == 4*4 + 4*4 + 4*4 + 4*4, "World is the expected size.");	

        
        static_assert(sizeof(Transform) == 16*4 + 16*4 + 16*4 , "Transform structure is packed");

		//pipeline layout
		VkPipelineLayout layout = VK_NULL_HANDLE;
		using Vertex = PosNorTanTexVertex;
		VkPipeline handle = VK_NULL_HANDLE;
		void create(RTG &,VkRenderPass render_pass, uint32_t subpass);
		void destroy(RTG &);
	}lambertian_objects_pipeline;

	struct EnvMirrorObjectsPipeline{
		//descriptor set layouts
		VkDescriptorSetLayout set0_Eye= VK_NULL_HANDLE;
		VkDescriptorSetLayout set1_Transforms = VK_NULL_HANDLE;
		VkDescriptorSetLayout set2_TEXTURE = VK_NULL_HANDLE;

		struct Push{
			int32_t is_env = 1;
		};

        //types for descriptors
		struct Eye {
			vec3 EYE;
		};
		static_assert(sizeof(Eye) == 4*3, "Eye is the expected size.");	

        static_assert(sizeof(Transform) == 16*4 + 16*4 + 16*4 , "Transform structure is packed");

		//pipeline layout
		VkPipelineLayout layout = VK_NULL_HANDLE;
		using Vertex = PosNorTanTexVertex;
		VkPipeline handle = VK_NULL_HANDLE;
		void create(RTG &,VkRenderPass render_pass, uint32_t subpass);
		void destroy(RTG &);
	}env_mirror_objects_pipeline;

	struct PbrObjectsPipeline{
		//descriptor set layouts
		VkDescriptorSetLayout set0_Eye= VK_NULL_HANDLE;
		VkDescriptorSetLayout set1_Transforms = VK_NULL_HANDLE;
		VkDescriptorSetLayout set2_TEXTURE = VK_NULL_HANDLE;

        //types for descriptors
		struct Eye {
			vec3 EYE;
		};
		static_assert(sizeof(Eye) == 4*3, "Eye is the expected size.");	

        static_assert(sizeof(Transform) == 16*4 + 16*4 + 16*4 , "Transform structure is packed");

		//pipeline layout
		VkPipelineLayout layout = VK_NULL_HANDLE;
		using Vertex = PosNorTanTexVertex;
		VkPipeline handle = VK_NULL_HANDLE;
		void create(RTG &,VkRenderPass render_pass, uint32_t subpass);
		void destroy(RTG &);
	}pbr_objects_pipeline;

	

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

		//location for LinesPipeline::Camera data: (streamed to GPU per-frame)
		Helpers::AllocatedBuffer Eye_src; //host coherent; mapped
		Helpers::AllocatedBuffer Eye; //device-local
		VkDescriptorSet Eye_descriptors; //references Camera
		
		//location for ObjectsPipeline::World data: (streamed to GPU per-frame)
		Helpers::AllocatedBuffer World_src; //host coherent; mapped
		Helpers::AllocatedBuffer World; //device-local
		VkDescriptorSet World_descriptors; //references World

		//location for ObjectPipeline::Transform data: (streamed to GPU per-frame)
		Helpers::AllocatedBuffer Transforms_src;
		Helpers::AllocatedBuffer Transforms;
		VkDescriptorSet Transforms_descriptors;

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
	struct AABB{
		vec3 max = vec3{-INFINITY,-INFINITY,-INFINITY};
		vec3 min = vec3{INFINITY,INFINITY,INFINITY};
		void get_box_corners(mat4 WORLD_FROM_LOCAL, std::array<vec3, 8> &box_corners);
	};

	//Mesh contains stats of a unique mesh asset
	struct Mesh{
		ObjectVertices verts;
		AABB bbox;
	};

	ObjectVertices fruit_vertices;//these vertices are hard coded 
	std::map<std::string, Mesh> meshes;

	void load_scene();
	void update_scene(float dt);


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
	enum class TextureType{
		ALBEDO = 1,
		ROUGHNESS = 2,
		METALNESS = 3,
		NORMAL = 4,
	};
	void make_one_off_texture(TextureType t_type, std::variant<vec3, float> value);



	enum class MaterialType{
		LAMBERTIAN = 1,
		ENVMIRROR = 2,
		PBR = 3,
	};

	void make_descriptor_set(std::string mat_name, MaterialType mat_type, Texture_Indices tex_inds);
	

	//textures, texture_views, and texture_descriptors are all indexed the same
	std::vector<Helpers::AllocatedImage> textures;
	std::vector<VkImageView> texture_views;
	VkSampler texture_sampler = VK_NULL_HANDLE;
	VkDescriptorPool texture_descriptor_pool = VK_NULL_HANDLE;
	std::vector<VkDescriptorSet> texture_descriptor_sets;//allocated from texture descriptor pool


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

	uint32_t frame_number = 0;
	float frame_time = 0.0f;
	

	
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


	mat4 CLIP_FROM_WORLD;//The CLIP_FROM_WORLD matrix for the camera currently used 
	vec3 EYE = vec3(0,0,0); 

	//camera loaded in from s72 files
	std::unordered_map<std::string, BasicCamera> loaded_cameras;
	//the camera currently in use 
	std::unordered_map<std::string, BasicCamera>::iterator current_camera;

	std::vector<LinesPipeline::Vertex> lines_vertices;

	//world has two lights env and sun
	LambertianObjectsPipeline::World world;
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
