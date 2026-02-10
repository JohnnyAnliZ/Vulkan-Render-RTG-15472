#include "Tutorial.hpp"

#include "VK.hpp"

#include <GLFW/glfw3.h>

#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <corecrt_math_defines.h>

#include <random>
#include <chrono>


unsigned long long seed = std::chrono::system_clock::now().time_since_epoch().count();
std::mt19937 engine((unsigned int)seed);
std::uniform_real_distribution<float> dist(-1.0f, 1.0f);


Tutorial::Tutorial(RTG &rtg_) : rtg(rtg_) {

	//load the scene file
	try {
		scene72 = S72::load(rtg.configuration.scene_file);
	} catch (std::exception &e) {
		std::cerr << "Failed to load s72-format scene from '" << rtg.configuration.scene_file << "':\n" << e.what() << std::endl;
		return ;
	}

	{//load nodes into a new nodes map, trimming the nodes that don't have or lead to anything(meshes, lights, cameras)
		//This is not necessary, not gonna do this now
	}

	
	{//preallocate some space on the lines buffer
		lines_vertices.clear();
		constexpr size_t lines_vert_count = 4096;
		lines_vertices.reserve(lines_vert_count);
		starts.reserve(512);
	}

	//select a depth format:
	//  (at least one of these two must be supported, according to the spec; but neither are required)
	depth_format = rtg.helpers.find_image_format(
		{ VK_FORMAT_D32_SFLOAT, VK_FORMAT_X8_D24_UNORM_PACK32 },
		VK_IMAGE_TILING_OPTIMAL,
		VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
	);

	{ //create render pass
		//specify attachments
		std::array< VkAttachmentDescription, 2 > attachments{
			VkAttachmentDescription{ //0 - color attachment:
				.format = rtg.surface_format.format,
				.samples = VK_SAMPLE_COUNT_1_BIT,
				.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
				.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
				.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
				.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
				.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
				.finalLayout = rtg.present_layout,
			},
			VkAttachmentDescription{ //1 - depth attachment:
				.format = depth_format,
				.samples = VK_SAMPLE_COUNT_1_BIT,
				.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
				.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
				.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
				.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
				.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
				.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
			},
		};
		//subpass
		VkAttachmentReference color_attachment_ref{
			.attachment = 0,
			.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
		};

		VkAttachmentReference depth_attachment_ref{
			.attachment = 1,
			.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		};

		VkSubpassDescription subpass{
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.inputAttachmentCount = 0,
			.pInputAttachments = nullptr,
			.colorAttachmentCount = 1,
			.pColorAttachments = &color_attachment_ref,
			.pDepthStencilAttachment = &depth_attachment_ref,
		};
		//dependencies
		std::array< VkSubpassDependency, 2 > dependencies {
			VkSubpassDependency{
				.srcSubpass = VK_SUBPASS_EXTERNAL,
				.dstSubpass = 0,
				.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
				.srcAccessMask = 0,
				.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
			},
			VkSubpassDependency{
				.srcSubpass = VK_SUBPASS_EXTERNAL,
				.dstSubpass = 0,
				.srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
				.dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
				.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
				.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
			}
		};
		VkRenderPassCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
			.attachmentCount = uint32_t(attachments.size()),
			.pAttachments = attachments.data(),
			.subpassCount = 1,
			.pSubpasses = &subpass,
			.dependencyCount = uint32_t(dependencies.size()),
			.pDependencies = dependencies.data(),
		};

		VK( vkCreateRenderPass(rtg.device, &create_info, nullptr, &render_pass) );
	}

	{ //create command pool
		VkCommandPoolCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = rtg.graphics_queue_family.value(),
		};
		VK( vkCreateCommandPool(rtg.device, &create_info, nullptr, &command_pool) );
	}

	background_pipeline.create(rtg, render_pass, 0);
	lines_pipeline.create(rtg, render_pass, 0);
	objects_pipeline.create(rtg, render_pass,0);
	{//create descriptor pool
		uint32_t per_workspace = uint32_t(rtg.workspaces.size());

		std::array<VkDescriptorPoolSize, 2> pool_sizes{
			VkDescriptorPoolSize{
				//we only need uniform buffer descriptors for the moment:
				.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.descriptorCount = 2 * per_workspace,//one descriptor per set, one set per workspace
			},
			VkDescriptorPoolSize{
				.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.descriptorCount = 1 * per_workspace,//one descriptor per set, one set per workspace
			},
		};
		VkDescriptorPoolCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.flags = 0,
			.maxSets = 3 * per_workspace,
			.poolSizeCount = uint32_t(pool_sizes.size()),
			.pPoolSizes = pool_sizes.data(),
		};
		VK(vkCreateDescriptorPool(rtg.device, &create_info, nullptr, &descriptor_pool));
	}

	workspaces.resize(rtg.workspaces.size());
	for (Workspace &workspace : workspaces) {
		{//allocate command buffer
			VkCommandBufferAllocateInfo alloc_info{
				.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,				
				.commandPool = command_pool,
				.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
				.commandBufferCount = 1,
			};
			VK(vkAllocateCommandBuffers(rtg.device, &alloc_info, &workspace.command_buffer));
		}
		//buffers for camera descriptors
		workspace.Camera_src = rtg.helpers.create_buffer(
			sizeof(LinesPipeline::Camera),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			Helpers::Mapped
		);
		workspace.Camera = rtg.helpers.create_buffer(
			sizeof(LinesPipeline::Camera),
			VK_BUFFER_USAGE_TRANSFER_DST_BIT|VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			Helpers::Unmapped
		);

		{//allocate descriptor set for Camera descriptor
			VkDescriptorSetAllocateInfo alloc_info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &lines_pipeline.set0_Camera,
			};
			VK(vkAllocateDescriptorSets(rtg.device, &alloc_info, &workspace.Camera_descriptors));
		}

	//buffers for World descriptors
		workspace.World_src = rtg.helpers.create_buffer(
			sizeof(ObjectsPipeline::World),
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			Helpers::Mapped
		);
		workspace.World = rtg.helpers.create_buffer(
			sizeof(ObjectsPipeline::World),
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			Helpers::Unmapped
		);

		{ //allocate descriptor set for World descriptor
			VkDescriptorSetAllocateInfo alloc_info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &objects_pipeline.set0_World,
			};
			VK( vkAllocateDescriptorSets(rtg.device, &alloc_info, &workspace.World_descriptors) );
			//NOTE: will actually fill in this descriptor set just a bit lower
		}

		{//allocate descriptor set for Transforms descriptor
			VkDescriptorSetAllocateInfo alloc_info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &objects_pipeline.set1_Transforms,
			};
			VK(vkAllocateDescriptorSets(rtg.device, &alloc_info, &workspace.Transforms_descriptors));
			//NOTE: will fill in this descriptor set in render when buffers are [re-]allocated
		}
		
		{//point descriptor to Camera buffer
			VkDescriptorBufferInfo Camera_info{
				.buffer = workspace.Camera.handle,
				.offset = 0,
				.range = workspace.Camera.size,
			};

			VkDescriptorBufferInfo World_info{
				.buffer = workspace.World.handle,
				.offset = 0,
				.range = workspace.World.size,
			};
			
			std::array<VkWriteDescriptorSet, 2> writes{
				VkWriteDescriptorSet{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = workspace.Camera_descriptors,
					.dstBinding = 0,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.pBufferInfo = &Camera_info,
				},
				VkWriteDescriptorSet{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = workspace.World_descriptors,
					.dstBinding = 0,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.pBufferInfo = &World_info,
				},
			};

			vkUpdateDescriptorSets(
				rtg.device, //device
				uint32_t(writes.size()), //descriptorWriteCount
				writes.data(), //pDescriptorWrites
				0, //descriptorCopyCount
				nullptr //pDescriptorCopies
			);
		}

	}
	
	{//create objects vertices
		std::vector<PosNorTanTexVertex> vertices;

		{//load vertices from s72 file, so that all the vertex data(attributes) are in one big pool
			mesh_vertices.clear();
			size_t base = 0;//base offset for writing in each mesh's vertices into std::vector<PosNorTanTexVertex> vertices
			uint32_t count = 0;//the number of vertices for each mesh
			for(auto const &mesh : scene72.meshes ){
				std::cout<<"loading mesh: "<< mesh.first<<std::endl;
				//assuming there aren't duplicate meshes, no indices property, use PosNorTanTex attributes
				assert(mesh_vertices.find(mesh.first) == mesh_vertices.end());
				//get the indices
				count = mesh.second.count;
				mesh_vertices[mesh.first].first = uint32_t(vertices.size());
				mesh_vertices[mesh.first].count = count;
				//copy the vertices from file into vertices
				base = vertices.size();
				vertices.resize(base + count);//resize for copying
				S72::Mesh::Attribute const &att_pos = mesh.second.attributes.at("POSITION");//12bytes 
				S72::Mesh::Attribute const &att_nor = mesh.second.attributes.at("NORMAL");//12bytes
				S72::Mesh::Attribute const &att_tan = mesh.second.attributes.at("TANGENT");//16bytes
				S72::Mesh::Attribute const &att_tex= mesh.second.attributes.at("TEXCOORD");//8bytes
				for(uint32_t i = 0; i<count; ++i){
					std::memcpy(&vertices[base + i].Position, att_pos.src.data.data() + att_pos.offset + att_pos.stride * i, sizeof(vec3));
					std::memcpy(&vertices[base + i].Normal, att_nor.src.data.data() + att_nor.offset + att_nor.stride * i, sizeof(vec3));
					std::memcpy(&vertices[base + i].Tangent, att_tan.src.data.data() + att_tan.offset + att_tan.stride * i, sizeof(vec4));
					std::memcpy(&vertices[base + i].TexCoord, att_tex.src.data.data() + att_tex.offset + att_tex.stride * i, sizeof(float) * 2);
				}
			}
		}


		{//a spiky fruit (durian?)
			fruit_vertices.first = uint32_t(vertices.size());

			//this code to generate an ico sphere is taken from:
			//https://schneide.blog/2016/07/15/generating-an-icosphere-in-c/#:~:text=for%20(%20int%20i=0;,marching%20cubes%20or%20marching%20tetrahedrons.
			IndexedMesh ball_mesh = make_spiky_icosphere(1);
			//now, use the indexed mesh to emplace vertices
			auto emplace_triangle = [&](Triangle tri){
				for(int32_t i = 2; i>-1; i--){
					vertices.emplace_back(PosNorTanTexVertex{
						.Position{
							.x = ball_mesh.first[tri.vertex[i]].x,
							.y = ball_mesh.first[tri.vertex[i]].y,
							.z = ball_mesh.first[tri.vertex[i]].z,
						},
						.Normal{
							.x = ball_mesh.first[tri.vertex[i]].x,
							.y = ball_mesh.first[tri.vertex[i]].y,
							.z = ball_mesh.first[tri.vertex[i]].z,
						},
						.Tangent{
							.x = - ball_mesh.first[tri.vertex[i]].y,
							.y = ball_mesh.first[tri.vertex[i]].x,
							.z = 0.0f,
							.w = 1.0f,
						},
						.TexCoord{
							.s = ball_mesh.first[tri.vertex[i]].x,
							.t = ball_mesh.first[tri.vertex[i]].y,
						}
					});
				}
				
			};
			for(Triangle const & tri: ball_mesh.second){
				emplace_triangle(tri);
			}
			fruit_vertices.count = uint32_t(vertices.size()) - fruit_vertices.first;
		}
		

		size_t bytes = sizeof(vertices[0]) * vertices.size();
		object_vertices = rtg.helpers.create_buffer(bytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, Helpers::Unmapped);
		//copy data to buffer
		rtg.helpers.transfer_to_buffer(vertices.data(), bytes, object_vertices);
	}

	{ // make some textures
		textures.reserve(2);
		{ //texture 0 will be a dark grey / light grey checkerboard with a red square at the origin.
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
	}

	{ //make image views for the textures
		texture_views.reserve(textures.size());
		for (Helpers::AllocatedImage const &image : textures) {
			VkImageViewCreateInfo create_info{
				.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
				.flags = 0,
				.image = image.handle,
				.viewType = VK_IMAGE_VIEW_TYPE_2D,
				.format = image.format,
				// .components sets swizzling and is fine when zero-initialized
				.subresourceRange{
					.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
					.baseMipLevel = 0,
					.levelCount = 1,
					.baseArrayLayer = 0,
					.layerCount = 1,
				},
			};

			VkImageView image_view = VK_NULL_HANDLE;
			VK( vkCreateImageView(rtg.device, &create_info, nullptr, &image_view) );

			texture_views.emplace_back(image_view);
		}
		assert(texture_views.size() == textures.size());
	}

	{ //make a sampler for the textures
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

	{ //TODO: allocate and write the texture descriptor sets
		//allocate the descriptors (using the same alloc_info):
		VkDescriptorSetAllocateInfo alloc_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
			.descriptorPool = texture_descriptor_pool,
			.descriptorSetCount = 1,
			.pSetLayouts = &objects_pipeline.set2_TEXTURE,
		};
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
				.dstSet = texture_descriptors[i],
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

Tutorial::~Tutorial() {
	//just in case rendering is still in flight, don't destroy resources:
	//(not using VK macro to avoid throw-ing in destructor)
	if (VkResult result = vkDeviceWaitIdle(rtg.device); result != VK_SUCCESS) {
		std::cerr << "Failed to vkDeviceWaitIdle in Tutorial::~Tutorial [" << string_VkResult(result) << "]; continuing anyway." << std::endl;
	}

	//freeing texture
	if (texture_descriptor_pool) {
		vkDestroyDescriptorPool(rtg.device, texture_descriptor_pool, nullptr);
		texture_descriptor_pool = nullptr;
		//this also frees the descriptor sets allocated from the pool:
		texture_descriptors.clear();
	}

	if (texture_sampler) {
		vkDestroySampler(rtg.device, texture_sampler, nullptr);
		texture_sampler = VK_NULL_HANDLE;
	}

	for (VkImageView &view : texture_views) {
		vkDestroyImageView(rtg.device, view, nullptr);
		view = VK_NULL_HANDLE;
	}
	texture_views.clear();

	for (auto &texture : textures) {
		rtg.helpers.destroy_image(std::move(texture));
	}
	textures.clear();

	//freeing static buffers
	rtg.helpers.destroy_buffer(std::move(object_vertices));

	if (swapchain_depth_image.handle != VK_NULL_HANDLE) {
		destroy_framebuffers();
	}

	for (Workspace &workspace : workspaces) {

		if (workspace.command_buffer != VK_NULL_HANDLE) {
			vkFreeCommandBuffers(rtg.device, command_pool, 1, &workspace.command_buffer);
			workspace.command_buffer = VK_NULL_HANDLE;
		}
		if(workspace.lines_vertices_src.handle != VK_NULL_HANDLE){
			rtg.helpers.destroy_buffer(std::move(workspace.lines_vertices_src));
		}
		if(workspace.lines_vertices.handle != VK_NULL_HANDLE){
			rtg.helpers.destroy_buffer(std::move(workspace.lines_vertices));
		}
		if (workspace.Camera_src.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.Camera_src));
		}
		if (workspace.Camera.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.Camera));
		}
		//Camera_descriptors freed when pool is destroyed.
		if (workspace.World_src.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.World_src));
		}
		if (workspace.World.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.World));
		}
		//World_descriptors freed when pool is destroyed.
		if (workspace.Transforms_src.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.Transforms_src));
		}
		if (workspace.Transforms.handle != VK_NULL_HANDLE) {
			rtg.helpers.destroy_buffer(std::move(workspace.Transforms));
		}
		//Transforms_descriptors freed when pool is destroyed.
	}
	workspaces.clear();

	if (descriptor_pool) {
		vkDestroyDescriptorPool(rtg.device, descriptor_pool, nullptr);
		descriptor_pool = nullptr;
		//(this also frees the descriptor sets allocated from the pool)
	}

	background_pipeline.destroy(rtg);
	lines_pipeline.destroy(rtg);
	objects_pipeline.destroy(rtg);

	if (command_pool != VK_NULL_HANDLE) {
		vkDestroyCommandPool(rtg.device, command_pool, nullptr);
		command_pool = VK_NULL_HANDLE;
	}

	if (render_pass != VK_NULL_HANDLE) {
		vkDestroyRenderPass(rtg.device, render_pass, nullptr);
		render_pass = VK_NULL_HANDLE;
	}

}

void Tutorial::on_swapchain(RTG &rtg_, RTG::SwapchainEvent const &swapchain) {
	//[re]create framebuffers:
	//clean up existing framebuffers
	if(swapchain_depth_image.handle != VK_NULL_HANDLE){
		destroy_framebuffers();
	}	
	//allocate depth image for framebuffers to share
	swapchain_depth_image = rtg.helpers.create_image(
		swapchain.extent, 
		depth_format, 
		VK_IMAGE_TILING_OPTIMAL, 
		VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		Helpers::Unmapped
	);
	{ //create depth image view:
		VkImageViewCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
			.image = swapchain_depth_image.handle,
			.viewType = VK_IMAGE_VIEW_TYPE_2D,
			.format = depth_format,
			.subresourceRange{
				.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
				.baseMipLevel = 0,
				.levelCount = 1,
				.baseArrayLayer = 0,
				.layerCount = 1
			},
		};
		VK( vkCreateImageView(rtg.device, &create_info, nullptr, &swapchain_depth_image_view) );
	}
	//Make framebuffers for each swapchain image:
	swapchain_framebuffers.assign(swapchain.image_views.size(), VK_NULL_HANDLE);
	for (size_t i = 0; i < swapchain.image_views.size(); ++i) {
		std::array< VkImageView, 2 > attachments{
			swapchain.image_views[i],
			swapchain_depth_image_view,
		};
		VkFramebufferCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
			.renderPass = render_pass,
			.attachmentCount = uint32_t(attachments.size()),
			.pAttachments = attachments.data(),
			.width = swapchain.extent.width,
			.height = swapchain.extent.height,
			.layers = 1,
		};

		VK( vkCreateFramebuffer(rtg.device, &create_info, nullptr, &swapchain_framebuffers[i]) );
	}
}

void Tutorial::destroy_framebuffers() {
	for (VkFramebuffer &framebuffer : swapchain_framebuffers) {
		assert(framebuffer != VK_NULL_HANDLE);
		vkDestroyFramebuffer(rtg.device, framebuffer, nullptr);
		framebuffer = VK_NULL_HANDLE;
	}
	swapchain_framebuffers.clear();

	assert(swapchain_depth_image_view != VK_NULL_HANDLE);
	vkDestroyImageView(rtg.	device, swapchain_depth_image_view, nullptr);
	swapchain_depth_image_view = VK_NULL_HANDLE;

	rtg.helpers.destroy_image(std::move(swapchain_depth_image));
}


void Tutorial::render(RTG &rtg_, RTG::RenderParams const &render_params) {
	//assert that parameters are valid:
	assert(&rtg == &rtg_);
	assert(render_params.workspace_index < workspaces.size());
	assert(render_params.image_index < swapchain_framebuffers.size());
	
	//get more convenient names for the current workspace and target framebuffer:
	Workspace &workspace = workspaces[render_params.workspace_index];
	[[maybe_unused]] VkFramebuffer framebuffer = swapchain_framebuffers[render_params.image_index];
	
	VK(vkResetCommandBuffer(workspace.command_buffer, 0));
	{//begin recording
		VkCommandBufferBeginInfo begin_info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};
		VK(vkBeginCommandBuffer(workspace.command_buffer, &begin_info));
	}
	//upload vertices
	if(!lines_vertices.empty()){
		//allocate or reallocate line buffer as needed
		size_t needed_bytes = lines_vertices.size() * sizeof(lines_vertices[0]);
		if(workspace.lines_vertices_src.handle == VK_NULL_HANDLE || workspace.lines_vertices_src.size < needed_bytes){
			//resize rounding up to 4k
			size_t new_bytes = ((needed_bytes + 4096) / 4096) * 4096;
			//clean-up code for buffers if they are already allocated(destroy and reallocate)
			if(workspace.lines_vertices_src.handle){
				rtg.helpers.destroy_buffer(std::move(workspace.lines_vertices_src));
			}
			if(workspace.lines_vertices.handle){
				rtg.helpers.destroy_buffer(std::move(workspace.lines_vertices));
			}
			//allocate the buffers
			workspace.lines_vertices_src = rtg.helpers.create_buffer(
				new_bytes,
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				Helpers::Mapped	
			);
			workspace.lines_vertices = rtg.helpers.create_buffer(
				new_bytes,
				VK_BUFFER_USAGE_VERTEX_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				Helpers::Unmapped	
			);
			std::cout<<"Re-allocated lines buffer to " << new_bytes << " bytes."<<std::endl;
		}
		assert(workspace.lines_vertices_src.size ==workspace.lines_vertices.size);
		assert(workspace.lines_vertices_src.size >= needed_bytes);
		//host side: copy cpu memory into staging buffer lines_vertices_src
		assert(workspace.lines_vertices_src.allocation.mapped);
		std::memcpy(workspace.lines_vertices_src.allocation.data(), lines_vertices.data(), needed_bytes);
		
		//copy device memory to GPU memory
		VkBufferCopy copy_region{
			.srcOffset = 0,
			.dstOffset = 0,
			.size = needed_bytes,
		};
		vkCmdCopyBuffer(workspace.command_buffer, workspace.lines_vertices_src.handle, workspace.lines_vertices.handle,1,&copy_region);
	}

	{//upload camera info:
		LinesPipeline::Camera camera{
			.CLIP_FROM_WORLD = CLIP_FROM_WORLD,
		};
		assert(workspace.Camera_src.size == sizeof(camera));

		//host side: copy into src
		memcpy(workspace.Camera_src.allocation.data(), &camera, sizeof(camera));

		//device side copy from Camera_src to GPU memory Camera
		assert(workspace.Camera_src.size == workspace.Camera.size);
		VkBufferCopy copy_region{
			.srcOffset = 0,
			.dstOffset = 0,
			.size = workspace.Camera_src.size,
		};
		vkCmdCopyBuffer(workspace.command_buffer, workspace.Camera_src.handle, workspace.Camera.handle,1, &copy_region);
	}

	{ //upload world info:
		assert(workspace.Camera_src.size == sizeof(world));

		//host-side copy into World_src:
		memcpy(workspace.World_src.allocation.data(), &world, sizeof(world));

		//add device-side copy from World_src -> World:
		assert(workspace.World_src.size == workspace.World.size);
		VkBufferCopy copy_region{
			.srcOffset = 0,
			.dstOffset = 0,
			.size = workspace.World_src.size,
		};
		vkCmdCopyBuffer(workspace.command_buffer, workspace.World_src.handle, workspace.World.handle, 1, &copy_region);
	}

	//upload object transforms
	if(!object_instances.empty()){
		//allocate or reallocate transforms buffer as needed
		size_t needed_bytes = object_instances.size() * sizeof(ObjectsPipeline::Transform);
		if(workspace.Transforms_src.handle == VK_NULL_HANDLE || workspace.Transforms_src.size < needed_bytes){
			//resize rounding up to 4k
			size_t new_bytes = ((needed_bytes + 4096) / 4096) * 4096;
			//clean-up code for buffers if they are already allocated(destroy and reallocate)
			if(workspace.Transforms_src.handle){
				rtg.helpers.destroy_buffer(std::move(workspace.Transforms_src));
			}
			if(workspace.Transforms.handle){
				rtg.helpers.destroy_buffer(std::move(workspace.Transforms));
			}
			//allocate the buffers
			workspace.Transforms_src = rtg.helpers.create_buffer(
				new_bytes,
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				Helpers::Mapped	
			);
			workspace.Transforms = rtg.helpers.create_buffer(
				new_bytes,
				//going to use as storage buffer, also going to have GPU into this memory
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				Helpers::Unmapped	
			);

			//update descriptor set
			VkDescriptorBufferInfo Transforms_info{
				.buffer = workspace.Transforms.handle,
				.offset = 0,
				.range = workspace.Transforms.size,
			};
			std::array<VkWriteDescriptorSet,1> writes{
				VkWriteDescriptorSet{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = workspace.Transforms_descriptors,
					.dstBinding = 0,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.pBufferInfo = &Transforms_info,
				},
			};
			vkUpdateDescriptorSets(rtg.device, uint32_t(writes.size()), writes.data(), 0, nullptr);
			std::cout<<"Re-allocated transforms buffer to " << new_bytes << " bytes."<<std::endl;			
		}
		assert(workspace.Transforms_src.size ==workspace.Transforms.size);
		assert(workspace.Transforms_src.size >= needed_bytes);
		{//copy transforms into Transforms_src
			assert(workspace.Transforms_src.allocation.mapped);
			// Strict aliasing violation, but it doesn't matter
			ObjectsPipeline::Transform * out = reinterpret_cast< ObjectsPipeline::Transform * >(workspace.Transforms_src.allocation.data()); 
			for (ObjectInstance const &inst : object_instances) {
				*out = inst.transform;
				++out;
			}
		}
		
		//copy device memory to GPU memory
		VkBufferCopy copy_region{
			.srcOffset = 0,
			.dstOffset = 0,
			.size = needed_bytes,
		};
		vkCmdCopyBuffer(workspace.command_buffer, workspace.Transforms_src.handle, workspace.Transforms.handle,1,&copy_region);
	}

	{//memory barriers to make sure the copy finishes before the render happens
		VkMemoryBarrier memory_barrier{
			.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
		};
		vkCmdPipelineBarrier(workspace.command_buffer, 
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
			0,
			1, &memory_barrier,
			0, nullptr,
			0,nullptr
		);
	}
	{//render pass
		std::array< VkClearValue, 2 > clear_values{
			VkClearValue{ .color{ .float32{0.7f,0.9f,0.3f,1.0f} } },
			VkClearValue{ .depthStencil{ .depth = 1.0f, .stencil = 0 } },
		}; 
		
		VkRenderPassBeginInfo begin_info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = render_pass,
			.framebuffer = framebuffer,
			.renderArea{.offset = {.x = 0, .y =  0}, 
				.extent = rtg.swapchain_extent
			},
			.clearValueCount = uint32_t(clear_values.size()),
			.pClearValues = clear_values.data(),
		};
		vkCmdBeginRenderPass(workspace.command_buffer, &begin_info, VK_SUBPASS_CONTENTS_INLINE);

		{//scissor rectangle
			VkRect2D scissor{
				.offset = {.x = 0, .y = 0},
				.extent = rtg.swapchain_extent,
			};
			vkCmdSetScissor(workspace.command_buffer, 0, 1, &scissor);
		}
		{//viewport transform 
			VkViewport viewport{
				.x = 0.0f,
				.y =0.0f,
				.width = float(rtg.swapchain_extent.width),
				.height = float(rtg.swapchain_extent.height),
				.minDepth = 0.0f,
				.maxDepth = 1.0f,
			};
			vkCmdSetViewport(workspace.command_buffer, 0, 1, &viewport);
		}
		{//draw with background pipeline
			vkCmdBindPipeline(workspace.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, background_pipeline.handle);
			{//push the constants
				BackgroundPipeline::Push push{
					.time = time,
				};
				vkCmdPushConstants(workspace.command_buffer, background_pipeline.layout, VK_SHADER_STAGE_FRAGMENT_BIT,
				0, sizeof(push), &push);
			}
			vkCmdDraw(workspace.command_buffer, 3, 1, 0, 0);
		}
		if (!lines_vertices.empty() && workspace.lines_vertices.handle != VK_NULL_HANDLE){//draw with the lines pipeline
			vkCmdBindPipeline(workspace.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, lines_pipeline.handle);
			{//bind vertex buffer at buffer binding 0 using line_vertices
				if(workspace.lines_vertices.handle == VK_NULL_HANDLE) std::cout<<"gotcah"<<std::endl;
				std::array<VkBuffer, 1> vertex_buffers{workspace.lines_vertices.handle};
				std::array<VkDeviceSize, 1> offsets{0};
				vkCmdBindVertexBuffers(workspace.command_buffer, 0, uint32_t(vertex_buffers.size()), vertex_buffers.data(), offsets.data());
			}

			{//bind Camera descriptor set:
				std::array<VkDescriptorSet,1> descriptor_sets{
					workspace.Camera_descriptors,
				};
				vkCmdBindDescriptorSets(workspace.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, lines_pipeline.layout,
				0,uint32_t(descriptor_sets.size()), descriptor_sets.data(), 0, nullptr);
			}
			//draw lines vertices
			vkCmdDraw(workspace.command_buffer, uint32_t(lines_vertices.size()), 1, 0, 0);
		}
		if(!object_instances.empty()){//draw with the objects pipeline
			vkCmdBindPipeline(workspace.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, objects_pipeline.handle);
			{//use object_vertices (offset 0) as vertex buffer binding 0:
				std::array<VkBuffer,1> vertex_buffers{object_vertices.handle};
				std::array<VkDeviceSize, 1> offsets{0};
				vkCmdBindVertexBuffers(workspace.command_buffer, 0, uint32_t(vertex_buffers.size()), vertex_buffers.data(), offsets.data());
			}
			{////bind World and Transforms descriptor sets:
				std::array<VkDescriptorSet, 2> descriptor_sets{
					workspace.World_descriptors,//0, Transforms
					workspace.Transforms_descriptors,//1, Transforms
				};
				vkCmdBindDescriptorSets(workspace.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, objects_pipeline.layout,
					0, uint32_t(descriptor_sets.size()),descriptor_sets.data(), 0, nullptr);
			}

			//Camera descriptor set is still bound but unused
			//draw dat ting
			for (ObjectInstance const &inst : object_instances) {
				uint32_t index = uint32_t(&inst - &object_instances[0]);

				//bind texture descriptor set
				vkCmdBindDescriptorSets(
					workspace.command_buffer,
					VK_PIPELINE_BIND_POINT_GRAPHICS,
					objects_pipeline.layout,
					2, 1, &texture_descriptors[inst.texture],
					0,nullptr
				);
				vkCmdDraw(workspace.command_buffer, inst.vertices.count, 1, inst.vertices.first, index);
			}
		}

		vkCmdEndRenderPass(workspace.command_buffer);
	}
	//End Recording
	VK(vkEndCommandBuffer(workspace.command_buffer));
	
	{//submit `workspace.command buffer` for the GPU to run:
		std::array<VkSemaphore,1> wait_semaphores{
			render_params.image_available,
		};
		std::array<VkPipelineStageFlags, 1> wait_stages{
			VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
		};
		std::array<VkSemaphore, 1> signal_semaphores{
			render_params.image_done,
		}; 
		VkSubmitInfo submit_info{
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.waitSemaphoreCount = (uint32_t)(wait_semaphores.size()),
			.pWaitSemaphores = wait_semaphores.data(),
			.pWaitDstStageMask = wait_stages.data(),
			.commandBufferCount = 1,
			.pCommandBuffers = &workspace.command_buffer,
			.signalSemaphoreCount = (uint32_t)signal_semaphores.size(),
			.pSignalSemaphores = signal_semaphores.data(),
		};
		VK(vkQueueSubmit(rtg.graphics_queue, 1, &submit_info, render_params.workspace_available));
	}
}


void Tutorial::update(float dt) {
	time = std::fmod(time + dt, 5.0f);
	time_elapsed += dt;
	

	//TODO: if there is no change in any of the nodes, the chunk below shouldn't be run 
	object_instances.clear();
	loaded_cameras.clear();
	{//load s72 into the object instances, two lights, and cameras
		struct Item {
			S72::Node* node;
			mat4 parentWorld;
		};
		std::deque<Item> current_nodes;
		for(auto n : scene72.scene.roots){//start with the root nodes
			current_nodes.emplace_back(Item{n,n->parent_from_local()});
		}

		
		while(!current_nodes.empty()){//go through the graph using this queue bfs setup (this creates two instances of a child if two nodes have it as one of the children)
			auto [node, world_from_parent] = current_nodes.front();
			current_nodes.pop_front();

			//this node's world transform
			mat4 world_from_local = world_from_parent * node->parent_from_local();//accumulate transform

			if(node->mesh != nullptr){//if the node has mesh, instance it
				assert(mesh_vertices.find(node->mesh->name) != mesh_vertices.end());//check that the mesh's vertices has been loaded
				//std::cout<<"instancing "<<node->mesh->name<<std::endl;
				object_instances.emplace_back(
					ObjectInstance{
						.vertices = mesh_vertices[node->mesh->name],
						.transform{
							.CLIP_FROM_LOCAL = CLIP_FROM_WORLD * world_from_local,
							.WORLD_FROM_LOCAL = world_from_local,
							.WORLD_FROM_LOCAL_NORMAL = world_from_local,//not correct, TODO	
						},
						.texture = 0,
					}
				);			
			}

			 
			if(node->camera != nullptr){//there could be numerous cameras, but every camera has only one instance 
				//put every unique camera in the unordered_map, if there is a duplicate, print err and exit
				
				if(loaded_cameras.find(node->camera->name) != loaded_cameras.end()){
					std::cout<<"skipping camera "<<node->camera->name<<std::endl;
					continue;//skip duplicate cameras
				}
				assert(!node->camera->projection.valueless_by_exception());

				S72::Camera::Perspective perspective = get<S72::Camera::Perspective>(node->camera->projection);
				loaded_cameras[node->camera->name] = BasicCamera{
					.eye = world_from_local.translation(),
					.dir = (world_from_local * vec4{0,0,-1,0}).xyz(),
					.up = (world_from_local * vec4{0,1,0,0}).xyz(),
					.aspect = perspective.aspect,
					.vfov = perspective.vfov,
					.near = perspective.near,
					.far = perspective.far,
				};
			}
			if(node->light != nullptr){//two lights for A1

				//resolve the variant				
				std::variant< S72::Light::Sun, S72::Light::Sphere, S72::Light::Spot > &v = node->light->source;
				vec3 world_dir = normalized((world_from_local * vec4{0,0,-1,0}).xyz());
				if(std::holds_alternative<S72::Light::Sun>(v)){
					default_world_lights = false;
					S72::Light::Sun &sun = get<S72::Light::Sun>(v);
					if(std::abs(sun.angle - 3.14156926) < 0.001){// a hemisphere light
						world.SKY_DIRECTION.x = world_dir.x;
						world.SKY_DIRECTION.y = world_dir.y;
						world.SKY_DIRECTION.z = world_dir.z;

						world.SKY_ENERGY.r = (node->light->tint * sun.strength).x;
						world.SKY_ENERGY.g = (node->light->tint * sun.strength).y;
						world.SKY_ENERGY.b = (node->light->tint * sun.strength).z;
					}
					else{
						world.SUN_DIRECTION.x = world_dir.x;
						world.SUN_DIRECTION.y = world_dir.y;
						world.SUN_DIRECTION.z = world_dir.z;
						world.SUN_ENERGY.r = (node->light->tint * sun.strength).x;
						world.SUN_ENERGY.g = (node->light->tint * sun.strength).y;
						world.SUN_ENERGY.b = (node->light->tint * sun.strength).z;
					}
				}
				else if(std::holds_alternative<S72::Light::Sphere>(v)){

				}

				else if(std::holds_alternative<S72::Light::Spot>(v)){

				}
			}
			for(S72::Node *child : node->children){
				current_nodes.emplace_back(child, world_from_local);
			}
		}
		if(rtg.configuration.required_camera != "") {// if there is a command-line specified camera
			if(loaded_cameras.find(rtg.configuration.required_camera) == loaded_cameras.end()){//and you can't find it *~*
				throw std::runtime_error(
					"Required camera named '" + rtg.configuration.required_camera +
					"' was not found");
			}
			//however if you do find it !o!
			current_camera = loaded_cameras.find(rtg.configuration.required_camera);
		}
		else{//if there is no camera specified
			current_camera = loaded_cameras.begin();
		}
	}

	if(camera_mode == CameraMode::Scene){//unresponsive camera orbiting the origin
		if(rtg.configuration.required_camera == ""){//hard coded scene camera that rotates around target
			float ang = float(M_PI) * 2.0f * (time/5.0f);
			CLIP_FROM_WORLD = perspective(
				60.0f * float(M_PI) / 180.0f, //vfov
				rtg.swapchain_extent.width / float(rtg.swapchain_extent.height), //aspect
				0.1f, //near
				1000.0f //far
			) * look_at(
				vec3(13.0f * std::cos(ang), 13.0f * std::sin(ang), 5.0f), //eye
				vec3(0.0f, 0.0f, 5.0f), //target
				vec3(0.0f, 0.0f, 1.0f) //up
			);
		}
		else{//fixed, potentially keyframed camera that is loaded from s72 file
			CLIP_FROM_WORLD = current_camera->second.clip_from_world();
		}
	} else if(camera_mode == CameraMode::Free){
		CLIP_FROM_WORLD = perspective(
			60.0f * float(M_PI) / 180.0f, //vfov
			rtg.swapchain_extent.width / float(rtg.swapchain_extent.height), //aspect
			0.1f, //near
			1000.0f //far
		) * orbit(free_camera.target, free_camera.azimuth, free_camera.elevation, free_camera.radius);
	}else {
		assert(0 && "only two camera modes");
	}
	
	
	if(default_world_lights){ //moving sun and sky:
		float cycle = (sin(6.28f * time / 5.0f) + 0.8f) / 1.8f;
		world.SKY_DIRECTION.x = 0.0f;
		world.SKY_DIRECTION.y = 0.0f;
		world.SKY_DIRECTION.z = sin(6.28f * time / 5.0f)-0.3f;

		world.SKY_ENERGY.r = sin(6.28f * time / 5.0f)-0.3f;
		world.SKY_ENERGY.g = sin(6.28f * time / 5.0f)-0.3f;
		world.SKY_ENERGY.b = 0.2f;

		world.SUN_DIRECTION.x = 0.0f;
		world.SUN_DIRECTION.y = sin(6.28f * time / 5.0f);
		world.SUN_DIRECTION.z = cos(6.28f * time / 5.0f)-0.3f;

		world.SUN_ENERGY.r = cycle;
		world.SUN_ENERGY.g = cycle;
		world.SUN_ENERGY.b = cycle;
	}


	// {//lines stuff
	// 	if(starts.size() > 64){
	// 		std::cout<<"too many nodes"<<std::endl;
	// 		object_instances.clear();
	// 		starts.clear();
	// 		lines_vertices.clear();
	// 		iters = 0;
	// 	}
	// 	if(lines_vertices.size() > 1024){
	// 		std::cout<<"too many vertices"<<std::endl;
	// 		object_instances.clear();
	// 		starts.clear();
	// 		lines_vertices.clear();
	// 		iters = 0;
	// 	}

	// 	if(starts.empty())starts.emplace_back(vec3(0.0f,0.0f,0.0f));
	// 	if(time_elapsed > 0.5f && growing){
	// 		size_t num_nodes = starts.size();
			
	// 		auto verts = mesh_vertices.begin();//go through the mesh vertcies (like with poker cards) to grow them on trees
	// 		for(size_t i = 0; i < num_nodes; ++i){
	// 			vec3 cur_node = starts[i];
	// 			starts.emplace_back(emplace_random_line(cur_node,iters));
				
	// 			if(dist(engine) > 0.2f){
	// 				vec3 fruit_node = emplace_random_line(cur_node,iters);
	// 				//make fruits
	// 				if(dist(engine) > 1.0f-(iters-5) * 0.07f && iters>5){
						
	// 					{//spiky ball shrunken by a factor
							
	// 						float scaling_factor = 0.5f;
	// 						mat4 WORLD_FROM_LOCAL{
	// 							scaling_factor, 0.0f,  0.0f, 0.0f,
	// 							0.0f,scaling_factor, 0.0f, 0.0f,
	// 							0.0f, 0.0f,   scaling_factor, 0.0f,
	// 							fruit_node.x,fruit_node.y,fruit_node.z, 1.0f,
	// 						};
	// 						object_instances.emplace_back(ObjectInstance{
	// 							.vertices = fruit_vertices,//which vertices to use
	// 							.transform{
	// 								.CLIP_FROM_LOCAL = CLIP_FROM_WORLD * WORLD_FROM_LOCAL,
	// 								.WORLD_FROM_LOCAL = WORLD_FROM_LOCAL,
	// 								.WORLD_FROM_LOCAL_NORMAL = WORLD_FROM_LOCAL,	
	// 							},
	// 							.texture = 1,
	// 						});
	// 					}
	// 					verts++;
	// 					if(verts == mesh_vertices.end())verts = mesh_vertices.begin();
	// 				}
	// 				//else branch
	// 				else starts.emplace_back(fruit_node);
	// 			}
	// 		}
	// 		// Remove the old nodes (keep only the new branches)
    //     	starts.erase(starts.begin(), starts.begin() + num_nodes);
	// 		time_elapsed = 0.0f;
	// 		iters++;
	// 	}
	// }	

	for(auto & obj: object_instances){
		//update camera matrix
		obj.transform.CLIP_FROM_LOCAL = CLIP_FROM_WORLD * obj.transform.WORLD_FROM_LOCAL;
	}
	
	{//make some objects
		
		// { //plane translated +x by one unit:
		// 	mat4 WORLD_FROM_LOCAL{
		// 		1.0f, 0.0f, 0.0f, 0.0f,
		// 		0.0f, 1.0f, 0.0f, 0.0f,
		// 		0.0f, 0.0f, 1.0f, 0.0f,
		// 		1.0f, 0.0f, 0.0f, 1.0f,
		// 	};

		// 	object_instances.emplace_back(ObjectInstance{
		// 		.vertices = plane_vertices,
		// 		.transform{
		// 			.CLIP_FROM_LOCAL = CLIP_FROM_WORLD * WORLD_FROM_LOCAL,
		// 			.WORLD_FROM_LOCAL = WORLD_FROM_LOCAL,
		// 			.WORLD_FROM_LOCAL_NORMAL = WORLD_FROM_LOCAL,
		// 		},
		// 		.texture = 1,
		// 	});
		// }
		// { //torus translated -x by one unit and rotated CCW around +y:
		// 	float ang = time / 60.0f * 2.0f * float(M_PI) * 10.0f;
		// 	float ca = std::cos(ang);
		// 	float sa = std::sin(ang);
		// 	mat4 WORLD_FROM_LOCAL{
		// 		  ca, 0.0f,  -sa, 0.0f,
		// 		0.0f, 1.0f, 0.0f, 0.0f,
		// 		  sa, 0.0f,   ca, 0.0f,
		// 		-1.0f,0.0f, 0.0f, 1.0f,
		// 	};

		// 	object_instances.emplace_back(ObjectInstance{
		// 		.vertices = torus_vertices,
		// 		.transform{
		// 			.CLIP_FROM_LOCAL = CLIP_FROM_WORLD * WORLD_FROM_LOCAL,
		// 			.WORLD_FROM_LOCAL = WORLD_FROM_LOCAL,
		// 			.WORLD_FROM_LOCAL_NORMAL = WORLD_FROM_LOCAL,
		// 		},
		// 	});
		// }
		// {//spiky ball shrunken by a factor of 0.5
		// 	float scaling_factor = 0.5f;
		// 	mat4 WORLD_FROM_LOCAL{
		// 		scaling_factor, 0.0f,  0.0f, 0.0f,
		// 		0.0f,scaling_factor, 0.0f, 0.0f,
		// 		0.0f, 0.0f,   scaling_factor, 0.0f,
		// 		0.0f,0.0f, 0.0f, 1.0f,
		// 	};
		// 	object_instances.emplace_back(ObjectInstance{
		// 		.vertices = fruit_vertices,
		// 		.transform{
		// 			.CLIP_FROM_LOCAL = CLIP_FROM_WORLD * WORLD_FROM_LOCAL,
		// 			.WORLD_FROM_LOCAL = WORLD_FROM_LOCAL,
		// 			.WORLD_FROM_LOCAL_NORMAL = WORLD_FROM_LOCAL,	
		// 		},
		// 		.texture = 1,
		// 	});
		// }
	}
}

vec3 Tutorial::emplace_random_line(vec3 start, uint32_t iter){
	//do some approximation of tree growing based on current iteration number and height

	float length_modifier = powf(0.9f, (float)iter);//length gets smaller
	
	float up_modifier = powf(0.3f, (float)iter);//tree grows less and less "up"

	vec3 growth = vec3(dist(engine), dist(engine), up_modifier + std::abs(dist(engine)));

	vec3 new_location = start + length_modifier * normalized(growth);
	float color_key = (dist(engine) + 1.0f)/2; 
	uint8_t r = (static_cast<uint8_t>(std::floor(color_key * 256.0f)));
	uint8_t g = (static_cast<uint8_t>(std::floor(color_key * 486.0f * color_key)));
	uint8_t b = (static_cast<uint8_t>(std::floor(color_key * 238.0f)));
	uint8_t a = 1;
	lines_vertices.emplace_back(PosColVertex{
		.Position{.x = start.x, .y = start.y, .z = start.z},
		.Color{.r = r, .g = g, .b = b, .a = a},
	});
	lines_vertices.emplace_back(PosColVertex{
		.Position{.x = new_location.x, .y = new_location.y, .z = new_location.z},
	});
	return new_location;
}


void Tutorial::on_input(InputEvent const & evt) {
	//if there is a current action, it gets input priority:
	if(action){
		action(evt);
		return;
	}
	//general controls
	if(evt.type == InputEvent::KeyDown && evt.key.key == GLFW_KEY_TAB){
		//switch camera modes
		camera_mode = CameraMode((int(camera_mode) + 1) % 2);
		return;
	}
	if(evt.type == InputEvent::KeyDown && evt.key.key == GLFW_KEY_G){
		//toggle growing
		growing = !growing;
		return;
	}

	//free camera controls
	if(camera_mode == CameraMode::Free){
		if(evt.type == InputEvent::MouseWheel){
			//zoom in/out
			free_camera.radius *= std::pow(1.1f , -evt.wheel.y);
			free_camera.radius = std::min(free_camera.radius, free_camera.far);
			free_camera.radius = std::max(free_camera.radius, free_camera.near);
		}
		if(evt.type == InputEvent::MouseButtonDown && evt.button.button == GLFW_MOUSE_BUTTON_LEFT && !(evt.button.mods & GLFW_MOD_SHIFT)){
			//start tumbling
			float init_x = evt.button.x;
			float init_y = evt.button.y;
			OrbitCamera init_camera = free_camera;

			action = [this, init_x, init_y, init_camera](InputEvent const &evt){
				if(evt.type == InputEvent::MouseButtonUp && evt.button.button ==GLFW_MOUSE_BUTTON_LEFT){
					//cancel upon button lifted:
					action = nullptr;
					return;
				}
				if(evt.type == InputEvent::MouseMotion){
					//handle panning
					float dx = (evt.motion.x - init_x) / rtg.swapchain_extent.height;
					float dy =-(evt.motion.y - init_y) / rtg.swapchain_extent.height; //note: negated because glfw uses y-down coordinate system
					
					//rotate camera based on motion:
					float speed = float(M_PI);
					float flip_x = (std::abs(init_camera.elevation) > 0.5f * float(M_PI) ? -1.0f : 1.0f); //switch azimuth rotation when camera is upside-down
					free_camera.azimuth = init_camera.azimuth - dx * speed * flip_x;
					free_camera.elevation = init_camera.elevation - dy * speed;

					//reduce azimuth and elevation to [-pi,pi] range:
					const float twopi = 2.0f * float(M_PI);
					free_camera.elevation -= std::round(free_camera.elevation / twopi) * twopi;
					free_camera.azimuth -= std::round(free_camera.azimuth / twopi) * twopi;
					return;
				}
			};
			return;
		}
		if(evt.type == InputEvent::MouseButtonDown && evt.button.button == GLFW_MOUSE_BUTTON_LEFT && (evt.button.mods & GLFW_MOD_SHIFT)){
			//start panning
			float init_x = evt.button.x;
			float init_y = evt.button.y;
			OrbitCamera init_camera = free_camera;
			std::cout<<"start panning"<<std::endl;
			//handle panning
			action = [this, init_x, init_y, init_camera](InputEvent const &evt){
				if(evt.type == InputEvent::MouseButtonUp && evt.button.button ==GLFW_MOUSE_BUTTON_LEFT){
					//cancel upon button lifted:
					action = nullptr;
					return;
				}
				if(evt.type == InputEvent::MouseMotion){
					//handle panning
					float height = 2.0f * std::tan(free_camera.fov * 0.5f) * free_camera.radius;
					//multiplying dx and dy by height because farther camera should move more so that stuff should glide across screen the same?
					float dx = (evt.motion.x - init_x) / rtg.swapchain_extent.height * height;
					float dy =-(evt.motion.y - init_y) / rtg.swapchain_extent.height * height; //note: negated because glfw uses y-down coordinate system
					
					//use orbit the extract right and up vectors
					mat4 camera_from_world = orbit(init_camera.target, init_camera.azimuth, init_camera.elevation, init_camera.radius);
					vec3 right = vec3(camera_from_world[0], camera_from_world[4], camera_from_world[8]);
					vec3 up = vec3(camera_from_world[1], camera_from_world[5], camera_from_world[9]);
					free_camera.target = init_camera.target - dx * right - dy * up;				
					return;
				}
			};
			return;
		}
	}
}
