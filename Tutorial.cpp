#include "Tutorial.hpp"

#include "VK.hpp"

#include <GLFW/glfw3.h>

#include <array>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <corecrt_math_defines.h>

#include <random>
#include <chrono>

unsigned long long seed = std::chrono::system_clock::now().time_since_epoch().count();
std::mt19937 engine((unsigned int)seed);
std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

Tutorial::Tutorial(RTG &rtg_) : rtg(rtg_)
{
	// command pool, descriptor pool,
	init_tutorial();

	std::cout << "LOADING SCENE" << std::endl;

	// set up loaded cameras, put mesh data into the object_vertices buffer
	load_scene();

	// load textures into texture_descriptor_sets
	load_textures();

	// stuff for shadow_mapping(framebuffers, allocating image and stuff)
	init_shadow_mapping();

	std::cout << "finished loading stuff" << std::endl;
}

Tutorial::~Tutorial()
{
	// just in case rendering is still in flight, don't destroy resources:
	//(not using VK macro to avoid throw-ing in destructor)
	if (VkResult result = vkDeviceWaitIdle(rtg.device); result != VK_SUCCESS)
	{
		std::cerr << "Failed to vkDeviceWaitIdle in Tutorial::~Tutorial [" << string_VkResult(result) << "]; continuing anyway." << std::endl;
	}

	// freeing texture
	if (texture_descriptor_pool)
	{
		vkDestroyDescriptorPool(rtg.device, texture_descriptor_pool, nullptr);
		texture_descriptor_pool = nullptr;
		// this also frees the descriptor sets allocated from the pool:
		texture_descriptor_sets.clear();
	}

	if (texture_sampler)
	{
		vkDestroySampler(rtg.device, texture_sampler, nullptr);
		texture_sampler = VK_NULL_HANDLE;
	}

	if (depth_texture_sampler)
	{
		vkDestroySampler(rtg.device, depth_texture_sampler, nullptr);
		depth_texture_sampler = VK_NULL_HANDLE;
	}

	for (VkImageView &view : texture_views)
	{
		vkDestroyImageView(rtg.device, view, nullptr);
		view = VK_NULL_HANDLE;
	}
	texture_views.clear();

	for (auto &texture : textures)
	{
		rtg.helpers.destroy_image(std::move(texture));
	}
	textures.clear();

	// freeing static buffers
	rtg.helpers.destroy_buffer(std::move(object_vertices));

	if (swapchain_depth_image.handle != VK_NULL_HANDLE)
	{
		destroy_framebuffers();
	}

	for (Workspace &workspace : workspaces)
	{

		// shadow stuff
		rtg.helpers.destroy_image(std::move(workspace.Shadow_Atlas));
		vkDestroyImageView(rtg.device, workspace.Shadow_Atlas_view, nullptr);
		workspace.Shadow_Atlas_view = nullptr;
		if (workspace.debug_buffer.handle != VK_NULL_HANDLE)
		{
			rtg.helpers.destroy_buffer(std::move(workspace.debug_buffer));
		}
		if (workspace.Shadow_Atlas_FB != VK_NULL_HANDLE)
		{
			vkDestroyFramebuffer(rtg.device, workspace.Shadow_Atlas_FB, nullptr);
			workspace.Shadow_Atlas_FB = VK_NULL_HANDLE;
		}
		if (workspace.command_buffer != VK_NULL_HANDLE)
		{
			vkFreeCommandBuffers(rtg.device, command_pool, 1, &workspace.command_buffer);
			workspace.command_buffer = VK_NULL_HANDLE;
		}
		if (workspace.lines_vertices_src.handle != VK_NULL_HANDLE)
		{
			rtg.helpers.destroy_buffer(std::move(workspace.lines_vertices_src));
		}
		if (workspace.lines_vertices.handle != VK_NULL_HANDLE)
		{
			rtg.helpers.destroy_buffer(std::move(workspace.lines_vertices));
		}
		if (workspace.Camera_src.handle != VK_NULL_HANDLE)
		{
			rtg.helpers.destroy_buffer(std::move(workspace.Camera_src));
		}
		if (workspace.Camera.handle != VK_NULL_HANDLE)
		{
			rtg.helpers.destroy_buffer(std::move(workspace.Camera));
		}
		if (workspace.Eye_src.handle != VK_NULL_HANDLE)
		{
			rtg.helpers.destroy_buffer(std::move(workspace.Eye_src));
		}
		if (workspace.Eye.handle != VK_NULL_HANDLE)
		{
			rtg.helpers.destroy_buffer(std::move(workspace.Eye));
		}
		// Camera_descriptors freed when pool is destroyed.
		if (workspace.Lights_src.handle != VK_NULL_HANDLE)
		{
			rtg.helpers.destroy_buffer(std::move(workspace.Lights_src));
		}
		if (workspace.Lights.handle != VK_NULL_HANDLE)
		{
			rtg.helpers.destroy_buffer(std::move(workspace.Lights));
		}
		// Lights_descriptors freed when pool is destroyed.
		if (workspace.Transforms_src.handle != VK_NULL_HANDLE)
		{
			rtg.helpers.destroy_buffer(std::move(workspace.Transforms_src));
		}
		if (workspace.Transforms.handle != VK_NULL_HANDLE)
		{
			rtg.helpers.destroy_buffer(std::move(workspace.Transforms));
		}
		// Transforms_descriptors freed when pool is destroyed.
	}
	workspaces.clear();

	if (descriptor_pool)
	{
		vkDestroyDescriptorPool(rtg.device, descriptor_pool, nullptr);
		descriptor_pool = nullptr;
		//(this also frees the descriptor sets allocated from the pool)
	}

	background_pipeline.destroy(rtg);
	lines_pipeline.destroy(rtg);
	lambertian_objects_pipeline.destroy(rtg);
	env_mirror_objects_pipeline.destroy(rtg);
	pbr_objects_pipeline.destroy(rtg);
	shadow_2D_pipeline.destroy(rtg);

	if (command_pool != VK_NULL_HANDLE)
	{
		vkDestroyCommandPool(rtg.device, command_pool, nullptr);
		command_pool = VK_NULL_HANDLE;
	}

	if (render_pass != VK_NULL_HANDLE)
	{
		vkDestroyRenderPass(rtg.device, render_pass, nullptr);
		render_pass = VK_NULL_HANDLE;
	}

	if (shadow_pass != VK_NULL_HANDLE)
	{
		vkDestroyRenderPass(rtg.device, shadow_pass, nullptr);
		shadow_pass = VK_NULL_HANDLE;
	}
}

void Tutorial::on_swapchain(RTG &rtg_, RTG::SwapchainEvent const &swapchain)
{
	//[re]create framebuffers:
	// clean up existing framebuffers
	if (swapchain_depth_image.handle != VK_NULL_HANDLE)
	{
		destroy_framebuffers();
	}
	// allocate depth image for framebuffers to share
	swapchain_depth_image = rtg.helpers.create_image(
		swapchain.extent,
		depth_format,
		VK_IMAGE_TILING_OPTIMAL,
		VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
		Helpers::Unmapped,
		false);
	{ // create depth image view:
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
				.layerCount = 1},
		};
		VK(vkCreateImageView(rtg.device, &create_info, nullptr, &swapchain_depth_image_view));
	}
	// Make framebuffers for each swapchain image:
	swapchain_framebuffers.assign(swapchain.image_views.size(), VK_NULL_HANDLE);
	for (size_t i = 0; i < swapchain.image_views.size(); ++i)
	{
		std::array<VkImageView, 2> attachments{
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

		VK(vkCreateFramebuffer(rtg.device, &create_info, nullptr, &swapchain_framebuffers[i]));
	}
}

void Tutorial::destroy_framebuffers()
{
	for (VkFramebuffer &framebuffer : swapchain_framebuffers)
	{
		assert(framebuffer != VK_NULL_HANDLE);
		vkDestroyFramebuffer(rtg.device, framebuffer, nullptr);
		framebuffer = VK_NULL_HANDLE;
	}
	swapchain_framebuffers.clear();

	assert(swapchain_depth_image_view != VK_NULL_HANDLE);
	vkDestroyImageView(rtg.device, swapchain_depth_image_view, nullptr);
	swapchain_depth_image_view = VK_NULL_HANDLE;

	rtg.helpers.destroy_image(std::move(swapchain_depth_image));
}

void Tutorial::render(RTG &rtg_, RTG::RenderParams const &render_params)
{
	// assert that parameters are valid:
	assert(&rtg == &rtg_);
	assert(render_params.workspace_index < workspaces.size());
	assert(render_params.image_index < swapchain_framebuffers.size());

	// get more convenient names for the current workspace and target framebuffer:
	Workspace &workspace = workspaces[render_params.workspace_index];
	[[maybe_unused]] VkFramebuffer framebuffer = swapchain_framebuffers[render_params.image_index];

	VK(vkResetCommandBuffer(workspace.command_buffer, 0));
	{ // begin recording
		VkCommandBufferBeginInfo begin_info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};
		VK(vkBeginCommandBuffer(workspace.command_buffer, &begin_info));
	}
	// upload vertices
	if (!lines_vertices.empty())
	{
		// allocate or reallocate line buffer as needed
		size_t needed_bytes = lines_vertices.size() * sizeof(lines_vertices[0]);
		if (workspace.lines_vertices_src.handle == VK_NULL_HANDLE || workspace.lines_vertices_src.size < needed_bytes)
		{
			// resize rounding up to 4k
			size_t new_bytes = ((needed_bytes + 4096) / 4096) * 4096;
			// clean-up code for buffers if they are already allocated(destroy and reallocate)
			if (workspace.lines_vertices_src.handle)
			{
				rtg.helpers.destroy_buffer(std::move(workspace.lines_vertices_src));
			}
			if (workspace.lines_vertices.handle)
			{
				rtg.helpers.destroy_buffer(std::move(workspace.lines_vertices));
			}
			// allocate the buffers
			workspace.lines_vertices_src = rtg.helpers.create_buffer(
				new_bytes,
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				Helpers::Mapped);
			workspace.lines_vertices = rtg.helpers.create_buffer(
				new_bytes,
				VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				Helpers::Unmapped);
			std::cout << "Re-allocated lines buffer to " << new_bytes << " bytes." << std::endl;
		}
		assert(workspace.lines_vertices_src.size == workspace.lines_vertices.size);
		assert(workspace.lines_vertices_src.size >= needed_bytes);
		// host side: copy cpu memory into staging buffer lines_vertices_src
		assert(workspace.lines_vertices_src.allocation.mapped);
		std::memcpy(workspace.lines_vertices_src.allocation.data(), lines_vertices.data(), needed_bytes);

		// copy device memory to GPU memory
		VkBufferCopy copy_region{
			.srcOffset = 0,
			.dstOffset = 0,
			.size = needed_bytes,
		};
		vkCmdCopyBuffer(workspace.command_buffer, workspace.lines_vertices_src.handle, workspace.lines_vertices.handle, 1, &copy_region);
	}

	{ // upload camera info:
		LinesPipeline::Camera camera{
			.CLIP_FROM_WORLD = CLIP_FROM_WORLD,
		};
		assert(workspace.Camera_src.size == sizeof(camera));

		// host side: copy into src
		memcpy(workspace.Camera_src.allocation.data(), &camera, sizeof(camera));

		// device side copy from Camera_src to GPU memory Camera
		assert(workspace.Camera_src.size == workspace.Camera.size);
		VkBufferCopy copy_region{
			.srcOffset = 0,
			.dstOffset = 0,
			.size = workspace.Camera_src.size,
		};
		vkCmdCopyBuffer(workspace.command_buffer, workspace.Camera_src.handle, workspace.Camera.handle, 1, &copy_region);
	}

	{ // upload lights info:
		if (workspace.Lights_src.handle == VK_NULL_HANDLE)
		{
			std::cout << "allocating the buffer for " << lights.size() << "lights" << std::endl;
			// allocate the buffers
			workspace.Lights_src = rtg.helpers.create_buffer(
				sizeof(Light) * lights.size(),
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				Helpers::Mapped);
			workspace.Lights = rtg.helpers.create_buffer(
				sizeof(Light) * lights.size(),
				// going to use as storage buffer, also going to have GPU into this memory
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				Helpers::Unmapped);
			assert(workspace.Lights_src.size == lights.size() * sizeof(Light));

			// update descriptor set
			VkDescriptorBufferInfo Lights_info{
				.buffer = workspace.Lights.handle,
				.offset = 0,
				.range = workspace.Lights.size,
			};
			std::array<VkWriteDescriptorSet, 1> writes{
				VkWriteDescriptorSet{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = workspace.Lights_descriptors,
					.dstBinding = 0,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.pBufferInfo = &Lights_info,
				},
			};
			vkUpdateDescriptorSets(rtg.device, uint32_t(writes.size()), writes.data(), 0, nullptr);
		}
		// host-side copy into Lights_src:
		memcpy(workspace.Lights_src.allocation.data(), lights.data(), lights.size() * sizeof(Light));

		// add device-side copy from Lights_src -> Lights:
		assert(workspace.Lights_src.size == workspace.Lights.size);
		VkBufferCopy copy_region{
			.srcOffset = 0,
			.dstOffset = 0,
			.size = workspace.Lights_src.size,
		};
		vkCmdCopyBuffer(workspace.command_buffer, workspace.Lights_src.handle, workspace.Lights.handle, 1, &copy_region);
	}

	{ // upload eye info:

		// host-side copy into Eye_src:
		memcpy(workspace.Eye_src.allocation.data(), &EYE, sizeof(EYE));

		// add device-side copy from Eye_src -> Eye:
		assert(workspace.Eye_src.size == workspace.Eye.size);
		VkBufferCopy copy_region{
			.srcOffset = 0,
			.dstOffset = 0,
			.size = workspace.Eye_src.size,
		};
		vkCmdCopyBuffer(workspace.command_buffer, workspace.Eye_src.handle, workspace.Eye.handle, 1, &copy_region);
	}

	// upload object transforms
	if (!lambertian_object_instances.empty() || !env_mirror_object_instances.empty() || !pbr_object_instances.empty())
	{ // upload object transforms
		// allocate or reallocate transforms buffer as needed
		size_t needed_bytes = (lambertian_object_instances.size() + env_mirror_object_instances.size() + pbr_object_instances.size()) * sizeof(Transform);
		if (workspace.Transforms_src.handle == VK_NULL_HANDLE || workspace.Transforms_src.size < needed_bytes)
		{
			// resize rounding up to 4k
			size_t new_bytes = ((needed_bytes + 4096) / 4096) * 4096;
			// clean-up code for buffers if they are already allocated(destroy and reallocate)
			if (workspace.Transforms_src.handle)
			{
				rtg.helpers.destroy_buffer(std::move(workspace.Transforms_src));
			}
			if (workspace.Transforms.handle)
			{
				rtg.helpers.destroy_buffer(std::move(workspace.Transforms));
			}
			// allocate the buffers
			workspace.Transforms_src = rtg.helpers.create_buffer(
				new_bytes,
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
				Helpers::Mapped);
			workspace.Transforms = rtg.helpers.create_buffer(
				new_bytes,
				// going to use as storage buffer, also going to have GPU into this memory
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
				Helpers::Unmapped);

			// update descriptor set
			VkDescriptorBufferInfo Transforms_info{
				.buffer = workspace.Transforms.handle,
				.offset = 0,
				.range = workspace.Transforms.size,
			};
			std::array<VkWriteDescriptorSet, 1> writes{
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
			std::cout << "Re-allocated transforms buffer to " << new_bytes << " bytes." << std::endl;
		}
		assert(workspace.Transforms_src.size == workspace.Transforms.size);
		assert(workspace.Transforms_src.size >= needed_bytes);
		{ // copy transforms into Transforms_src
			assert(workspace.Transforms_src.allocation.mapped);
			// Strict aliasing violation, but it doesn't matter
			Transform *out = reinterpret_cast<Transform *>(workspace.Transforms_src.allocation.data());
			for (LambertianObjectInstance const &inst : lambertian_object_instances)
			{
				*out = inst.transform;
				++out;
			}
			for (EnvMirrorObjectInstance const &inst : env_mirror_object_instances)
			{
				*out = inst.transform;
				++out;
			}
			for (PbrObjectInstance const &inst : pbr_object_instances)
			{
				*out = inst.transform;
				++out;
			}
		}

		// copy device memory to GPU memory
		VkBufferCopy copy_region{
			.srcOffset = 0,
			.dstOffset = 0,
			.size = needed_bytes,
		};
		vkCmdCopyBuffer(workspace.command_buffer, workspace.Transforms_src.handle, workspace.Transforms.handle, 1, &copy_region);
	}

	{ // memory barriers to make sure the copy finishes before the render happens
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
							 0, nullptr);
	}

	{ // shadow pass
		std::array<VkClearValue, 1> clear_values{
			VkClearValue{.depthStencil{.depth = 1.0f, .stencil = 0}},
		};

		VkRenderPassBeginInfo begin_info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = shadow_pass,
			.framebuffer = workspace.Shadow_Atlas_FB,
			.renderArea{
				.offset = {.x = 0, .y = 0},
				.extent = {.width = atlas_size, .height = atlas_size},
			},
			.clearValueCount = uint32_t(clear_values.size()),
			.pClearValues = clear_values.data(),
		};
		vkCmdBeginRenderPass(workspace.command_buffer, &begin_info, VK_SUBPASS_CONTENTS_INLINE);

		if (shadows_on)
		{ // draw with the shadow map pipeline
			vkCmdBindPipeline(workspace.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_2D_pipeline.handle);

			{ // use object_vertices (offset 0) as vertex buffer binding 0: they all share vertex buffer
				std::array<VkBuffer, 1> vertex_buffers{object_vertices.handle};
				std::array<VkDeviceSize, 1> offsets{0};
				vkCmdBindVertexBuffers(workspace.command_buffer, 0, uint32_t(vertex_buffers.size()), vertex_buffers.data(), offsets.data());
			}

			{ // bind Transform
				std::array<VkDescriptorSet, 1> descriptor_sets{
					workspace.Transforms_descriptors, // 0, Transforms
				};
				vkCmdBindDescriptorSets(workspace.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_2D_pipeline.layout,
										0, uint32_t(descriptor_sets.size()), descriptor_sets.data(), 0, nullptr);
			}

			for (uint32_t i = 0; i < lights.size(); i++)
			{
				if (lights[i].shadow_atlases[0].z == 0.0f)
				{
					continue;
				}
				// get the light's CLIP_FROM_WORLD and draw all the objects for each light
				if (lights[i].type == 0)
				{ // sun
					// TODO: do some shadow cascade thingy
				}
				else if (lights[i].type == 1)
				{ // sphere

					for (uint32_t j = 0; j < 6; j++)
					{	
						draw_all_objects(workspace.command_buffer, lights[i].CLIP_FROM_WORLD[j], lights[i].shadow_atlases[j]);
					}
				}
				else
				{ // spot
					// std::cout<<"computing CLIP_FROM_WORLD for: fov "<< lights[i].fov << "direction: "<<lights[i].direction.convert_to_string()<<std::endl;
					draw_all_objects(workspace.command_buffer, lights[i].CLIP_FROM_WORLD[0], lights[i].shadow_atlases[0]);
				}
			}
		}
		vkCmdEndRenderPass(workspace.command_buffer);
	}

	if(camera_mode == CameraMode::Debug){ // transfer shadow_atlas to debug buffer
		VkImageMemoryBarrier barrier{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT, // or DEPTH_WRITE if just rendered
			.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
			.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = workspace.Shadow_Atlas.handle,
			.subresourceRange = {
				VK_IMAGE_ASPECT_DEPTH_BIT,
				0, 1,
				0, 1}};

		vkCmdPipelineBarrier(
			workspace.command_buffer,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &barrier);

		VkBufferImageCopy region{
			.bufferOffset = 0,
			.bufferRowLength = 0,
			.bufferImageHeight = 0,
			.imageSubresource = {
				VK_IMAGE_ASPECT_DEPTH_BIT,
				0, 0, 1},
			.imageOffset = {0, 0, 0},
			.imageExtent = {atlas_size, atlas_size, 1}};

		vkCmdCopyImageToBuffer(
			workspace.command_buffer,
			workspace.Shadow_Atlas.handle,
			VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			workspace.debug_buffer.handle,
			1,
			&region);
		
		VkBufferMemoryBarrier buffer_barrier{
			.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
			.dstAccessMask = VK_ACCESS_HOST_READ_BIT,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.buffer = workspace.debug_buffer.handle,
			.offset = 0,
			.size = VK_WHOLE_SIZE
		};

		vkCmdPipelineBarrier(
			workspace.command_buffer,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_HOST_BIT,
			0,
			0, nullptr,
			1, &buffer_barrier,
			0, nullptr);
			
	
		VkImageMemoryBarrier barrier2{
			.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
			.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT, // or DEPTH_WRITE if just rendered
			.dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
			.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
			.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
			.image = workspace.Shadow_Atlas.handle,
			.subresourceRange = {
				VK_IMAGE_ASPECT_DEPTH_BIT,
				0, 1,
				0, 1}};

		vkCmdPipelineBarrier(
			workspace.command_buffer,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
			0,
			0, nullptr,
			0, nullptr,
			1, &barrier2);
	}




	{ // actual render pass
		std::array<VkClearValue, 2> clear_values{
			VkClearValue{.color{.float32{0.7f, 0.9f, 0.3f, 1.0f}}},
			VkClearValue{.depthStencil{.depth = 1.0f, .stencil = 0}},
		};

		VkRenderPassBeginInfo begin_info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = render_pass,
			.framebuffer = framebuffer,
			.renderArea{.offset = {.x = 0, .y = 0},
						.extent = rtg.swapchain_extent},
			.clearValueCount = uint32_t(clear_values.size()),
			.pClearValues = clear_values.data(),
		};
		vkCmdBeginRenderPass(workspace.command_buffer, &begin_info, VK_SUBPASS_CONTENTS_INLINE);

		{ // scissor rectangle
			VkRect2D scissor{
				.offset = {.x = 0, .y = 0},
				.extent = rtg.swapchain_extent,
			};
			vkCmdSetScissor(workspace.command_buffer, 0, 1, &scissor);
		}
		{ // viewport transform
			VkViewport viewport{
				.x = 0.0f,
				.y = 0.0f,
				.width = float(rtg.swapchain_extent.width),
				.height = float(rtg.swapchain_extent.height),
				.minDepth = 0.0f,
				.maxDepth = 1.0f,
			};
			vkCmdSetViewport(workspace.command_buffer, 0, 1, &viewport);
		}

		{ // draw with background pipeline
			vkCmdBindPipeline(workspace.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, background_pipeline.handle);
			{ // push the constants
				BackgroundPipeline::Push push{
					.time = time,
				};
				vkCmdPushConstants(workspace.command_buffer, background_pipeline.layout, VK_SHADER_STAGE_FRAGMENT_BIT,
								   0, sizeof(push), &push);
			}
			vkCmdDraw(workspace.command_buffer, 3, 1, 0, 0);
		}
		if (!lines_vertices.empty() && workspace.lines_vertices.handle != VK_NULL_HANDLE)
		{ // draw with the lines pipeline
			vkCmdBindPipeline(workspace.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, lines_pipeline.handle);
			{ // bind vertex buffer at buffer binding 0 using line_vertices
				if (workspace.lines_vertices.handle == VK_NULL_HANDLE)
					std::cout << "gotcah" << std::endl;
				std::array<VkBuffer, 1> vertex_buffers{workspace.lines_vertices.handle};
				std::array<VkDeviceSize, 1> offsets{0};
				vkCmdBindVertexBuffers(workspace.command_buffer, 0, uint32_t(vertex_buffers.size()), vertex_buffers.data(), offsets.data());
			}

			{ // bind Camera descriptor set:
				std::array<VkDescriptorSet, 1> descriptor_sets{
					workspace.Camera_descriptors,
				};
				// std::cout << "Camera: " << workspace.Camera_descriptors << std::endl;
				vkCmdBindDescriptorSets(workspace.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, lines_pipeline.layout,
										0, uint32_t(descriptor_sets.size()), descriptor_sets.data(), 0, nullptr);
			}
			// draw lines vertices
			vkCmdDraw(workspace.command_buffer, uint32_t(lines_vertices.size()), 1, 0, 0);
		}

		{ // use object_vertices (offset 0) as vertex buffer binding 0: they all share vertex buffer
			std::array<VkBuffer, 1> vertex_buffers{object_vertices.handle};
			std::array<VkDeviceSize, 1> offsets{0};
			vkCmdBindVertexBuffers(workspace.command_buffer, 0, uint32_t(vertex_buffers.size()), vertex_buffers.data(), offsets.data());
		}

		{
			// draw with the different materialed objects' pipelines
			uint32_t index_offset = 0; // since they all share the same transforms descriptor as well, an offset for indexing the transforms is needed
			if (!lambertian_object_instances.empty())
			{ // draw the lambertian object instances
				vkCmdBindPipeline(workspace.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, lambertian_objects_pipeline.handle);

				{ // bind Lights and Transforms and Shadow descriptor sets:
					std::array<VkDescriptorSet, 3> descriptor_sets{
						workspace.Lights_descriptors,		// 0, Lights
						workspace.Transforms_descriptors,	// 1, Transforms
						workspace.Shadow_Atlas_descriptors, // 2, shadows
					};
					vkCmdBindDescriptorSets(workspace.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, lambertian_objects_pipeline.layout,
											0, uint32_t(descriptor_sets.size()), descriptor_sets.data(), 0, nullptr);
				}

				{ // push constant for lambertian pipeline
					LambertianObjectsPipeline::Push push{
						.light_count = (uint32_t)lights.size(),
					};
					vkCmdPushConstants(workspace.command_buffer, lambertian_objects_pipeline.layout, VK_SHADER_STAGE_FRAGMENT_BIT,
									   0, sizeof(push), &push);
				}
				// draw dat ting
				for (LambertianObjectInstance const &inst : lambertian_object_instances)
				{
					uint32_t index = uint32_t(&inst - &lambertian_object_instances[0]);

					// bind texture descriptor set
					vkCmdBindDescriptorSets(
						workspace.command_buffer,
						VK_PIPELINE_BIND_POINT_GRAPHICS,
						lambertian_objects_pipeline.layout,
						3, 1, &texture_descriptor_sets[inst.texture],
						0, nullptr);
					vkCmdDraw(workspace.command_buffer, inst.vertices.count, 1, inst.vertices.first, index + index_offset);
				}
			}
			index_offset = (uint32_t)lambertian_object_instances.size(); // update index_offset for the next batch of instances

			if (!env_mirror_object_instances.empty())
			{ // draw the env_mirror object instances
				vkCmdBindPipeline(workspace.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, env_mirror_objects_pipeline.handle);
				{ ////bind Lights and Transforms descriptor sets: (shared by env_mirror and pbr objects)
					std::array<VkDescriptorSet, 2> descriptor_sets{
						workspace.Eye_descriptors,		  // 0, Eye
						workspace.Transforms_descriptors, // 1, Transforms
					};

					vkCmdBindDescriptorSets(workspace.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, env_mirror_objects_pipeline.layout,
											0, uint32_t(descriptor_sets.size()), descriptor_sets.data(), 0, nullptr);
				}
				// draw dat ting
				for (EnvMirrorObjectInstance const &inst : env_mirror_object_instances)
				{
					uint32_t index = uint32_t(&inst - &env_mirror_object_instances[0]);

					// bind texture descriptor set
					vkCmdBindDescriptorSets(
						workspace.command_buffer,
						VK_PIPELINE_BIND_POINT_GRAPHICS,
						env_mirror_objects_pipeline.layout,
						2, 1, &texture_descriptor_sets[inst.texture],
						0, nullptr);

					{ // push constant to determine whether it's mirror or environment
						EnvMirrorObjectsPipeline::Push push{
							.is_env = inst.is_env,
							.exposure = (float)rtg.configuration.exposure,
							.tone_map_op = (int32_t)rtg.configuration.tone_map_op,
						};
						vkCmdPushConstants(workspace.command_buffer, env_mirror_objects_pipeline.layout, VK_SHADER_STAGE_FRAGMENT_BIT,
										   0, sizeof(push), &push);
					}
					vkCmdDraw(workspace.command_buffer, inst.vertices.count, 1, inst.vertices.first, index + index_offset);
				}
			}
			index_offset += (uint32_t)env_mirror_object_instances.size(); // update index_offset for the next batch of instances

			if (!pbr_object_instances.empty())
			{ // draw the env_mirror object instances
				vkCmdBindPipeline(workspace.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pbr_objects_pipeline.handle);
				{ ////bind Lights and Transforms descriptor sets: (shared by env_mirror and pbr objects)
					std::array<VkDescriptorSet, 2> descriptor_sets{
						workspace.Eye_descriptors,		  // 0, Eye
						workspace.Transforms_descriptors, // 1, Transforms
					};
					vkCmdBindDescriptorSets(workspace.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pbr_objects_pipeline.layout,
											0, uint32_t(descriptor_sets.size()), descriptor_sets.data(), 0, nullptr);

					vkCmdBindDescriptorSets(workspace.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pbr_objects_pipeline.layout,
											3, 1, &workspace.Lights_descriptors, 0, nullptr);

					vkCmdBindDescriptorSets(workspace.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pbr_objects_pipeline.layout,
											4, 1, &workspace.Shadow_Atlas_descriptors, 0, nullptr);
				}

				{ // push constants for pbr pipeline
					PbrObjectsPipeline::Push push{
						.exposure = (float)rtg.configuration.exposure,
						.tone_map_op = (uint32_t)rtg.configuration.tone_map_op,
						.light_count = (uint32_t)lights.size(),
					};
					vkCmdPushConstants(workspace.command_buffer, pbr_objects_pipeline.layout, VK_SHADER_STAGE_FRAGMENT_BIT,
									   0, sizeof(push), &push);
				}
				// draw dat ting
				for (PbrObjectInstance const &inst : pbr_object_instances)
				{
					uint32_t index = uint32_t(&inst - &pbr_object_instances[0]);
					// bind texture descriptor set
					vkCmdBindDescriptorSets(
						workspace.command_buffer,
						VK_PIPELINE_BIND_POINT_GRAPHICS,
						pbr_objects_pipeline.layout,
						2, 1, &texture_descriptor_sets[inst.texture],
						0, nullptr);

					vkCmdDraw(workspace.command_buffer, inst.vertices.count, 1, inst.vertices.first, index + index_offset);
				}
			}
			index_offset += (uint32_t)pbr_object_instances.size(); // update index_offset for the next batch of instances
		}

		vkCmdEndRenderPass(workspace.command_buffer);
	}



	// End Recording
	VK(vkEndCommandBuffer(workspace.command_buffer));

	{ // submit `workspace.command buffer` for the GPU to run:
		std::array<VkSemaphore, 1> wait_semaphores{
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
	
	if(camera_mode == CameraMode::Debug){ // now read the debug buffer and output it
		vkQueueWaitIdle(rtg.graphics_queue);
		std::vector<char> shadow_map_output;
		shadow_map_output.resize(atlas_size * atlas_size);
		if (workspace.debug_buffer.allocation.mapped == nullptr)
			std::cout << "mapped is fucking null" << std::endl;
		float *floats = reinterpret_cast<float *>(workspace.debug_buffer.allocation.data());
		for (uint32_t i = 0; i < atlas_size * atlas_size; i++)
		{
			shadow_map_output[i] = static_cast<unsigned char>(floats[i] * 255.0f);
		}
		stbi_write_png("debug/shadow_map_debug.png", atlas_size, atlas_size, 1, shadow_map_output.data(), atlas_size);
	}

}

void Tutorial::update(float dt)
{

	// Add at the very beginning of the function
	static auto start_time = std::chrono::high_resolution_clock::now();
	static auto last_update_time = std::chrono::high_resolution_clock::now();
	auto current_time = std::chrono::high_resolution_clock::now();
	auto actual_dt = std::chrono::duration<float>(current_time - last_update_time).count();
	if (rtg.configuration.headless)
	{
		std::cout << "Actual time between updates: " << actual_dt << " seconds (dt parameter: " << dt << ") ";
		std::cout << "Current time: " << std::chrono::duration<float>(current_time - start_time).count() << " seconds " << std::endl;
	}
	last_update_time = current_time;

	time = std::fmod(time + dt, 5.0f);
	time_elapsed += dt;

	//----update scene graph----
	// TODO: if there is no change in any of the nodes, the chunk below shouldn't be run
	lines_vertices.clear();
	lambertian_object_instances.clear();
	env_mirror_object_instances.clear();
	pbr_object_instances.clear();

	lights.clear();
	light_shadow_map_sizes.clear();

	//----Update camera----
	{ // compute CLIP_FROM_WORLD based on what camera mode it is now
		if (camera_mode == CameraMode::Scene)
		{ // unresponsive camera orbiting the origin
			if (current_camera == loaded_cameras.end())
			{ // hard coded scene camera that rotates around target
				float ang = float(M_PI) * 2.0f * (time / 5.0f);
				CLIP_FROM_WORLD = perspective(
									  60.0f * float(M_PI) / 180.0f,									   // vfov
									  rtg.swapchain_extent.width / float(rtg.swapchain_extent.height), // aspect
									  0.1f,															   // near
									  1000.0f														   // far
									  ) *
								  look_at(
									  vec3(13.0f * std::cos(ang), 13.0f * std::sin(ang), 5.0f), // eye
									  vec3(0.0f, 0.0f, 5.0f),									// target
									  vec3(0.0f, 0.0f, 1.0f)									// up
								  );
				EYE = vec3(13.0f * std::cos(ang), 13.0f * std::sin(ang), 5.0f);
			}
			else
			{ // fixed, potentially keyframed camera that is loaded from s72 file
				CLIP_FROM_WORLD = current_camera->second.clip_from_world();
				EYE = current_camera->second.eye;
			}
		}
		else if (camera_mode == CameraMode::Free)
		{
			CLIP_FROM_WORLD = free_camera.clip_from_world(rtg.swapchain_extent.width / float(rtg.swapchain_extent.height)); // aspect passed in
			EYE = free_camera.get_eye();
		}
		else if (camera_mode == CameraMode::Debug)
		{
			CLIP_FROM_WORLD = debug_camera.clip_from_world(rtg.swapchain_extent.width / float(rtg.swapchain_extent.height)); // aspect passed in
			EYE = debug_camera.get_eye();
			// draw frustrum for previous camera
			add_debug_lines_frustrum();
			
			//change debug camera to be spotlight
			lights[0].compute_clip_from_world_sphere();
			CLIP_FROM_WORLD = lights[0].CLIP_FROM_WORLD[0];
			EYE = lights[0].position.xyz();

			// add debug lines for lights
			//add_cuboid_from_corners(lights[0].get_frustum_corners(), 155, 22, 56);
		}
		else
		{
			assert(0 && "only three camera modes");
		}
	}


	// animation drivers, per-frame graph walk updating changes in transforms
	update_scene(dt);


	if (default_world_lights)
	{ // moving sun and sky:
		float cycle = (sin(6.28f * time / 5.0f) + 0.8f) / 1.8f;
		vec4 dir;
		dir.x = 0.0f;
		dir.y = 0.0f;
		dir.z = sin(6.28f * time / 5.0f) - 0.3f;
		dir.w = 0.0f;

		vec3 color;
		color.x = sin(6.28f * time / 5.0f) - 0.3f;
		color.y = sin(6.28f * time / 5.0f) - 0.3f;
		color.z = 0.2f;

		lights.emplace_back(Light{
			.color = vec4(color.x, color.y, color.z, 0),
			.direction = dir,
			.type = 0, // 0 indicates SUN
			.angle = 0,
			.strength = 1.0f,
		});

		vec4 sun_dir;
		sun_dir.x = 0.0f;
		sun_dir.y = sin(6.28f * time / 5.0f);
		sun_dir.z = cos(6.28f * time / 5.0f) - 0.3f;
		sun_dir.w = 0.0f;

		vec4 sun_color;
		sun_color.x = cycle;
		sun_color.y = cycle;
		sun_color.z = cycle;
		lights.emplace_back(Light{
			.color = vec4(sun_color.x, sun_color.y, sun_color.z, 0),
			.direction = sun_dir,
			.type = 0, // 0 indicates SUN
			.angle = 0,
			.strength = 1.0f,
		});
	}

	{ // lines stuff
	  //  if(starts.size() > 64){
	  //  	std::cout<<"too many nodes"<<std::endl;
	  //  	object_instances.clear();
	  //  	starts.clear();
	  //  	lines_vertices.clear();
	  //  	iters = 0;
	  //  }
	  //  if(lines_vertices.size() > 1024){
	  //  	std::cout<<"too many vertices"<<std::endl;
	  //  	object_instances.clear();
	  //  	starts.clear();
	  //  	lines_vertices.clear();
	  //  	iters = 0;
	  //  }

		//----tree stuff----
		// if(starts.empty())starts.emplace_back(vec3(0.0f,0.0f,0.0f));
		// if(time_elapsed > 0.5f && growing){
		// 	size_t num_nodes = starts.size();

		// 	auto verts = meshes.begin();//go through the mesh vertcies (like with poker cards) to grow them on trees
		// 	for(size_t i = 0; i < num_nodes; ++i){
		// 		vec3 cur_node = starts[i];
		// 		starts.emplace_back(emplace_random_line(cur_node,iters));

		// 		if(dist(engine) > 0.2f){
		// 			vec3 fruit_node = emplace_random_line(cur_node,iters);
		// 			//make fruits
		// 			if(dist(engine) > 1.0f-(iters-5) * 0.07f && iters>5){

		// 				{//spiky ball shrunken by a factor

		// 					float scaling_factor = 0.5f;
		// 					mat4 WORLD_FROM_LOCAL{
		// 						scaling_factor, 0.0f,  0.0f, 0.0f,
		// 						0.0f,scaling_factor, 0.0f, 0.0f,
		// 						0.0f, 0.0f,   scaling_factor, 0.0f,
		// 						fruit_node.x,fruit_node.y,fruit_node.z, 1.0f,
		// 					};
		// 					object_instances.emplace_back(ObjectInstance{
		// 						.vertices = fruit_vertices,//which vertices to use
		// 						.transform{
		// 							.CLIP_FROM_LOCAL = CLIP_FROM_WORLD * WORLD_FROM_LOCAL,
		// 							.WORLD_FROM_LOCAL = WORLD_FROM_LOCAL,
		// 							.WORLD_FROM_LOCAL_NORMAL = WORLD_FROM_LOCAL,
		// 						},
		// 						.texture = 1,
		// 					});
		// 				}
		// 				verts++;
		// 				if(verts == meshes.end())verts = meshes.begin();
		// 			}
		// 			//else branch
		// 			else starts.emplace_back(fruit_node);
		// 		}
		// 	}
		// 	// Remove the old nodes (keep only the new branches)
		// 	starts.erase(starts.begin(), starts.begin() + num_nodes);
		// 	time_elapsed = 0.0f;
		// 	iters++;
		// }
	}

	{ // make some objects

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

// helper functions

vec3 Tutorial::emplace_random_line(vec3 start, uint32_t iter)
{
	// do some approximation of tree growing based on current iteration number and height

	float length_modifier = powf(0.9f, (float)iter); // length gets smaller

	float up_modifier = powf(0.3f, (float)iter); // tree grows less and less "up"

	vec3 growth = vec3(dist(engine), dist(engine), up_modifier + std::abs(dist(engine)));

	vec3 new_location = start + length_modifier * normalized(growth);
	float color_key = (dist(engine) + 1.0f) / 2;
	uint8_t r = (static_cast<uint8_t>(std::floor(color_key * 256.0f)));
	uint8_t g = (static_cast<uint8_t>(std::floor(color_key * 486.0f * color_key)));
	uint8_t b = (static_cast<uint8_t>(std::floor(color_key * 238.0f)));
	uint8_t a = 1;
	lines_vertices.emplace_back(PosColVertex{
		.Position = start,
		.Color{.r = r, .g = g, .b = b, .a = a},
	});
	lines_vertices.emplace_back(PosColVertex{
		.Position = new_location,
	});
	return new_location;
}

void Tutorial::on_input(InputEvent const &evt)
{
	// if there is a current action, it gets input priority:
	if (action)
	{
		action(evt);
		return;
	}
	// general controls

	if (evt.type == InputEvent::KeyDown && evt.key.key == GLFW_KEY_M)
	{
		shadows_on = !shadows_on;
		std::cout << "Switching shadows " << shadows_on << std::endl;
	}
	if (evt.type == InputEvent::KeyDown && evt.key.key == GLFW_KEY_TAB)
	{
		// switch between 2 camera modes
		if (camera_mode != CameraMode::Debug)
			camera_mode = CameraMode((int(camera_mode) + 1) % 2);
		else if (camera_mode == CameraMode::Debug)
			prev_camera_mode = CameraMode((int(prev_camera_mode) + 1) % 2);
		return;
	}
	if (evt.type == InputEvent::KeyDown && evt.key.key == GLFW_KEY_D)
	{
		// go between debug and not debug
		CameraMode temp_current_mode = camera_mode;
		camera_mode = prev_camera_mode;
		prev_camera_mode = temp_current_mode;
		return;
	}
	if (evt.type == InputEvent::KeyDown && evt.key.key == GLFW_KEY_S)
	{
		std::cout << loaded_cameras.size() << "previous camera: " << current_camera->first << std::endl;
		current_camera++;
		if (current_camera == loaded_cameras.end())
		{
			current_camera = loaded_cameras.begin();
		}
		std::cout << loaded_cameras.size() << "current camera: " << current_camera->first << std::endl;
	}
	if (evt.type == InputEvent::KeyDown && evt.key.key == GLFW_KEY_G)
	{
		// toggle growing
		growing = !growing;
		return;
	}

	// free camera controls
	if (camera_mode == CameraMode::Free)
	{
		if (evt.type == InputEvent::MouseWheel)
		{
			// zoom in/out
			free_camera.radius *= std::pow(1.1f, -evt.wheel.y);
			free_camera.radius = std::min(free_camera.radius, free_camera.far);
			free_camera.radius = std::max(free_camera.radius, free_camera.near);
		}
		if (evt.type == InputEvent::MouseButtonDown && evt.button.button == GLFW_MOUSE_BUTTON_LEFT && !(evt.button.mods & GLFW_MOD_SHIFT))
		{
			// start tumbling
			float init_x = evt.button.x;
			float init_y = evt.button.y;
			OrbitCamera init_camera = free_camera;

			action = [this, init_x, init_y, init_camera](InputEvent const &evt)
			{
				if (evt.type == InputEvent::MouseButtonUp && evt.button.button == GLFW_MOUSE_BUTTON_LEFT)
				{
					// cancel upon button lifted:
					action = nullptr;
					return;
				}
				if (evt.type == InputEvent::MouseMotion)
				{
					// handle panning
					float dx = (evt.motion.x - init_x) / rtg.swapchain_extent.height;
					float dy = -(evt.motion.y - init_y) / rtg.swapchain_extent.height; // note: negated because glfw uses y-down coordinate system

					// rotate camera based on motion:
					float speed = float(M_PI);
					float flip_x = (std::abs(init_camera.elevation) > 0.5f * float(M_PI) ? -1.0f : 1.0f); // switch azimuth rotation when camera is upside-down
					free_camera.azimuth = init_camera.azimuth - dx * speed * flip_x;
					free_camera.elevation = init_camera.elevation - dy * speed;

					// reduce azimuth and elevation to [-pi,pi] range:
					const float twopi = 2.0f * float(M_PI);
					free_camera.elevation -= std::round(free_camera.elevation / twopi) * twopi;
					free_camera.azimuth -= std::round(free_camera.azimuth / twopi) * twopi;
					return;
				}
			};
			return;
		}
		if (evt.type == InputEvent::MouseButtonDown && evt.button.button == GLFW_MOUSE_BUTTON_LEFT && (evt.button.mods & GLFW_MOD_SHIFT))
		{
			// start panning
			float init_x = evt.button.x;
			float init_y = evt.button.y;
			OrbitCamera init_camera = free_camera;
			// handle panning
			action = [this, init_x, init_y, init_camera](InputEvent const &evt)
			{
				if (evt.type == InputEvent::MouseButtonUp && evt.button.button == GLFW_MOUSE_BUTTON_LEFT)
				{
					// cancel upon button lifted:
					action = nullptr;
					return;
				}
				if (evt.type == InputEvent::MouseMotion)
				{
					// handle panning
					float height = 2.0f * std::tan(free_camera.fov * 0.5f) * free_camera.radius;
					// multiplying dx and dy by height because farther camera should move more so that stuff should glide across screen the same?
					float dx = (evt.motion.x - init_x) / rtg.swapchain_extent.height * height;
					float dy = -(evt.motion.y - init_y) / rtg.swapchain_extent.height * height; // note: negated because glfw uses y-down coordinate system

					// use orbit the extract right and up vectors
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
	else if (camera_mode == CameraMode::Debug)
	{
		if (evt.type == InputEvent::MouseWheel)
		{
			// zoom in/out
			debug_camera.radius *= std::pow(1.1f, -evt.wheel.y);
			debug_camera.radius = std::min(debug_camera.radius, debug_camera.far);
			debug_camera.radius = std::max(debug_camera.radius, debug_camera.near);
		}
		if (evt.type == InputEvent::MouseButtonDown && evt.button.button == GLFW_MOUSE_BUTTON_LEFT && !(evt.button.mods & GLFW_MOD_SHIFT))
		{
			// start tumbling
			float init_x = evt.button.x;
			float init_y = evt.button.y;
			OrbitCamera init_camera = debug_camera;

			action = [this, init_x, init_y, init_camera](InputEvent const &evt)
			{
				if (evt.type == InputEvent::MouseButtonUp && evt.button.button == GLFW_MOUSE_BUTTON_LEFT)
				{
					// cancel upon button lifted:
					action = nullptr;
					return;
				}
				if (evt.type == InputEvent::MouseMotion)
				{
					// handle panning
					float dx = (evt.motion.x - init_x) / rtg.swapchain_extent.height;
					float dy = -(evt.motion.y - init_y) / rtg.swapchain_extent.height; // note: negated because glfw uses y-down coordinate system

					// rotate camera based on motion:
					float speed = float(M_PI);
					float flip_x = (std::abs(init_camera.elevation) > 0.5f * float(M_PI) ? -1.0f : 1.0f); // switch azimuth rotation when camera is upside-down
					debug_camera.azimuth = init_camera.azimuth - dx * speed * flip_x;
					debug_camera.elevation = init_camera.elevation - dy * speed;

					// reduce azimuth and elevation to [-pi,pi] range:
					const float twopi = 2.0f * float(M_PI);
					debug_camera.elevation -= std::round(debug_camera.elevation / twopi) * twopi;
					debug_camera.azimuth -= std::round(debug_camera.azimuth / twopi) * twopi;
					return;
				}
			};
			return;
		}
		if (evt.type == InputEvent::MouseButtonDown && evt.button.button == GLFW_MOUSE_BUTTON_LEFT && (evt.button.mods & GLFW_MOD_SHIFT))
		{
			// start panning
			float init_x = evt.button.x;
			float init_y = evt.button.y;
			OrbitCamera init_camera = debug_camera;
			// handle panning
			action = [this, init_x, init_y, init_camera](InputEvent const &evt)
			{
				if (evt.type == InputEvent::MouseButtonUp && evt.button.button == GLFW_MOUSE_BUTTON_LEFT)
				{
					// cancel upon button lifted:
					action = nullptr;
					return;
				}
				if (evt.type == InputEvent::MouseMotion)
				{
					// handle panning
					float height = 2.0f * std::tan(debug_camera.fov * 0.5f) * debug_camera.radius;
					// multiplying dx and dy by height because farther camera should move more so that stuff should glide across screen the same?
					float dx = (evt.motion.x - init_x) / rtg.swapchain_extent.height * height;
					float dy = -(evt.motion.y - init_y) / rtg.swapchain_extent.height * height; // note: negated because glfw uses y-down coordinate system

					// use orbit the extract right and up vectors
					mat4 camera_from_world = orbit(init_camera.target, init_camera.azimuth, init_camera.elevation, init_camera.radius);
					vec3 right = vec3(camera_from_world[0], camera_from_world[4], camera_from_world[8]);
					vec3 up = vec3(camera_from_world[1], camera_from_world[5], camera_from_world[9]);
					debug_camera.target = init_camera.target - dx * right - dy * up;
					return;
				}
			};
			return;
		}
	}
}

void Tutorial::add_debug_lines_frustrum()
{
	//----frustrum drawing
	std::array<vec3, 8> frustrum_corners; // get the corners
	vec3 color;
	if (prev_camera_mode == CameraMode::Scene)
	{
		frustrum_corners = current_camera->second.get_frustum_corners();
		color = vec3{0.0, 0.0, 1.0};
	}
	else if (prev_camera_mode == CameraMode::Free)
	{
		frustrum_corners = free_camera.get_frustum_corners(rtg.swapchain_extent.width / float(rtg.swapchain_extent.height)); // aspect passed in
		color = vec3{0.0, 1.0, 0.0};
	}
	else
	{
		assert(0 && "prev can't also be debug");
	}
	// Order: near plane (bottom-left, bottom-right, top-right, top-left),
	//        far plane (bottom-left, bottom-right, top-right, top-left)
	uint8_t r = uint8_t(std::round(256 * color.x));
	uint8_t g = uint8_t(std::round(256 * color.y));
	uint8_t b = uint8_t(std::round(256 * color.z));

	add_cuboid_from_corners(frustrum_corners, r, g, b);
}

void Tutorial::add_cuboid_from_corners(std::array<vec3, 8> const &box_corners, uint8_t r, uint8_t g, uint8_t b)
{
	auto emplace_line = [&](uint32_t from, uint32_t to)
	{
		lines_vertices.emplace_back(PosColVertex{.Position = box_corners[from], .Color = {.r = r, .g = g, .b = b, .a = 255}});
		lines_vertices.emplace_back(PosColVertex{.Position = box_corners[to], .Color = {.r = r, .g = g, .b = b, .a = 255}});
	};
	// near face
	emplace_line(0, 1);
	emplace_line(1, 2);
	emplace_line(2, 3);
	emplace_line(3, 0);
	// far face
	emplace_line(4, 5);
	emplace_line(5, 6);
	emplace_line(6, 7);
	emplace_line(7, 4);
	// connections
	emplace_line(0, 4);
	emplace_line(1, 5);
	emplace_line(2, 6);
	emplace_line(3, 7);
}

void Tutorial::add_debug_lines_bbox(AABB &bbox, mat4 WORLD_FROM_LOCAL)
{
	// local corners
	std::array<vec3, 8> bbox_corners;
	bbox.get_box_corners(WORLD_FROM_LOCAL, bbox_corners);

	add_cuboid_from_corners(bbox_corners, 0, 255, 0);
}

std::array<vec3, 8> Tutorial::BasicCamera::get_frustum_corners() const
{
	std::array<vec3, 8> corners;

	// Calculate frustum dimensions at near plane
	float near_height = 2.0f * near * std::tan(vfov * 0.5f);
	float near_width = near_height * aspect;

	// Calculate frustum dimensions at far plane
	float far_height = 2.0f * far * std::tan(vfov * 0.5f);
	float far_width = far_height * aspect;

	// Build camera basis vectors
	vec3 forward = normalized(dir);
	vec3 right = normalized(cross(forward, up));
	vec3 camera_up = cross(right, forward);

	// Near plane center and far plane center
	vec3 near_center = eye + forward * near;
	vec3 far_center = eye + forward * far;

	// Near plane corners
	corners[0] = near_center - right * (near_width * 0.5f) - camera_up * (near_height * 0.5f); // bottom-left
	corners[1] = near_center + right * (near_width * 0.5f) - camera_up * (near_height * 0.5f); // bottom-right
	corners[2] = near_center + right * (near_width * 0.5f) + camera_up * (near_height * 0.5f); // top-right
	corners[3] = near_center - right * (near_width * 0.5f) + camera_up * (near_height * 0.5f); // top-left

	// Far plane corners
	corners[4] = far_center - right * (far_width * 0.5f) - camera_up * (far_height * 0.5f); // bottom-left
	corners[5] = far_center + right * (far_width * 0.5f) - camera_up * (far_height * 0.5f); // bottom-right
	corners[6] = far_center + right * (far_width * 0.5f) + camera_up * (far_height * 0.5f); // top-right
	corners[7] = far_center - right * (far_width * 0.5f) + camera_up * (far_height * 0.5f); // top-left

	return corners;
}

vec3 Tutorial::OrbitCamera::get_eye() const
{
	// Calculate camera position and orientation from orbit camera parameters
	float cos_elev = std::cos(elevation);
	float sin_elev = std::sin(elevation);
	float cos_azim = std::cos(azimuth);
	float sin_azim = std::sin(azimuth);

	vec3 eye = target + vec3{
							radius * cos_elev * cos_azim,
							radius * cos_elev * sin_azim,
							radius * sin_elev};
	return eye;
}




std::array<vec3, 8> Tutorial::Light::get_frustum_corners() const
{ // only use this on spotlights, where a frustum is defined
	assert(type == 2);
	std::array<vec3, 8> corners;

	vec3 eye = position.xyz();

	// Camera basis vectors
	vec3 forward = normalized(direction.xyz());
	vec3 world_up = vec3{0.0f, 0.0f, 1.0f};
	vec3 right = normalized(cross(forward, world_up));
	vec3 up = cross(right, forward);

	float near = 0.1f;
	float far = std::min(limit,1000.0f);
	float aspect = 1.0f;

	// Calculate frustum dimensions
	float near_height = 2.0f * near * std::tan(fov * 0.5f);
	float near_width = near_height * aspect;
	float far_height = 2.0f * far * std::tan(fov * 0.5f);
	float far_width = far_height * aspect;

	// Centers of near and far planes
	vec3 near_center = eye + forward * near;
	vec3 far_center = eye + forward * far;

	// Near plane corners
	corners[0] = near_center - right * (near_width * 0.5f) - up * (near_height * 0.5f);
	corners[1] = near_center + right * (near_width * 0.5f) - up * (near_height * 0.5f);
	corners[2] = near_center + right * (near_width * 0.5f) + up * (near_height * 0.5f);
	corners[3] = near_center - right * (near_width * 0.5f) + up * (near_height * 0.5f);

	// Far plane corners
	corners[4] = far_center - right * (far_width * 0.5f) - up * (far_height * 0.5f);
	corners[5] = far_center + right * (far_width * 0.5f) - up * (far_height * 0.5f);
	corners[6] = far_center + right * (far_width * 0.5f) + up * (far_height * 0.5f);
	corners[7] = far_center - right * (far_width * 0.5f) + up * (far_height * 0.5f);

	return corners;
}

std::array<vec3, 8> Tutorial::OrbitCamera::get_frustum_corners(float aspect) const
{
	std::array<vec3, 8> corners;

	vec3 eye = get_eye();

	// Camera basis vectors
	vec3 forward = normalized(target - eye);
	vec3 world_up = vec3{0.0f, 0.0f, 1.0f};
	vec3 right = normalized(cross(forward, world_up));
	vec3 up = cross(right, forward);

	// Calculate frustum dimensions
	float near_height = 2.0f * near * std::tan(fov * 0.5f);
	float near_width = near_height * aspect;
	float far_height = 2.0f * far * std::tan(fov * 0.5f);
	float far_width = far_height * aspect;

	// Centers of near and far planes
	vec3 near_center = eye + forward * near;
	vec3 far_center = eye + forward * far;

	// Near plane corners
	corners[0] = near_center - right * (near_width * 0.5f) - up * (near_height * 0.5f);
	corners[1] = near_center + right * (near_width * 0.5f) - up * (near_height * 0.5f);
	corners[2] = near_center + right * (near_width * 0.5f) + up * (near_height * 0.5f);
	corners[3] = near_center - right * (near_width * 0.5f) + up * (near_height * 0.5f);

	// Far plane corners
	corners[4] = far_center - right * (far_width * 0.5f) - up * (far_height * 0.5f);
	corners[5] = far_center + right * (far_width * 0.5f) - up * (far_height * 0.5f);
	corners[6] = far_center + right * (far_width * 0.5f) + up * (far_height * 0.5f);
	corners[7] = far_center - right * (far_width * 0.5f) + up * (far_height * 0.5f);

	return corners;
}

bool Tutorial::do_cull(std::array<vec3, 8> const &frustrum_corners, std::array<vec3, 8> const &box_corners)
{
	auto test_axis = [&](vec3 const &dir) -> bool
	{
		// plot all points on a number line
		std::array<float, 8> box_points;
		std::array<float, 8> frustrum_points;
		for (uint32_t i = 0; i < 8; i++)
		{
			box_points[i] = dot(box_corners[i], dir);
			frustrum_points[i] = dot(frustrum_corners[i], dir);
		}
		// check for overlap
		std::sort(box_points.begin(), box_points.end());
		std::sort(frustrum_points.begin(), frustrum_points.end());

		if (box_points[7] < frustrum_points[0] || box_points[0] > frustrum_points[7])
		{
			return true;
		}
		return false;
	};

	//------go through all the faces' Separating axes
	vec3 box_axes[3] = {
		box_corners[1] - box_corners[0], // right edge (x-axis)
		box_corners[3] - box_corners[0], // up edge (y-axis)
		box_corners[4] - box_corners[0], // forward edge (z-axis)
	};
	{ // Test box face normals (3 axes)
		for (uint32_t i = 0; i < 3; i++)
		{
			if (test_axis(box_axes[i]))
				return true;
		}
	}
	//------Test frustum face normals (6 faces)
	vec3 frustrum_edges[6] = {
		frustrum_corners[1] - frustrum_corners[0], // near face: bottom, right
		frustrum_corners[2] - frustrum_corners[1],
		frustrum_corners[4] - frustrum_corners[0], // bottom left projecting edge
		frustrum_corners[6] - frustrum_corners[2], // top right projecting edge
		frustrum_corners[5] - frustrum_corners[1], // bottom right projecting edge
		frustrum_corners[7] - frustrum_corners[3], // top left projecting edges
	};
	{ // Compute face normals
		vec3 frustrum_normals[5] = {
			cross(frustrum_edges[0], frustrum_edges[1]), // near
			cross(frustrum_edges[2], frustrum_edges[1]), // left
			cross(frustrum_edges[3], frustrum_edges[1]), // right
			cross(frustrum_edges[2], frustrum_edges[0]), // top
			cross(frustrum_edges[3], frustrum_edges[0]), // bottom
		};

		for (uint32_t i = 0; i < 5; i++)
		{
			if (test_axis(frustrum_normals[i]))
				return true;
		}
	}
	//------go through all the cross products
	// bbox x axis, y axis, z axis cross...
	for (vec3 const &bedge : box_axes)
	{
		// frustrum near horizontal edge, near vertical edge, four projecting edges
		for (vec3 const &fedge : frustrum_edges)
		{
			if (test_axis(cross(bedge, fedge)))
				return true;
		}
	}
	return false; // none of the SATs work
}

void Tutorial::AABB::get_box_corners(mat4 WORLD_FROM_LOCAL, std::array<vec3, 8> &box_corners)
{
	box_corners[0] = vec3{min.x, min.y, min.z};
	box_corners[1] = vec3{max.x, min.y, min.z};
	box_corners[2] = vec3{max.x, max.y, min.z};
	box_corners[3] = vec3{min.x, max.y, min.z};
	box_corners[4] = vec3{min.x, min.y, max.z};
	box_corners[5] = vec3{max.x, min.y, max.z};
	box_corners[6] = vec3{max.x, max.y, max.z};
	box_corners[7] = vec3{min.x, max.y, max.z};

	auto transform_corner = [&](uint32_t i)
	{
		box_corners[i] = (WORLD_FROM_LOCAL * vec4{box_corners[i].x, box_corners[i].y, box_corners[i].z, 1.0f}).xyz();
	};

	for (uint32_t i = 0; i < 8; i++)
	{
		transform_corner(i);
	}
}
