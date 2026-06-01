//this file includes stuff regarding vulkan stuff for running a compute shader separate from the graphics pipeline
#include "ComputeSystem.hpp"

#include "VK.hpp"
#include <iostream>
#include <GLFW/glfw3.h>


void ComputeSystem::init_compute(RTG &rtg){
    { //create command pool
		VkCommandPoolCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = rtg.graphics_queue_family.value(),
		};
		VK(vkCreateCommandPool(rtg.device, &create_info, nullptr, &compute_cmd_pool));
	}
    
    {//allocate command buffer
        VkCommandBufferAllocateInfo alloc_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,				
            .commandPool = compute_cmd_pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        VK(vkAllocateCommandBuffers(rtg.device, &alloc_info, &compute_cmd_buf));
    }

    {//make a descriptor pool
        //number of fluid 3d texture descriptors

        //2 * 2 for density (read and write for each ping-pong pair) 2 * 2 for velocity, 2 * 2 for pressure, 1 for gradient subtract read/write
		uint32_t fluid_descriptors = 2 * 2 + 2 * 2 + 2 * 2 + 1; 
        std::array<VkDescriptorPoolSize, 1> pool_sizes{
            VkDescriptorPoolSize{
                .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .descriptorCount = fluid_descriptors
            },
        };

        uint32_t fluid_sets = 2 + 2 + 2 + 1;
        //two descriptor sets for velocity, two for density, two for pressure, one for gradient subtract

        uint32_t leaf_descriptors = 1 + 1;//one for leaf buffer, one for heightmap for piling 
        VkDescriptorPoolCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.flags = 0, //because CREATE_FREE_DESCRIPTOR_SET_BIT isn't included, *can't* free individual descriptors allocated from this pool
			.maxSets = fluid_sets + leaf_descriptors,
			.poolSizeCount = uint32_t(pool_sizes.size()),
			.pPoolSizes = pool_sizes.data(),
		};
		std::cout<<"storage descriptor pool has max count"<<fluid_descriptors<<std::endl;
		VK( vkCreateDescriptorPool(rtg.device, &create_info, nullptr, &storage_descriptor_pool) );	
    }
}



void ComputeSystem::destroy(RTG &rtg){
    if(storage_descriptor_pool)
	{
		vkDestroyDescriptorPool(rtg.device, storage_descriptor_pool, nullptr);
		storage_descriptor_pool = nullptr;
	}

	if(compute_cmd_pool != VK_NULL_HANDLE)
	{
		vkDestroyCommandPool(rtg.device, compute_cmd_pool, nullptr);
		compute_cmd_pool = VK_NULL_HANDLE;
	}
}


ComputeContext ComputeSystem::begin_frame(RTG &rtg) {
    vkResetCommandBuffer(compute_cmd_buf, 0);

    VkCommandBufferBeginInfo begin_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };

    vkBeginCommandBuffer(compute_cmd_buf, &begin_info);


    ComputeContext ret{
        .rtg = rtg,
        .storage_descriptor_pool = storage_descriptor_pool,
        .compute_cmd_pool = compute_cmd_pool,
        .compute_cmd_buf = compute_cmd_buf,
    };
    return ret;
}


void ComputeSystem::end_frame(ComputeContext const &ctx) {
    {//submit compute work
        VK(vkEndCommandBuffer(ctx.compute_cmd_buf));

        VkSubmitInfo submit{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &ctx.compute_cmd_buf,
        };

        vkQueueSubmit(ctx.rtg.graphics_queue, 1, &submit, VK_NULL_HANDLE);
        vkQueueWaitIdle(ctx.rtg.graphics_queue);
    }    
}
