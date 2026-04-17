//this file includes stuff regarding vulkan stuff for running a compute shader separate from the graphics pipeline
#include "Tutorial.hpp"

#include "VK.hpp"

#include <GLFW/glfw3.h>


void Tutorial::init_compute(){
    { //create command pool
		VkCommandPoolCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
			.queueFamilyIndex = rtg.graphics_queue_family.value(),
		};
		VK(vkCreateCommandPool(rtg.device, &create_info, nullptr, &compute_command_pool));
	}
    
    {//allocate command buffer
        VkCommandBufferAllocateInfo alloc_info{
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,				
            .commandPool = compute_command_pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        VK(vkAllocateCommandBuffers(rtg.device, &alloc_info, &compute_cmd_buf));
    }

    {//make a descriptor pool
        //number of fluid 3d texture descriptors

        //2 * 2 for velocity (read and write for each ping-pong pair), 2 * 2 for pressure, 1 for gradient subtract read/write
		uint32_t fluid_descriptors = 2 * 2 + 2 * 2 + 1; 
        std::array<VkDescriptorPoolSize, 1> pool_sizes{
            VkDescriptorPoolSize{
                .type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .descriptorCount = fluid_descriptors
            },
        };
        VkDescriptorPoolCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
			.flags = 0, //because CREATE_FREE_DESCRIPTOR_SET_BIT isn't included, *can't* free individual descriptors allocated from this pool
			.maxSets = 2 + 2 + 1,//two descriptor sets for velocity, two for pressure, one for gradient subtract
			.poolSizeCount = uint32_t(pool_sizes.size()),
			.pPoolSizes = pool_sizes.data(),
		};
		std::cout<<"storage descriptor pool has max count"<<fluid_descriptors<<std::endl;
		VK( vkCreateDescriptorPool(rtg.device, &create_info, nullptr, &storage_descriptor_pool) );	
    }
}


