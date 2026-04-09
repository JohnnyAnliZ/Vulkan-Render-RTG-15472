//this file includes stuff regarding vulkan stuff for running a compute shader separate from the graphics pipeline
#include "Tutorial.hpp"

#include "VK.hpp"

#include <GLFW/glfw3.h>


void Tutorial::init_compute_pipeline(){
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

}


