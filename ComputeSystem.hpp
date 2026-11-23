#pragma once
#include "RTG.hpp"



struct ComputeContext{
    ComputeContext(){};
    ComputeContext(RTG &rtg, VkCommandBuffer cmd, VkCommandPool cmd_pool);
    void destroy(RTG &rtg);

    VkDescriptorPool storage_descriptor_pool;
    VkCommandBuffer compute_cmd_buf; 
    VkCommandPool compute_cmd_pool;
};