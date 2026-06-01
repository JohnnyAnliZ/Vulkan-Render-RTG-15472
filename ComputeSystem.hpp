#pragma once
#include "RTG.hpp"


struct ComputeContext {
    RTG &rtg;

    VkDescriptorPool storage_descriptor_pool = VK_NULL_HANDLE;
    VkCommandPool compute_cmd_pool = VK_NULL_HANDLE;
    VkCommandBuffer compute_cmd_buf = VK_NULL_HANDLE;
};

struct ComputeSystem{
    void init_compute(RTG &rtg);
    void destroy(RTG &rtg);

    ComputeContext begin_frame(RTG &rtg);
    void end_frame(ComputeContext const &ctx);


    VkDescriptorPool storage_descriptor_pool = VK_NULL_HANDLE;
    VkCommandPool compute_cmd_pool = VK_NULL_HANDLE;
    VkCommandBuffer compute_cmd_buf = VK_NULL_HANDLE;
};