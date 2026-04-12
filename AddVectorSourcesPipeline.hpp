#pragma once

#include "mat4.hpp"
#include "RTG.hpp"

struct AddVectorSourcesPipeline{
    //descriptor set layouts

    VkDescriptorSetLayout set0_velocity_volume;

    
    //push cosntants
    struct Push{
        uint32_t N;
        float dt;
    };

    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline handle = VK_NULL_HANDLE;

    void create(RTG &, VkRenderPass render_pass, uint32_t subpass);
    void destroy(RTG &);
};