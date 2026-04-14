#pragma once

#include "mat4.hpp"
#include "RTG.hpp"

struct DiffuseVectorPipeline{
    //descriptor set layouts

    VkDescriptorSetLayout set0_velocity_volume;

    
    //push cosntants
    struct Push{
        uint32_t N;
        float dt;
        uint32_t color;
    };

    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline handle = VK_NULL_HANDLE;

    void create(RTG &);
    void destroy(RTG &);
};