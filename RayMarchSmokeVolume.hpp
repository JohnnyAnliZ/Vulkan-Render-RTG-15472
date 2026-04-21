#pragma once
//pipeline that's a full screen ray march of the smoke volume, with lighting and shadows. This is the main rendering pipeline for the smoke.
#include "mat4.hpp"
#include "RTG.hpp"

struct RayMarchSmokeVolumePipeline{
    //descriptor set layouts

    VkDescriptorSetLayout set0_density_volume;
    
    //push cosntants
    struct Push{
        mat4 WORLD_FROM_CLIP;
        vec4 eye;
        vec4 volume_center;
        uint32_t N;//same for x,y,z
    };

    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline handle = VK_NULL_HANDLE;

    void create(RTG &, VkRenderPass render_pass, uint32_t subpass);
    void destroy(RTG &);
};
