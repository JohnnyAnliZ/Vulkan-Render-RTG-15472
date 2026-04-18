#pragma once

#include "mat4.hpp"
#include "RTG.hpp"


struct TextureDebugPipeline{
    //descriptor set layouts
    VkDescriptorSetLayout set0_shadow_atlas = VK_NULL_HANDLE;
    VkDescriptorSetLayout set1_vel_vol = VK_NULL_HANDLE;
    VkDescriptorSetLayout set2_dens_vol = VK_NULL_HANDLE;

    //pipeline layout
    VkPipelineLayout layout = VK_NULL_HANDLE;

    //vertex bindings
    
    VkPipeline handle = VK_NULL_HANDLE;

    void create(RTG &, VkRenderPass render_pass, uint32_t subpass);
    void destroy(RTG &);
};
