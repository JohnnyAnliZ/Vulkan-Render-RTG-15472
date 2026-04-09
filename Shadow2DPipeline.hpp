#pragma once

#include "mat4.hpp"
#include "RTG.hpp"
#include "PosNorTanTexVertex.hpp"

struct Shadow2DPipeline{
    //descriptor set layouts
    VkDescriptorSetLayout set0_Transforms =VK_NULL_HANDLE;

    //push cosntants
    struct Push{
        mat4 LIGHT_CLIP_FROM_WORLD;
    };

    //pipeline layout
    VkPipelineLayout layout = VK_NULL_HANDLE;
    using Vertex = PosNorTanTexVertex;
    VkPipeline handle = VK_NULL_HANDLE;
    void create(RTG &,VkRenderPass render_pass, uint32_t subpass);
    void destroy(RTG &);
    static uint32_t find_fitting_atlas_size(uint64_t total_shadow_map_size);
    
};
