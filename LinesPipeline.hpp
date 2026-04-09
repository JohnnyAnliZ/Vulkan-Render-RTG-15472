#pragma once

#include "mat4.hpp"
#include "RTG.hpp"
#include "PosColVertex.hpp"

struct LinesPipeline{

    //descriptor set layouts
    VkDescriptorSetLayout set0_Camera = VK_NULL_HANDLE;
    //types for descriptors
    struct Camera{
        mat4 CLIP_FROM_WORLD;
    };
    static_assert(sizeof(Camera) == 16*4, "camera buffer structure is packed");

    using Vertex = PosColVertex;

    //pipeline layout
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline handle = VK_NULL_HANDLE;
    void create(RTG &,VkRenderPass render_pass, uint32_t subpass);
    void destroy(RTG &);
};