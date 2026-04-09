#pragma once

#include "mat4.hpp"
#include "RTG.hpp"

struct BackgroundPipeline{
    //descriptor set layouts

    struct Push{
        float time;
    };

    VkPipelineLayout layout = VK_NULL_HANDLE;

    //vertex bindings
    
    VkPipeline handle = VK_NULL_HANDLE;

    void create(RTG &, VkRenderPass render_pass, uint32_t subpass);
    void destroy(RTG &);
};