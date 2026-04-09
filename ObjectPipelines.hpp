#pragma once

#include "mat4.hpp"
#include "RTG.hpp"
#include "PosNorTanTexVertex.hpp"


struct LambertianObjectsPipeline{
    //descriptor set layouts
    VkDescriptorSetLayout set0_Lights = VK_NULL_HANDLE;
    VkDescriptorSetLayout set1_Transforms = VK_NULL_HANDLE;
    VkDescriptorSetLayout set2_Shadows = VK_NULL_HANDLE;
    VkDescriptorSetLayout set3_Texture = VK_NULL_HANDLE;

    //push constants
    struct Push{
        uint32_t light_count = 0;
    };

    //pipeline layout
    VkPipelineLayout layout = VK_NULL_HANDLE;
    using Vertex = PosNorTanTexVertex;
    VkPipeline handle = VK_NULL_HANDLE;
    void create(RTG &,VkRenderPass render_pass, uint32_t subpass);
    void destroy(RTG &);
};

struct EnvMirrorObjectsPipeline{
    //descriptor set layouts
    VkDescriptorSetLayout set0_Eye= VK_NULL_HANDLE;
    VkDescriptorSetLayout set1_Transforms = VK_NULL_HANDLE;
    VkDescriptorSetLayout set2_TEXTURE = VK_NULL_HANDLE;

    struct Push{
        int32_t is_env = 1;
        float exposure = 0;
        int32_t tone_map_op = 0;
    };

    //pipeline layout
    VkPipelineLayout layout = VK_NULL_HANDLE;
    using Vertex = PosNorTanTexVertex;
    VkPipeline handle = VK_NULL_HANDLE;
    void create(RTG &,VkRenderPass render_pass, uint32_t subpass);
    void destroy(RTG &);
};

struct PbrObjectsPipeline{
    //descriptor set layouts
    VkDescriptorSetLayout set0_Eye= VK_NULL_HANDLE;
    VkDescriptorSetLayout set1_Transforms = VK_NULL_HANDLE;
    VkDescriptorSetLayout set2_TEXTURE = VK_NULL_HANDLE;
    VkDescriptorSetLayout set3_Lights = VK_NULL_HANDLE;
    VkDescriptorSetLayout set4_Shadows = VK_NULL_HANDLE;


    //push constants
    struct Push{
        float exposure = 0;
        uint32_t tone_map_op = 0;//0 is linear, 1 is reinhard
        uint32_t light_count = 0;
    };


    //pipeline layout
    VkPipelineLayout layout = VK_NULL_HANDLE;
    using Vertex = PosNorTanTexVertex;
    VkPipeline handle = VK_NULL_HANDLE;
    void create(RTG &,VkRenderPass render_pass, uint32_t subpass);
    void destroy(RTG &);
};