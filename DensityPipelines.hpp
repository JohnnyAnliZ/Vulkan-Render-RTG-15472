#pragma once

#include "mat4.hpp"
#include "RTG.hpp"

struct AddScalarSourcesPipeline{
    //descriptor set layouts

    VkDescriptorSetLayout set0_density_volume = VK_NULL_HANDLE;;

    //push cosntants
    struct Push{
        uint32_t N;
        float dt;
    };

    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline handle = VK_NULL_HANDLE;

    void create(RTG &);
    void destroy(RTG &);
};

struct DiffuseScalarPipeline{
    //descriptor set layouts

    VkDescriptorSetLayout set0_density_volume = VK_NULL_HANDLE;;

    
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

struct AdvectDensityPipeline{
    //descriptor set layouts

    VkDescriptorSetLayout set0_density_volume = VK_NULL_HANDLE;;
    VkDescriptorSetLayout set1_velocity_volume = VK_NULL_HANDLE;;
    
    //push cosntants
    struct Push{
        uint32_t N;
        float dt;
    };

    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline handle = VK_NULL_HANDLE;

    void create(RTG &);
    void destroy(RTG &);
};