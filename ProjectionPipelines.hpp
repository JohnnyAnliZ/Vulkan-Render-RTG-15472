#pragma once

#include "mat4.hpp"
#include "RTG.hpp"

struct DivergencePipeline{
    //descriptor set layouts

    VkDescriptorSetLayout set0_velocity_volume;
    VkDescriptorSetLayout set1_pressure_volume;
    VkDescriptorSetLayout set2_divergence_volume;
    
    //push cosntants
    struct Push{
        uint32_t N;
    };

    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline handle = VK_NULL_HANDLE;

    void create(RTG &);
    void destroy(RTG &);
};

struct PressureSolvePipeline{
    //descriptor set layouts
    VkDescriptorSetLayout set0_divergence_volume;
    VkDescriptorSetLayout set1_pressure_volume;
    
    
    //push cosntants
    struct Push{
        uint32_t N;
    };

    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline handle = VK_NULL_HANDLE;

    void create(RTG &);
    void destroy(RTG &);
};

struct GradientSubtractPipeline{
    //descriptor set layouts
    VkDescriptorSetLayout set0_velocity_volume;
    VkDescriptorSetLayout set1_pressure_volume;
    
    
    //push cosntants
    struct Push{
        uint32_t N;
    };

    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline handle = VK_NULL_HANDLE;

    void create(RTG &);
    void destroy(RTG &);
};