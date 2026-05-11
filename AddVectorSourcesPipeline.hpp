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
        uint32_t motor_active;
        float motor_radius;
        vec4 cam_pos_grid; // xyz = camera position in grid space
        vec4 cam_dir;      // xyz = camera forward direction
    };

    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline handle = VK_NULL_HANDLE;

    void create(RTG &);
    void destroy(RTG &);
};