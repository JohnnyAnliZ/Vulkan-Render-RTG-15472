# pragma once

#include <vulkan/vulkan_core.h>

#include <cstdint>
#include "mat4.hpp"

struct PosColVertex{
    vec3 Position;
    struct {uint8_t r,g,b,a;} Color;
    static const VkPipelineVertexInputStateCreateInfo array_input_state;
};

static_assert(sizeof(PosColVertex) == 3*4 + 4*1, "PosColVertex is packer");