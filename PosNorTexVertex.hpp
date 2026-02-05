# pragma once

#include <vulkan/vulkan_core.h>

#include <cstdint>

struct PosNorTexVertex{
    struct {float x,y,z;} Position;
    struct {float x,y,z;} Normal;
    struct {float s, t;} TexCoord;
    static const VkPipelineVertexInputStateCreateInfo array_input_state;
};

static_assert(sizeof(PosNorTexVertex) == 3*4 + 4*3 + 2 * 4, "PosNorTexVertex is packed");