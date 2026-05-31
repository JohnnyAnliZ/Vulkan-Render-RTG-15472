#pragma once

#include "RTG.hpp"
#include "Helpers.hpp"

struct Workspace {
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;

    Helpers::AllocatedBuffer lines_vertices_src;
    Helpers::AllocatedBuffer lines_vertices;

    Helpers::AllocatedBuffer Camera_src;
    Helpers::AllocatedBuffer Camera;
    VkDescriptorSet Camera_descriptors = VK_NULL_HANDLE;

    Helpers::AllocatedBuffer Eye_src;
    Helpers::AllocatedBuffer Eye;
    VkDescriptorSet Eye_descriptors = VK_NULL_HANDLE;

    Helpers::AllocatedBuffer Lights_src;
    Helpers::AllocatedBuffer Lights;
    VkDescriptorSet Lights_descriptors = VK_NULL_HANDLE;

    Helpers::AllocatedBuffer Transforms_src;
    Helpers::AllocatedBuffer Transforms;
    VkDescriptorSet Transforms_descriptors = VK_NULL_HANDLE;

    Helpers::AllocatedImage Shadow_Atlas;
    VkImageView Shadow_Atlas_view = VK_NULL_HANDLE;
    VkFramebuffer Shadow_Atlas_FB = VK_NULL_HANDLE;
    VkDescriptorSet Shadow_Atlas_descriptors = VK_NULL_HANDLE;

    Helpers::AllocatedBuffer debug_buffer;
};