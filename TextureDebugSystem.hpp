#pragma once
#include "RTG.hpp"
#include "TextureDebugPipeline.hpp"
#include "FluidSystem.hpp"
#include "MaterialSystem.hpp"



struct TextureDebugSystem{
    void init_texture_debug(RTG &rtg, MaterialSystem &material_system, FluidSystem & fluid_system);
    void update_texture_debug(RTG &rtg, MaterialSystem &material_system, FluidSystem & fluid_system);

    VkDescriptorSet velocity_tex = VK_NULL_HANDLE;
    VkDescriptorSet density_tex = VK_NULL_HANDLE;
    TextureDebugPipeline texture_debug_pipeline;

};