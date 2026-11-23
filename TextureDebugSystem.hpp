#pragma once
#include "RTG.hpp"
#include "TextureDebugPipeline.hpp"
#include "FluidSystem.hpp"
#include "MaterialSystem.hpp"



struct TextureDebugSystem{
    void init_texture_debug(RTG &rtg, MaterialSystem &material_system, FluidSystem & fluid_system);
    void update_texture_debug(RTG &rtg, MaterialSystem &material_system, FluidSystem & fluid_system);

    std::array<VkDescriptorSet,2> velocity_tex;
    std::array<VkDescriptorSet,2> density_tex;
    TextureDebugPipeline texture_debug_pipeline;

};