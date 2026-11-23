#include "TextureDebugSystem.hpp"


void TextureDebugSystem::init_texture_debug(RTG &rtg, MaterialSystem &material_system, FluidSystem & fluid_system){

    const std::array<VkDescriptorSetLayout, 2> velocity_layouts{
        texture_debug_pipeline.set1_vel_vol,
        texture_debug_pipeline.set1_vel_vol,
    };
    const std::array<VkDescriptorSetLayout, 2> density_layouts{
        texture_debug_pipeline.set2_dens_vol,
        texture_debug_pipeline.set2_dens_vol,
    };
    //allocate the sampled descriptor sets
    {
        VkDescriptorSetAllocateInfo alloc_info_v{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = material_system.texture_descriptor_pool,
            .descriptorSetCount = 2,
            .pSetLayouts = velocity_layouts.data(),
        };
        vkAllocateDescriptorSets(rtg.device, &alloc_info_v, &velocity_tex[0]);

        VkDescriptorSetAllocateInfo alloc_info_d{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = material_system.texture_descriptor_pool,
            .descriptorSetCount = 2,
            .pSetLayouts = density_layouts.data(),
        };
        vkAllocateDescriptorSets(rtg.device, &alloc_info_d, &density_tex[0]);
    }

    {//initialize sampled descriptor sets so they are valid before the first draw
        std::array<VkDescriptorImageInfo, 2> image_infos{
            VkDescriptorImageInfo{ 
                .sampler = material_system.texture_sampler, 
                .imageView = fluid_system.velocity_3D_views[fluid_system.velocity_ind], 
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL },
            VkDescriptorImageInfo{ 
                .sampler = material_system.texture_sampler, 
                .imageView = fluid_system.density_3D_views[fluid_system.density_ind], 
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL },
        };
        for(int i = 0; i < 2; i++){
            std::array<VkWriteDescriptorSet, 2> writes{
                VkWriteDescriptorSet{
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = velocity_tex[i],
                    .dstBinding = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .pImageInfo = &image_infos[0],
                },
                VkWriteDescriptorSet{
                    .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    .dstSet = density_tex[i],
                    .dstBinding = 0,
                    .descriptorCount = 1,
                    .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    .pImageInfo = &image_infos[1],
                },
            };
            vkUpdateDescriptorSets(rtg.device, (uint32_t)writes.size(), writes.data(), 0, nullptr);
        }
    }
}


