#include "FluidSystem.hpp"
#include "Helpers.hpp"
#include "VK.hpp"

//this function make sure the written image is available for reading in the next dispatch, by inserting a memory barrier for the written image
void FluidSystem::ping_pong_barrier(VkCommandBuffer cmd, VkImage &img){
    VkImageMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .image = img, // written image
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1,
            .layerCount = 1
        }
    };
    vkCmdPipelineBarrier(
        cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );
}

void FluidSystem::make_ping_pong_descriptor_sets(
    RTG &rtg,
    ComputeContext comp_context,
    VkDescriptorSet *pressure_sets,
    VkDescriptorSetLayout const &layout,
    VkImageView *image_views
){
    
    // Allocate both sets at once
    std::array<VkDescriptorSetLayout,2> layouts{layout, layout};
    VkDescriptorSetAllocateInfo alloc_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = comp_context.storage_descriptor_pool,
        .descriptorSetCount = (uint32_t)layouts.size(),
        .pSetLayouts = layouts.data(), // same layout twice
    };

    VK(vkAllocateDescriptorSets(rtg.device, &alloc_info, pressure_sets));

    // --- Descriptor setup ---
    for (uint32_t i = 0; i < 2; i++) {

        uint32_t read  = i;
        uint32_t write = 1 - i;

        std::array<VkDescriptorImageInfo, 2> infos{
            VkDescriptorImageInfo{
                .sampler = volume_sampler,
                .imageView = image_views[read],
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL
            },
            VkDescriptorImageInfo{
                .sampler = volume_sampler,
                .imageView = image_views[write],
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL
            }
        };

        std::array<VkWriteDescriptorSet, 2> writes;

        for (uint32_t b = 0; b < 2; b++) {
            writes[b] = VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = pressure_sets[i],
                .dstBinding = b,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .pImageInfo = &infos[b],
            };
        }

        vkUpdateDescriptorSets(rtg.device, 2, writes.data(), 0, nullptr);
    }
}



void FluidSystem::add_sources_density(float dt, ComputeContext comp_context){
    vkCmdBindPipeline(
        comp_context.compute_cmd_buf,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        add_scalar_sources_pipeline.handle
    );

    vkCmdBindDescriptorSets(
        comp_context.compute_cmd_buf,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        add_scalar_sources_pipeline.layout,
        0,
        1,
        &density_volumes[density_ind],
        0,
        nullptr
    );

    AddScalarSourcesPipeline::Push push{
        .N = v_volume_side_length,
        .dt = dt,
    };
    vkCmdPushConstants(comp_context.compute_cmd_buf, add_scalar_sources_pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    vkCmdDispatch(
        comp_context.compute_cmd_buf,
        v_volume_side_length/groupCounts[0],
        v_volume_side_length/groupCounts[1],
        v_volume_side_length/groupCounts[2]
    );  

    //memory barrier
    ping_pong_barrier(comp_context.compute_cmd_buf, density_3D_textures[1 - density_ind].handle);

    //swap
    density_ind = 1 - density_ind;
}


void FluidSystem::diffuse_density(float dt, ComputeContext comp_context){
    const uint32_t ITERS = 5;
    vkCmdBindPipeline(comp_context.compute_cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, diffuse_scalar_pipeline.handle);
    for(uint32_t i = 0; i < ITERS; i++){//red-black gauss-seidel method, red pass fills half buffer and the black pass reads from that 
        //red pass
        DiffuseScalarPipeline::Push push_red{
            .N = v_volume_side_length,
            .dt = dt,
            .color = 0, 
        };
        vkCmdPushConstants(comp_context.compute_cmd_buf, diffuse_scalar_pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_red), &push_red);
        vkCmdBindDescriptorSets(
            comp_context.compute_cmd_buf,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            diffuse_scalar_pipeline.layout,
            0,
            1,
            &density_volumes[density_ind],
            0,
            nullptr
        );
        vkCmdDispatch(comp_context.compute_cmd_buf,
            v_volume_side_length/groupCounts[0],
            v_volume_side_length/groupCounts[1],
            v_volume_side_length/groupCounts[2]
        );

        //memory barrier
        ping_pong_barrier(comp_context.compute_cmd_buf, density_3D_textures[1 - density_ind].handle);
        //swap
        density_ind = 1 - density_ind;

        //black pass
        DiffuseScalarPipeline::Push push_black{
            .N = v_volume_side_length,
            .dt = dt,
            .color = 1, 
        };
        vkCmdPushConstants(comp_context.compute_cmd_buf, diffuse_scalar_pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_black), &push_black);
        vkCmdBindDescriptorSets(
            comp_context.compute_cmd_buf,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            diffuse_scalar_pipeline.layout,
            0,
            1,
            &density_volumes[density_ind],
            0,
            nullptr
        );
        vkCmdDispatch(comp_context.compute_cmd_buf,
            v_volume_side_length/groupCounts[0],
            v_volume_side_length/groupCounts[1],
            v_volume_side_length/groupCounts[2]
        );

        //memory barrier
        ping_pong_barrier(comp_context.compute_cmd_buf, density_3D_textures[1 - density_ind].handle);
        //swap
        density_ind = 1 - density_ind;
    }
}

void FluidSystem::advect_density(float dt, ComputeContext comp_context){
    vkCmdBindPipeline(comp_context.compute_cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, advect_density_pipeline.handle);
    AdvectDensityPipeline::Push push{
        .N = v_volume_side_length,
        .dt = dt,
    };
    vkCmdPushConstants(comp_context.compute_cmd_buf, advect_density_pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    std::array<VkDescriptorSet, 2> sets{
        density_volumes[density_ind],//read_write
        velocity_volumes[velocity_ind],//read
    };
    vkCmdBindDescriptorSets(
        comp_context.compute_cmd_buf,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        advect_density_pipeline.layout,
        0,
        (uint32_t)sets.size(),
        sets.data(),
        0,
        nullptr
    );
    vkCmdDispatch(comp_context.compute_cmd_buf,
        v_volume_side_length/groupCounts[0],
        v_volume_side_length/groupCounts[1],
        v_volume_side_length/groupCounts[2]
    );
    //memory barrier
    ping_pong_barrier(comp_context.compute_cmd_buf, density_3D_textures[1 - density_ind].handle);
    //swap
    density_ind = 1 - density_ind;
}

void FluidSystem::add_sources_velocity(float dt, ComputeContext comp_context, SceneSystem &scene_system){
    vkCmdBindPipeline(
        comp_context.compute_cmd_buf,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        add_vector_sources_pipeline.handle
    );

    vkCmdBindDescriptorSets(
        comp_context.compute_cmd_buf,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        add_vector_sources_pipeline.layout,
        0,
        1,
        &velocity_volumes[velocity_ind],
        0,
        nullptr
    );

    vec3 volume_center = vec3(0.0f, 0.0f, 10.0f);
    float volume_size  = cell_size_ws;
    vec3 volume_min    = volume_center - vec3(volume_size * 0.5f);
    vec3 cam_grid      = (scene_system.EYE - volume_min) / volume_size * float(v_volume_side_length);

    AddVectorSourcesPipeline::Push push{
        .N = v_volume_side_length,
        .dt = dt,
        .motor_active = scene_system.wind_motor_active ? 1u : 0u,
        .motor_radius = float(v_volume_side_length) * 0.08f,
        .cam_pos_grid = vec4(cam_grid.x, cam_grid.y, cam_grid.z, 0.0f),
        .cam_dir      = vec4(scene_system.CAM_DIR.x, scene_system.CAM_DIR.y, scene_system.CAM_DIR.z, 0.0f),
    };
    vkCmdPushConstants(comp_context.compute_cmd_buf, add_vector_sources_pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    vkCmdDispatch(  
        comp_context.compute_cmd_buf,
        v_volume_side_length/groupCounts[0],
        v_volume_side_length/groupCounts[1],
        v_volume_side_length/groupCounts[2]
    );  

    //memory barrier
    ping_pong_barrier(comp_context.compute_cmd_buf, velocity_3D_textures[1 - velocity_ind].handle);

    //swap
    velocity_ind = 1 - velocity_ind;
}


void FluidSystem::project_velocity(ComputeContext comp_context){
    //Phase 1: calculate divergence and clear pressure
    vkCmdBindPipeline(comp_context.compute_cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, divergence_pipeline.handle);
    DivergencePipeline::Push push_div{
        .N = v_volume_side_length,
    };
    vkCmdPushConstants(comp_context.compute_cmd_buf, divergence_pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_div), &push_div);
    std::array<VkDescriptorSet, 3> sets{
        velocity_volumes[velocity_ind],//read
        pressure_volumes[pressure_ind],//read
        divergence_volume,//write
    };
    vkCmdBindDescriptorSets(
        comp_context.compute_cmd_buf,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        divergence_pipeline.layout,
        0,
        (uint32_t)sets.size(),
        sets.data(),
        0,
        nullptr
    );
    vkCmdDispatch(comp_context.compute_cmd_buf,
        v_volume_side_length/groupCounts[0],
        v_volume_side_length/groupCounts[1],
        v_volume_side_length/groupCounts[2]
    );
    //memory barrier for divergence volume
    VkImageMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .image = divergence_3D_texture.handle, // written image
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1,
            .layerCount = 1
        }
    };

    //Phase 2: pressure solve (jacobi iterations)
    vkCmdBindPipeline(comp_context.compute_cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pressure_solve_pipeline.handle);
    PressureSolvePipeline::Push push_ps{
        .N = v_volume_side_length,
    };
    vkCmdPushConstants(comp_context.compute_cmd_buf, pressure_solve_pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_ps), &push_ps);
    vkCmdBindDescriptorSets(comp_context.compute_cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pressure_solve_pipeline.layout, 0, 1, &divergence_volume, 0, nullptr);
    const uint32_t ITERS = 10;
    for(uint32_t i = 0; i < ITERS; i++){//jacobi    
        vkCmdBindDescriptorSets(comp_context.compute_cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pressure_solve_pipeline.layout, 1, 1, &pressure_volumes[pressure_ind], 0, nullptr);
        vkCmdDispatch(comp_context.compute_cmd_buf,
            v_volume_side_length/groupCounts[0],
            v_volume_side_length/groupCounts[1],
            v_volume_side_length/groupCounts[2]
        );
        ping_pong_barrier(comp_context.compute_cmd_buf, pressure_3D_textures[1 - pressure_ind].handle);
        pressure_ind = 1 - pressure_ind;//ping-pong
    }

    //Phase 3: gradient subtract
    vkCmdBindPipeline(comp_context.compute_cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, gradient_subtract_pipeline.handle);
    GradientSubtractPipeline::Push push_gs{
        .N = v_volume_side_length,
    };
    vkCmdPushConstants(comp_context.compute_cmd_buf, gradient_subtract_pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_gs), &push_gs);
    std::array<VkDescriptorSet, 2> sets_gs{
        velocity_volumes[velocity_ind],//read_write
        pressure_volumes[pressure_ind],//read
    };
    vkCmdBindDescriptorSets(comp_context.compute_cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, gradient_subtract_pipeline.layout, 0, (uint32_t)sets_gs.size(), sets_gs.data(), 0, nullptr);
    vkCmdDispatch(comp_context.compute_cmd_buf,
        v_volume_side_length/groupCounts[0],
        v_volume_side_length/groupCounts[1],
        v_volume_side_length/groupCounts[2]
    );
    ping_pong_barrier(comp_context.compute_cmd_buf, velocity_3D_textures[1 -velocity_ind].handle);
    velocity_ind = 1 - velocity_ind;//ping-pong

}

void FluidSystem::diffuse_velocity(float dt, ComputeContext comp_context){
    const uint32_t ITERS = 5;
    vkCmdBindPipeline(comp_context.compute_cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, diffuse_vector_pipeline.handle);
    for(uint32_t i = 0; i < ITERS; i++){//red-black gauss-seidel method, red pass fills half buffer and the black pass reads from that 
        //red pass
        DiffuseVectorPipeline::Push push_red{
            .N = v_volume_side_length,
            .dt = dt,
            .color = 0, 
        };
        vkCmdPushConstants(comp_context.compute_cmd_buf, diffuse_vector_pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_red), &push_red);
        vkCmdBindDescriptorSets(
            comp_context.compute_cmd_buf,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            diffuse_vector_pipeline.layout,
            0,
            1,
            &velocity_volumes[velocity_ind],
            0,
            nullptr
        );
        vkCmdDispatch(comp_context.compute_cmd_buf,
            v_volume_side_length/groupCounts[0],
            v_volume_side_length/groupCounts[1],
            v_volume_side_length/groupCounts[2]
        );

        //memory barrier
        ping_pong_barrier(comp_context.compute_cmd_buf, velocity_3D_textures[1- velocity_ind].handle);
        //swap
        velocity_ind = 1 - velocity_ind;

        //black pass
        DiffuseVectorPipeline::Push push_black{
            .N = v_volume_side_length,
            .dt = dt,
            .color = 1, 
        };
        vkCmdPushConstants(comp_context.compute_cmd_buf, diffuse_vector_pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_black), &push_black);
        vkCmdBindDescriptorSets(
            comp_context.compute_cmd_buf,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            diffuse_vector_pipeline.layout,
            0,
            1,
            &velocity_volumes[velocity_ind],
            0,
            nullptr
        );
        vkCmdDispatch(comp_context.compute_cmd_buf,
            v_volume_side_length/groupCounts[0],
            v_volume_side_length/groupCounts[1],
            v_volume_side_length/groupCounts[2]
        );

        //memory barrier
        ping_pong_barrier(comp_context.compute_cmd_buf, velocity_3D_textures[1- velocity_ind].handle);
        //swap
        velocity_ind = 1 - velocity_ind;
    }
}

void FluidSystem::advect_velocity(float dt, ComputeContext comp_context){
    vkCmdBindPipeline(comp_context.compute_cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, advect_vector_pipeline.handle);
    AdvectVectorPipeline::Push push{
        .N = v_volume_side_length,
        .dt = dt,
    };
    vkCmdPushConstants(comp_context.compute_cmd_buf, advect_vector_pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdBindDescriptorSets(
        comp_context.compute_cmd_buf,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        advect_vector_pipeline.layout,
        0,
        1,
        &velocity_volumes[velocity_ind],
        0,
        nullptr
    );
    vkCmdDispatch(comp_context.compute_cmd_buf,
        v_volume_side_length/groupCounts[0],
        v_volume_side_length/groupCounts[1],
        v_volume_side_length/groupCounts[2]
    );
    //memory barrier
    ping_pong_barrier(comp_context.compute_cmd_buf, velocity_3D_textures[1- velocity_ind].handle);
    //swap
    velocity_ind = 1 - velocity_ind;
}

void FluidSystem::init_fluid(RTG &rtg, ComputeContext &comp_context, MaterialSystem &material_system, SceneSystem &scene_system){
    add_scalar_sources_pipeline.create(rtg);
    diffuse_scalar_pipeline.create(rtg);
    advect_density_pipeline.create(rtg);
    add_vector_sources_pipeline.create(rtg);
    diffuse_vector_pipeline.create(rtg);
    advect_vector_pipeline.create(rtg);
    //projection pipelines
    divergence_pipeline.create(rtg);
    pressure_solve_pipeline.create(rtg);
    gradient_subtract_pipeline.create(rtg);

    //make the image, image view and sampler for the fluid 3d texture, allocate descriptor, 
    {
        density_3D_textures[0] = rtg.helpers.create_image_3D(
            VkExtent3D{.width = v_volume_side_length, .height = v_volume_side_length, .depth = v_volume_side_length}, 
            VK_FORMAT_R32_SFLOAT, 
            VK_IMAGE_TILING_OPTIMAL, 
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        density_3D_textures[1] = rtg.helpers.create_image_3D(
            VkExtent3D{.width = v_volume_side_length, .height = v_volume_side_length, .depth = v_volume_side_length}, 
            VK_FORMAT_R32_SFLOAT, 
            VK_IMAGE_TILING_OPTIMAL, 
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        material_system.make_image_view_3D(density_3D_views[0], density_3D_textures[0]);
        material_system.make_image_view_3D(density_3D_views[1], density_3D_textures[1]);
        
        velocity_3D_textures[0] = rtg.helpers.create_image_3D(
            VkExtent3D{.width = v_volume_side_length, .height = v_volume_side_length, .depth = v_volume_side_length}, 
            VK_FORMAT_R32G32B32A32_SFLOAT, 
            VK_IMAGE_TILING_OPTIMAL, 
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        velocity_3D_textures[1] = rtg.helpers.create_image_3D(
            VkExtent3D{.width = v_volume_side_length, .height = v_volume_side_length, .depth = v_volume_side_length}, 
            VK_FORMAT_R32G32B32A32_SFLOAT, 
            VK_IMAGE_TILING_OPTIMAL, 
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        material_system.make_image_view_3D(velocity_3D_views[0], velocity_3D_textures[0]);
        material_system.make_image_view_3D(velocity_3D_views[1], velocity_3D_textures[1]);

        pressure_3D_textures[0] = rtg.helpers.create_image_3D(
            VkExtent3D{.width = v_volume_side_length, .height = v_volume_side_length, .depth = v_volume_side_length}, 
            VK_FORMAT_R32_SFLOAT, 
            VK_IMAGE_TILING_OPTIMAL, 
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        pressure_3D_textures[1] = rtg.helpers.create_image_3D(
            VkExtent3D{.width = v_volume_side_length, .height = v_volume_side_length, .depth = v_volume_side_length}, 
            VK_FORMAT_R32_SFLOAT, 
            VK_IMAGE_TILING_OPTIMAL, 
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        material_system.make_image_view_3D(pressure_3D_views[0], pressure_3D_textures[0]);
        material_system.make_image_view_3D(pressure_3D_views[1], pressure_3D_textures[1]);

        divergence_3D_texture = rtg.helpers.create_image_3D(
            VkExtent3D{.width = v_volume_side_length, .height = v_volume_side_length, .depth = v_volume_side_length}, 
            VK_FORMAT_R32_SFLOAT, 
            VK_IMAGE_TILING_OPTIMAL, 
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        material_system.make_image_view_3D(divergence_3D_view, divergence_3D_texture);
    }


    { //make a sampler for the 3D volume, nearest
		VkSamplerCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.flags = 0,
			.magFilter = VK_FILTER_NEAREST,
			.minFilter = VK_FILTER_NEAREST,
			.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
			.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
			.mipLodBias = 0.0f,
			.anisotropyEnable = VK_FALSE,
			.maxAnisotropy = 0.0f, //doesn't matter if anisotropy isn't enabled
			.compareEnable = VK_FALSE,
			.compareOp = VK_COMPARE_OP_ALWAYS, //doesn't matter if compare isn't enabled
			.minLod = 0.0f,
			.maxLod = VK_LOD_CLAMP_NONE,
			.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
			.unnormalizedCoordinates = VK_FALSE,
		};
		VK( vkCreateSampler(rtg.device, &create_info, nullptr, &volume_sampler) );
	}


    std::cout<<"allocating ping-pong descriptor sets for Density"<<std::endl;
    make_ping_pong_descriptor_sets(rtg, comp_context, density_volumes, add_scalar_sources_pipeline.set0_density_volume, density_3D_views);   
    std::cout<<"allocating ping-pong descriptor sets for Velocity"<<std::endl;
    make_ping_pong_descriptor_sets(rtg, comp_context, velocity_volumes, add_vector_sources_pipeline.set0_velocity_volume, velocity_3D_views);
    std::cout<<"allocating ping-pong descriptor sets for Pressure"<<std::endl;
    make_ping_pong_descriptor_sets(rtg, comp_context, pressure_volumes, divergence_pipeline.set1_pressure_volume, pressure_3D_views);
    //make descriptor set for divergence volume and updated with the divergence image view
    {
        VkDescriptorSetAllocateInfo alloc_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = comp_context.storage_descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &divergence_pipeline.set2_divergence_volume,
        };
        vkAllocateDescriptorSets(rtg.device, &alloc_info, &divergence_volume);

        VkDescriptorImageInfo info{
            .sampler = volume_sampler,
            .imageView = divergence_3D_view,
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL
        };

        VkWriteDescriptorSet write{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = divergence_volume,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &info
        };

        vkUpdateDescriptorSets(rtg.device, 1, &write, 0, nullptr);
    }


    //compute command buffer already started recording:
    {//image layout transition
        auto layout_transition = [](VkCommandBuffer cmd_buf, Helpers::AllocatedImage3D &image){
            VkImageMemoryBarrier barrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .srcAccessMask = 0,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .image = image.handle,
                .subresourceRange = {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1
                }
            };

            vkCmdPipelineBarrier( cmd_buf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        };

        for (int i = 0; i < 2; i++) {
            
            layout_transition(comp_context.compute_cmd_buf, density_3D_textures[i]);
            layout_transition(comp_context.compute_cmd_buf, velocity_3D_textures[i]);
            layout_transition(comp_context.compute_cmd_buf, pressure_3D_textures[i]);
        }
        layout_transition(comp_context.compute_cmd_buf, divergence_3D_texture);
    }

    {//clear both volumes
        VkClearColorValue zero = {0.0f, 0.0f, 0.0f, 0.0f};
        VkImageSubresourceRange range{
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        };
        for (int i = 0; i < 2; i++) {
            vkCmdClearColorImage(comp_context.compute_cmd_buf, velocity_3D_textures[i].handle, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
            vkCmdClearColorImage( comp_context.compute_cmd_buf, density_3D_textures[i].handle, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &range);
        }
    }

    
    //add sources
    add_sources_velocity(0.5f, comp_context, scene_system);

    //project
    project_velocity(comp_context);




}

void FluidSystem::update_fluid(float dt, ComputeContext comp_context, SceneSystem &scene_system){

    {//velocity
        const bool do_add_sources = true;
        const bool do_diffuse = true;
        const bool do_advect = true;
        const bool do_project = true;

        if(do_add_sources){//add sources
            add_sources_velocity(dt, comp_context, scene_system);
        }

        if(do_diffuse){//diffuse
            diffuse_velocity(dt, comp_context);
        }

        if(do_project){//project
            project_velocity(comp_context);
        }

        if(do_advect){//advect
            advect_velocity(dt, comp_context);
        }

        if(do_project){//project
            project_velocity(comp_context);
        }
    }

    {//density
        const bool do_add_sources = true;
        const bool do_diffuse = false;
        const bool do_advect = true;
        
        if(do_add_sources){//add sources
            add_sources_density(dt, comp_context);
        }
        if(do_diffuse){//diffuse
            diffuse_density(dt, comp_context);
        }
        if(do_advect){//advect
            advect_density(dt, comp_context);
        }
    }


    



}


void FluidSystem::destroy(RTG &rtg){
    {//fluid stuff		
		if(volume_sampler)
		{
			vkDestroySampler(rtg.device, volume_sampler, nullptr);
			volume_sampler = VK_NULL_HANDLE;
		}

		//image views
		for (VkImageView &v_view_3D : velocity_3D_views){
			vkDestroyImageView(rtg.device, v_view_3D, nullptr);
			v_view_3D = VK_NULL_HANDLE;
		}
		for (VkImageView &p_view_3D : pressure_3D_views){
			vkDestroyImageView(rtg.device, p_view_3D, nullptr);
			p_view_3D = VK_NULL_HANDLE;
		}
		vkDestroyImageView(rtg.device, divergence_3D_view, nullptr);
		divergence_3D_view = VK_NULL_HANDLE;

		//images
		for (auto &v_image_3D : velocity_3D_textures){
			rtg.helpers.destroy_image_3D(std::move(v_image_3D));
		}
		for (auto &p_image_3D : pressure_3D_textures){
			rtg.helpers.destroy_image_3D(std::move(p_image_3D));
		}
		rtg.helpers.destroy_image_3D(std::move(divergence_3D_texture));

        add_scalar_sources_pipeline.destroy(rtg);
        diffuse_scalar_pipeline.destroy(rtg);
        advect_density_pipeline.destroy(rtg);

        add_vector_sources_pipeline.destroy(rtg);
        diffuse_vector_pipeline.destroy(rtg);
        advect_vector_pipeline.destroy(rtg);
        divergence_pipeline.destroy(rtg);
        pressure_solve_pipeline.destroy(rtg);
        gradient_subtract_pipeline.destroy(rtg);
	}
}