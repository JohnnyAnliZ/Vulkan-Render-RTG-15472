#include "Tutorial.hpp"
#include "Helpers.hpp"
#include "VK.hpp"

//this function make sure the written image is available for reading in the next dispatch, by inserting a memory barrier for the written image
void Tutorial::velocity_barrier(){
    VkImageMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .image = velocity_3D_textures[1 - velocity_ind].handle, // written image
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1,
            .layerCount = 1
        }
    };
    vkCmdPipelineBarrier(
        compute_cmd_buf,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );
}


void Tutorial::pressure_barrier(){
    VkImageMemoryBarrier barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .image = pressure_3D_textures[1 - pressure_ind].handle, // written image
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1,
            .layerCount = 1
        }
    };
    vkCmdPipelineBarrier(
        compute_cmd_buf,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &barrier
    );
}


void Tutorial::make_ping_pong_descriptor_sets(
    VkDescriptorSet *pressure_sets,
    VkDescriptorSetLayout const &layout,
    VkImageView *image_views
){
    
    // Allocate both sets at once
    std::array<VkDescriptorSetLayout,2> layouts{layout, layout};
    VkDescriptorSetAllocateInfo alloc_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = storage_descriptor_pool,
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


void Tutorial::add_sources_velocity(float dt){
    vkCmdBindPipeline(
        compute_cmd_buf,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        add_vector_sources_pipeline.handle
    );

    vkCmdBindDescriptorSets(
        compute_cmd_buf,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        add_vector_sources_pipeline.layout,
        0,
        1,
        &velocity_volumes[velocity_ind],
        0,
        nullptr
    );

    AddVectorSourcesPipeline::Push push{
        .N = v_volume_side_length,
        .dt = dt,
    };
    vkCmdPushConstants(compute_cmd_buf, add_vector_sources_pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    vkCmdDispatch(  
        compute_cmd_buf,
        v_volume_side_length/groupCounts[0],
        v_volume_side_length/groupCounts[1],
        v_volume_side_length/groupCounts[2]
    );  

    //memory barrier
    velocity_barrier();

    //swap
    velocity_ind = 1 - velocity_ind;
}


void Tutorial::project_velocity(){
    //Phase 1: calculate divergence and clear pressure
    vkCmdBindPipeline(compute_cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, divergence_pipeline.handle);
    DivergencePipeline::Push push_div{
        .N = v_volume_side_length,
    };
    vkCmdPushConstants(compute_cmd_buf, divergence_pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_div), &push_div);
    std::array<VkDescriptorSet, 3> sets{
        velocity_volumes[velocity_ind],//read
        pressure_volumes[pressure_ind],//read
        divergence_volume,//write
    };
    vkCmdBindDescriptorSets(
        compute_cmd_buf,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        divergence_pipeline.layout,
        0,
        sets.size(),
        sets.data(),
        0,
        nullptr
    );
    vkCmdDispatch(compute_cmd_buf,
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
    vkCmdBindPipeline(compute_cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pressure_solve_pipeline.handle);
    PressureSolvePipeline::Push push_ps{
        .N = v_volume_side_length,
    };
    vkCmdPushConstants(compute_cmd_buf, pressure_solve_pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_ps), &push_ps);
    vkCmdBindDescriptorSets(compute_cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pressure_solve_pipeline.layout, 0, 1, &divergence_volume, 0, nullptr);
    const uint32_t ITERS = 20;
    for(uint32_t i = 0; i < ITERS; i++){//jacobi    
        vkCmdBindDescriptorSets(compute_cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, pressure_solve_pipeline.layout, 1, 1, &pressure_volumes[pressure_ind], 0, nullptr);
        vkCmdDispatch(compute_cmd_buf,
            v_volume_side_length/groupCounts[0],
            v_volume_side_length/groupCounts[1],
            v_volume_side_length/groupCounts[2]
        );
        pressure_barrier();
        pressure_ind = 1 - pressure_ind;//ping-pong
    }

    //Phase 3: gradient subtract
    vkCmdBindPipeline(compute_cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, gradient_subtract_pipeline.handle);
    GradientSubtractPipeline::Push push_gs{
        .N = v_volume_side_length,
    };
    vkCmdPushConstants(compute_cmd_buf, gradient_subtract_pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_gs), &push_gs);
    std::array<VkDescriptorSet, 2> sets_gs{
        velocity_volumes[velocity_ind],//read_write
        pressure_volumes[pressure_ind],//read
    };
    vkCmdBindDescriptorSets(compute_cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, gradient_subtract_pipeline.layout, 0, sets_gs.size(), sets_gs.data(), 0, nullptr);
    vkCmdDispatch(compute_cmd_buf,
        v_volume_side_length/groupCounts[0],
        v_volume_side_length/groupCounts[1],
        v_volume_side_length/groupCounts[2]
    );
    velocity_barrier();
    velocity_ind = 1 - velocity_ind;//ping-pong

}

void Tutorial::diffuse_velocity(float dt){
    const uint32_t ITERS = 20;
    vkCmdBindPipeline(compute_cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, diffuse_vector_pipeline.handle);
    for(uint32_t i = 0; i < ITERS; i++){//red-black gauss-seidel method, red pass fills half buffer and the black pass reads from that 
        //red pass
        DiffuseVectorPipeline::Push push_red{
            .N = v_volume_side_length,
            .dt = dt,
            .color = 0, 
        };
        vkCmdPushConstants(compute_cmd_buf, diffuse_vector_pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_red), &push_red);
        vkCmdBindDescriptorSets(
            compute_cmd_buf,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            diffuse_vector_pipeline.layout,
            0,
            1,
            &velocity_volumes[velocity_ind],
            0,
            nullptr
        );
        vkCmdDispatch(compute_cmd_buf,
            v_volume_side_length/groupCounts[0],
            v_volume_side_length/groupCounts[1],
            v_volume_side_length/groupCounts[2]
        );

        //memory barrier
        velocity_barrier();
        //swap
        velocity_ind = 1 - velocity_ind;

        //black pass
        DiffuseVectorPipeline::Push push_black{
            .N = v_volume_side_length,
            .dt = dt,
            .color = 1, 
        };
        vkCmdPushConstants(compute_cmd_buf, diffuse_vector_pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_black), &push_black);
        vkCmdBindDescriptorSets(
            compute_cmd_buf,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            diffuse_vector_pipeline.layout,
            0,
            1,
            &velocity_volumes[velocity_ind],
            0,
            nullptr
        );
        vkCmdDispatch(compute_cmd_buf,
            v_volume_side_length/groupCounts[0],
            v_volume_side_length/groupCounts[1],
            v_volume_side_length/groupCounts[2]
        );

        //memory barrier
        velocity_barrier();
        //swap
        velocity_ind = 1 - velocity_ind;
    }
}

void Tutorial::advect_velocity(float dt){
    vkCmdBindPipeline(compute_cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, advect_vector_pipeline.handle);
    AdvectVectorPipeline::Push push{
        .N = v_volume_side_length,
        .dt = dt,
    };
    vkCmdPushConstants(compute_cmd_buf, advect_vector_pipeline.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdBindDescriptorSets(
        compute_cmd_buf,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        advect_vector_pipeline.layout,
        0,
        1,
        &velocity_volumes[velocity_ind],
        0,
        nullptr
    );
    vkCmdDispatch(compute_cmd_buf,
        v_volume_side_length/groupCounts[0],
        v_volume_side_length/groupCounts[1],
        v_volume_side_length/groupCounts[2]
    );
    //memory barrier
    velocity_barrier();
    //swap
    velocity_ind = 1 - velocity_ind;
}

void Tutorial::init_fluid(){
    add_vector_sources_pipeline.create(rtg);
    diffuse_vector_pipeline.create(rtg);
    advect_vector_pipeline.create(rtg);
    //projection pipelines
    divergence_pipeline.create(rtg);
    pressure_solve_pipeline.create(rtg);
    gradient_subtract_pipeline.create(rtg);

    //make the image, image view and sampler for the fluid 3d texture, allocate descriptor, 
    {
        velocity_3D_textures[0] = rtg.helpers.create_image_3D(
            VkExtent3D{.width = 128, .height = 128, .depth = 128}, 
            VK_FORMAT_R32G32B32A32_SFLOAT, 
            VK_IMAGE_TILING_OPTIMAL, 
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        velocity_3D_textures[1] = rtg.helpers.create_image_3D(
            VkExtent3D{.width = 128, .height = 128, .depth = 128}, 
            VK_FORMAT_R32G32B32A32_SFLOAT, 
            VK_IMAGE_TILING_OPTIMAL, 
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        make_image_view_3D(velocity_3D_views[0], velocity_3D_textures[0]);
        make_image_view_3D(velocity_3D_views[1], velocity_3D_textures[1]);

        pressure_3D_textures[0] = rtg.helpers.create_image_3D(
            VkExtent3D{.width = 128, .height = 128, .depth = 128}, 
            VK_FORMAT_R32_SFLOAT, 
            VK_IMAGE_TILING_OPTIMAL, 
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        pressure_3D_textures[1] = rtg.helpers.create_image_3D(
            VkExtent3D{.width = 128, .height = 128, .depth = 128}, 
            VK_FORMAT_R32_SFLOAT, 
            VK_IMAGE_TILING_OPTIMAL, 
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        make_image_view_3D(pressure_3D_views[0], pressure_3D_textures[0]);
        make_image_view_3D(pressure_3D_views[1], pressure_3D_textures[1]);

        divergence_3D_texture = rtg.helpers.create_image_3D(
            VkExtent3D{.width = 128, .height = 128, .depth = 128}, 
            VK_FORMAT_R32_SFLOAT, 
            VK_IMAGE_TILING_OPTIMAL, 
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        );
        make_image_view_3D(divergence_3D_view, divergence_3D_texture);
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

    //allocate the sampled descriptor set
    {
        VkDescriptorSetAllocateInfo alloc_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = texture_descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &texture_debug_pipeline.set1_vel_vol,
        };
        vkAllocateDescriptorSets(rtg.device, &alloc_info, &velocity_tex);
    }
    
    std::cout<<"allocating ping-pong descriptor sets for Velocity";
    make_ping_pong_descriptor_sets(velocity_volumes, add_vector_sources_pipeline.set0_velocity_volume, velocity_3D_views);
    std::cout<<"allocating ping-pong descriptor sets for Pressure";
    make_ping_pong_descriptor_sets(pressure_volumes, divergence_pipeline.set1_pressure_volume, pressure_3D_views);
    

    //record:
    VkCommandBufferBeginInfo begin_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = 0,
    };
    VK(vkBeginCommandBuffer(compute_cmd_buf, &begin_info));
    {//image layout transition
        for (int i = 0; i < 2; i++) {
            VkImageMemoryBarrier barrier{
                .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                .srcAccessMask = 0,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                .image = velocity_3D_textures[i].handle,
                .subresourceRange = {
                    .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1
                }
            };

            vkCmdPipelineBarrier(
                compute_cmd_buf,
                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0, nullptr,
                0, nullptr,
                1, &barrier
            );
        }
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
            vkCmdClearColorImage(
                compute_cmd_buf,
                velocity_3D_textures[i].handle,
                VK_IMAGE_LAYOUT_GENERAL,
                &zero,
                1,
                &range
            );
        }
    }

    //add sources
    add_sources_velocity(0.5f);

    //project
    project_velocity();

    VK(vkEndCommandBuffer(compute_cmd_buf));

    VkSubmitInfo submit{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &compute_cmd_buf,
    };
    vkQueueSubmit(rtg.graphics_queue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(rtg.graphics_queue);
    
}

void Tutorial::update_fluid(float dt){
    vkResetCommandBuffer(compute_cmd_buf, 0);

    VkCommandBufferBeginInfo begin_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
    };
    VK(vkBeginCommandBuffer(compute_cmd_buf, &begin_info));

    const bool do_add_sources = false;
    const bool do_diffuse = false;
    const bool do_advect = true;
    const bool do_project = true;

    if(do_add_sources){//add sources
        add_sources_velocity(dt);
    }

    if(do_diffuse){//diffuse
        diffuse_velocity(dt);
    }

    if(do_project){//project
        project_velocity();
    }

    if(do_advect){//advect
        advect_velocity(dt);
    }

    if(do_project){//project
        project_velocity();
    }
    
    {//submit compute work
        VK(vkEndCommandBuffer(compute_cmd_buf));

        VkSubmitInfo submit{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &compute_cmd_buf,
        };

        vkQueueSubmit(rtg.graphics_queue, 1, &submit, VK_NULL_HANDLE);
        vkQueueWaitIdle(rtg.graphics_queue);
    }    

    
    {//update descriptor sets for the two sampled images
        VkDescriptorImageInfo image_info{
            .sampler = texture_sampler,
            .imageView = velocity_3D_views[velocity_ind],
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };

        VkWriteDescriptorSet write{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = velocity_tex,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &image_info,
        };

        vkUpdateDescriptorSets(rtg.device, 1, &write, 0, nullptr);
    }


}