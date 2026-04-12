#include "Tutorial.hpp"
#include "Helpers.hpp"
#include "VK.hpp"

void Tutorial::make_velocity_descriptor_sets(
    VkDescriptorSet *velocity_sets,
    VkDescriptorSetLayout const &layout
){

    std::cout << "Allocating ping-pong velocity descriptor sets\n";
    // Allocate both sets at once
    std::array<VkDescriptorSetLayout,2> layouts{layout, layout};
    VkDescriptorSetAllocateInfo alloc_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = storage_descriptor_pool,
        .descriptorSetCount = (uint32_t)layouts.size(),
        .pSetLayouts = layouts.data(), // same layout twice
    };

    VK(vkAllocateDescriptorSets(rtg.device, &alloc_info, velocity_sets));

    // --- Descriptor setup ---
    for (uint32_t i = 0; i < 2; i++) {

        uint32_t read  = i;
        uint32_t write = 1 - i;

        std::array<VkDescriptorImageInfo, 2> infos{
            VkDescriptorImageInfo{
                .sampler = texture_sampler,
                .imageView = velocity_3D_views[read],
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL
            },
            VkDescriptorImageInfo{
                .sampler = texture_sampler,
                .imageView = velocity_3D_views[write],
                .imageLayout = VK_IMAGE_LAYOUT_GENERAL
            }
        };

        std::array<VkWriteDescriptorSet, 2> writes;

        for (uint32_t b = 0; b < 2; b++) {
            writes[b] = VkWriteDescriptorSet{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = velocity_sets[i],
                .dstBinding = b,
                .descriptorCount = 1,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                .pImageInfo = &infos[b],
            };
        }

        vkUpdateDescriptorSets(rtg.device, 2, writes.data(), 0, nullptr);
    }
}


void Tutorial::init_fluid(){
    add_vector_sources_pipeline.create(rtg, VK_NULL_HANDLE, 0);

    //make the image, image view and sampler for the fluid 3d texture, allocate descriptor, 
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

    //use the regular texture sampler


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
    
    make_velocity_descriptor_sets(velocity_volumes, add_vector_sources_pipeline.set0_velocity_volume);


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

    auto compute_shader_write_barrier = [&](){
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
    };


    {//add sources
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
        compute_shader_write_barrier();

        //swap
        velocity_ind = 1 - velocity_ind;
    }

    {//diffuse

    }

    {//project

    }

    {//advect

    }

    {//project

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