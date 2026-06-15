#include "ParticleSystem.hpp"
#include "VK.hpp"
#include <array>
#include <cstring>

static uint32_t comp_code[] =
#include "spv/particles.comp.inl"
;

static uint32_t vert_code[] =
#include "spv/particles.vert.inl"
;

static uint32_t frag_code[] =
#include "spv/particles.frag.inl"
;

// -----------------------------------------------------------------------
// ParticleComputePipeline
// -----------------------------------------------------------------------

void ParticleComputePipeline::create(RTG &rtg) {
    VkShaderModule comp_module = rtg.helpers.create_shader_module(comp_code);

    { // set0: binding 0 = SSBO, binding 1 = combined image sampler (velocity)
        std::array<VkDescriptorSetLayoutBinding, 2> bindings{
            VkDescriptorSetLayoutBinding{
                .binding         = 0,
                .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT,
            },
            VkDescriptorSetLayoutBinding{
                .binding         = 1,
                .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = 1,
                .stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT,
            },
        };
        VkDescriptorSetLayoutCreateInfo ci{
            .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = uint32_t(bindings.size()),
            .pBindings    = bindings.data(),
        };
        VK(vkCreateDescriptorSetLayout(rtg.device, &ci, nullptr, &set0_layout));
    }

    {
        VkPushConstantRange range{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset     = 0,
            .size       = sizeof(Push),
        };
        std::array<VkDescriptorSetLayout, 1> layouts{set0_layout};
        VkPipelineLayoutCreateInfo ci{
            .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount         = uint32_t(layouts.size()),
            .pSetLayouts            = layouts.data(),
            .pushConstantRangeCount = 1,
            .pPushConstantRanges    = &range,
        };
        VK(vkCreatePipelineLayout(rtg.device, &ci, nullptr, &layout));
    }

    {
        VkPipelineShaderStageCreateInfo stage{
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = comp_module,
            .pName  = "main",
        };
        VkComputePipelineCreateInfo ci{
            .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage  = stage,
            .layout = layout,
        };
        VK(vkCreateComputePipelines(rtg.device, VK_NULL_HANDLE, 1, &ci, nullptr, &handle));
    }

    vkDestroyShaderModule(rtg.device, comp_module, nullptr);
}

void ParticleComputePipeline::destroy(RTG &rtg) {
    if (handle != VK_NULL_HANDLE) {
        vkDestroyPipeline(rtg.device, handle, nullptr);
        handle = VK_NULL_HANDLE;
    }
    if (layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(rtg.device, layout, nullptr);
        layout = VK_NULL_HANDLE;
    }
    if (set0_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(rtg.device, set0_layout, nullptr);
        set0_layout = VK_NULL_HANDLE;
    }
}

// -----------------------------------------------------------------------
// ParticleRenderPipeline
// -----------------------------------------------------------------------

void ParticleRenderPipeline::create(RTG &rtg, VkRenderPass render_pass, uint32_t subpass) {
    VkShaderModule vert_module = rtg.helpers.create_shader_module(vert_code);
    VkShaderModule frag_module = rtg.helpers.create_shader_module(frag_code);

    {
        VkPushConstantRange range{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            .offset     = 0,
            .size       = sizeof(Push),
        };
        VkPipelineLayoutCreateInfo ci{
            .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount         = 0,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges    = &range,
        };
        VK(vkCreatePipelineLayout(rtg.device, &ci, nullptr, &layout));
    }

    {
        std::array<VkPipelineShaderStageCreateInfo, 2> stages{
            VkPipelineShaderStageCreateInfo{
                .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage  = VK_SHADER_STAGE_VERTEX_BIT,
                .module = vert_module,
                .pName  = "main",
            },
            VkPipelineShaderStageCreateInfo{
                .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage  = VK_SHADER_STAGE_FRAGMENT_BIT,
                .module = frag_module,
                .pName  = "main",
            },
        };

        VkVertexInputBindingDescription binding{
            .binding   = 0,
            .stride    = sizeof(float) * 4,
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        };
        VkVertexInputAttributeDescription attr{
            .location = 0,
            .binding  = 0,
            .format   = VK_FORMAT_R32G32B32A32_SFLOAT,
            .offset   = 0,
        };
        VkPipelineVertexInputStateCreateInfo vertex_input{
            .sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount   = 1,
            .pVertexBindingDescriptions      = &binding,
            .vertexAttributeDescriptionCount = 1,
            .pVertexAttributeDescriptions    = &attr,
        };

        VkPipelineInputAssemblyStateCreateInfo input_assembly{
            .sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
        };

        std::array<VkDynamicState, 2> dynamic_states{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic_state{
            .sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = uint32_t(dynamic_states.size()),
            .pDynamicStates    = dynamic_states.data(),
        };

        VkPipelineViewportStateCreateInfo viewport_state{
            .sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .scissorCount  = 1,
        };

        VkPipelineRasterizationStateCreateInfo rasterization{
            .sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode    = VK_CULL_MODE_NONE,
            .frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE,
            .lineWidth   = 1.0f,
        };

        VkPipelineMultisampleStateCreateInfo multisample{
            .sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        };

        VkPipelineDepthStencilStateCreateInfo depth_stencil{
            .sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable  = VK_TRUE,
            .depthWriteEnable = VK_FALSE,
            .depthCompareOp   = VK_COMPARE_OP_LESS,
        };

        std::array<VkPipelineColorBlendAttachmentState, 1> blend_attachments{
            VkPipelineColorBlendAttachmentState{
                .blendEnable         = VK_TRUE,
                .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
                .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                .colorBlendOp        = VK_BLEND_OP_ADD,
                .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
                .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
                .alphaBlendOp        = VK_BLEND_OP_ADD,
                .colorWriteMask      = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                       VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
            },
        };
        VkPipelineColorBlendStateCreateInfo color_blend{
            .sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .attachmentCount = uint32_t(blend_attachments.size()),
            .pAttachments    = blend_attachments.data(),
        };

        VkGraphicsPipelineCreateInfo ci{
            .sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount          = uint32_t(stages.size()),
            .pStages             = stages.data(),
            .pVertexInputState   = &vertex_input,
            .pInputAssemblyState = &input_assembly,
            .pViewportState      = &viewport_state,
            .pRasterizationState = &rasterization,
            .pMultisampleState   = &multisample,
            .pDepthStencilState  = &depth_stencil,
            .pColorBlendState    = &color_blend,
            .pDynamicState       = &dynamic_state,
            .layout              = layout,
            .renderPass          = render_pass,
            .subpass             = subpass,
        };
        VK(vkCreateGraphicsPipelines(rtg.device, VK_NULL_HANDLE, 1, &ci, nullptr, &handle));
    }

    vkDestroyShaderModule(rtg.device, vert_module, nullptr);
    vkDestroyShaderModule(rtg.device, frag_module, nullptr);
}

void ParticleRenderPipeline::destroy(RTG &rtg) {
    if (handle != VK_NULL_HANDLE) {
        vkDestroyPipeline(rtg.device, handle, nullptr);
        handle = VK_NULL_HANDLE;
    }
    if (layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(rtg.device, layout, nullptr);
        layout = VK_NULL_HANDLE;
    }
}

// -----------------------------------------------------------------------
// ParticleSystem
// -----------------------------------------------------------------------

void ParticleSystem::init(RTG &rtg, ComputeContext &comp_ctx,
                          VkDescriptorPool particle_pool,
                          VkImageView velocity_views[2]) {
    compute_pipeline.create(rtg);

    particle_buffer = rtg.helpers.create_buffer(
        NUM_PARTICLES * sizeof(float) * 4,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT  |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        Helpers::Unmapped
    );

    {
        VkSamplerCreateInfo ci{
            .sType        = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter    = VK_FILTER_LINEAR,
            .minFilter    = VK_FILTER_LINEAR,
            .mipmapMode   = VK_SAMPLER_MIPMAP_MODE_LINEAR,
            .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .minLod       = 0.0f,
            .maxLod       = VK_LOD_CLAMP_NONE,
        };
        VK(vkCreateSampler(rtg.device, &ci, nullptr, &linear_sampler));
    }

    {
        std::array<VkDescriptorSetLayout, 2> layouts{
            compute_pipeline.set0_layout,
            compute_pipeline.set0_layout,
        };
        VkDescriptorSetAllocateInfo ai{
            .sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool     = particle_pool,
            .descriptorSetCount = 2,
            .pSetLayouts        = layouts.data(),
        };
        VK(vkAllocateDescriptorSets(rtg.device, &ai, compute_sets));
    }

    VkDescriptorBufferInfo buf_info{
        .buffer = particle_buffer.handle,
        .offset = 0,
        .range  = VK_WHOLE_SIZE,
    };
    for (uint32_t i = 0; i < 2; i++) {
        VkDescriptorImageInfo img_info{
            .sampler     = linear_sampler,
            .imageView   = velocity_views[i],
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };
        std::array<VkWriteDescriptorSet, 2> writes{
            VkWriteDescriptorSet{
                .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet          = compute_sets[i],
                .dstBinding      = 0,
                .descriptorCount = 1,
                .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo     = &buf_info,
            },
            VkWriteDescriptorSet{
                .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet          = compute_sets[i],
                .dstBinding      = 1,
                .descriptorCount = 1,
                .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .pImageInfo      = &img_info,
            },
        };
        vkUpdateDescriptorSets(rtg.device, uint32_t(writes.size()), writes.data(), 0, nullptr);
    }

    // Zero the buffer so w=0.0f (dead) causes all particles to respawn on frame 1
    vkCmdFillBuffer(comp_ctx.compute_cmd_buf, particle_buffer.handle, 0, VK_WHOLE_SIZE, 0);

    VkBufferMemoryBarrier barrier{
        .sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask       = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask       = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer              = particle_buffer.handle,
        .offset              = 0,
        .size                = VK_WHOLE_SIZE,
    };
    vkCmdPipelineBarrier(
        comp_ctx.compute_cmd_buf,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 1, &barrier, 0, nullptr
    );
}

void ParticleSystem::destroy(RTG &rtg) {
    if (linear_sampler != VK_NULL_HANDLE) {
        vkDestroySampler(rtg.device, linear_sampler, nullptr);
        linear_sampler = VK_NULL_HANDLE;
    }
    if (particle_buffer.handle != VK_NULL_HANDLE) {
        rtg.helpers.destroy_buffer(std::move(particle_buffer));
    }
    // compute_sets freed when particle_descriptor_pool is destroyed
    compute_pipeline.destroy(rtg);
    render_pipeline.destroy(rtg);
}

void ParticleSystem::update(ComputeContext &comp_ctx, float dt,
                            uint32_t velocity_ind, uint32_t frame) {
    vkCmdBindPipeline(
        comp_ctx.compute_cmd_buf,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        compute_pipeline.handle
    );

    vkCmdBindDescriptorSets(
        comp_ctx.compute_cmd_buf,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        compute_pipeline.layout,
        0, 1, &compute_sets[velocity_ind],
        0, nullptr
    );

    ParticleComputePipeline::Push push{
        .num_particles = NUM_PARTICLES,
        .dt            = dt,
        .N             = 128,
        .frame         = frame,
    };
    vkCmdPushConstants(
        comp_ctx.compute_cmd_buf,
        compute_pipeline.layout,
        VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(push), &push
    );

    uint32_t groups = (NUM_PARTICLES + 63) / 64;
    vkCmdDispatch(comp_ctx.compute_cmd_buf, groups, 1, 1);

    // Ensure compute writes are visible to the vertex shader
    VkBufferMemoryBarrier barrier{
        .sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask       = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer              = particle_buffer.handle,
        .offset              = 0,
        .size                = VK_WHOLE_SIZE,
    };
    vkCmdPipelineBarrier(
        comp_ctx.compute_cmd_buf,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
        0, 0, nullptr, 1, &barrier, 0, nullptr
    );
}

void ParticleSystem::draw(VkCommandBuffer cmd, mat4 const &CLIP_FROM_WORLD,
                          vec3 volume_center, float cell_size_ws, uint32_t N) {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, render_pipeline.handle);

    std::array<VkBuffer, 1>     vbs{particle_buffer.handle};
    std::array<VkDeviceSize, 1> offsets{0};
    vkCmdBindVertexBuffers(cmd, 0, 1, vbs.data(), offsets.data());

    ParticleRenderPipeline::Push push{
        .CLIP_FROM_WORLD    = CLIP_FROM_WORLD,
        .volume_center_cell = vec4(volume_center.x, volume_center.y,
                                   volume_center.z, cell_size_ws),
        .N                  = N,
    };
    vkCmdPushConstants(
        cmd, render_pipeline.layout,
        VK_SHADER_STAGE_VERTEX_BIT,
        0, sizeof(push), &push
    );

    vkCmdDraw(cmd, NUM_PARTICLES, 1, 0, 0);
}
