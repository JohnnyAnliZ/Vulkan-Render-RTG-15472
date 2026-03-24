#include "Tutorial.hpp"
#include "Helpers.hpp"
#include "VK.hpp"


static uint32_t vert_code[] = 
#include "spv/shadows2D.vert.inl"
;

void Tutorial::draw_all_objects(VkCommandBuffer const &cmd, mat4 const &LIGHTS_CLIP_FROM_WORLD, vec4 const &_shadow_atlas){
    //push constants
    Shadow2DPipeline::Push push{
        .LIGHT_CLIP_FROM_WORLD = LIGHTS_CLIP_FROM_WORLD,
        .shadow_atlas = _shadow_atlas,
    };
    vkCmdPushConstants(cmd, shadow_2D_pipeline.layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push), &push);
    
    float atlas_size_fl = float(atlas_size);
    float offset_x = atlas_size_fl * _shadow_atlas.x;
    float offset_y = atlas_size_fl * _shadow_atlas.y;
    float size_x = atlas_size_fl * _shadow_atlas.z;
    float size_y = atlas_size_fl * _shadow_atlas.w;


    {//scissor rectangle
        VkRect2D scissor{
            .offset = {.x = int32_t(offset_x), .y = int32_t(offset_y)},
            .extent = {.width =uint32_t(size_x), .height = uint32_t(size_y)},
        };
        vkCmdSetScissor(cmd, 0, 1, &scissor);
    }
    {//viewport transform 
        VkViewport viewport{
            .x = offset_x,
            .y = offset_y,
            .width = size_x,
            .height = size_y,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        vkCmdSetViewport(cmd, 0, 1, &viewport);
    }

    //literally draw all the objects
    uint32_t index_offset = 0;
    if(!lambertian_object_instances.empty()){
        for (LambertianObjectInstance const &inst : lambertian_object_instances) {
            uint32_t index = uint32_t(&inst - &lambertian_object_instances[0]);

            vkCmdDraw(cmd, inst.vertices.count, 1, inst.vertices.first, index+index_offset);
        }
    }
    index_offset = (uint32_t)lambertian_object_instances.size();

    if(!env_mirror_object_instances.empty()){
        for (EnvMirrorObjectInstance const &inst : env_mirror_object_instances) {
            uint32_t index = uint32_t(&inst - &env_mirror_object_instances[0]);

            vkCmdDraw(cmd, inst.vertices.count, 1, inst.vertices.first, index+index_offset);
        }
    }
    index_offset = (uint32_t)env_mirror_object_instances.size();

    if(!pbr_object_instances.empty()){
        for (PbrObjectInstance const &inst : pbr_object_instances) {
            uint32_t index = uint32_t(&inst - &pbr_object_instances[0]);

            vkCmdDraw(cmd, inst.vertices.count, 1, inst.vertices.first, index+index_offset);
        }
    }
}


void Tutorial::Shadow2DPipeline::create(RTG &rtg, VkRenderPass render_pass, uint32_t subpass){
    VkShaderModule vert_module = rtg.helpers.create_shader_module(vert_code);

    {//the set0_Transform layout holds an array of Transform structures in a storage buffer used in the vertex shader:
        std::array<VkDescriptorSetLayoutBinding, 1> bindings{
            VkDescriptorSetLayoutBinding{
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            }
        };

        VkDescriptorSetLayoutCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = uint32_t(bindings.size()),
            .pBindings = bindings.data(),
        };
        VK(vkCreateDescriptorSetLayout(rtg.device, &create_info, nullptr, &set0_Transforms));
    }
   

    {//create pipeline layout
        VkPushConstantRange range{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            .offset = 0,
            .size = sizeof(Push),
        };

        std::array< VkDescriptorSetLayout, 1 > layouts{
			set0_Transforms
		};

        VkPipelineLayoutCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = uint32_t(layouts.size()),
            .pSetLayouts = layouts.data(),
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &range,
        };

        VK(vkCreatePipelineLayout(rtg.device, &create_info, nullptr, &layout));
    }

    {//create pipeline
        std::array<VkPipelineShaderStageCreateInfo, 1> stages{
            VkPipelineShaderStageCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage =VK_SHADER_STAGE_VERTEX_BIT,
                .module = vert_module,
                .pName = "main",
            }, 
        };
        //dynamic states (viewport and scissor will be set dynamically through state commands)
        std::vector<VkDynamicState> dynamic_states{
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR,
            VK_DYNAMIC_STATE_DEPTH_BIAS,
        };
        VkPipelineDynamicStateCreateInfo dynamic_state{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = uint32_t(dynamic_states.size()),
            .pDynamicStates = dynamic_states.data(),
        };

    //inpute assembly state structure, this one draws lines
        VkPipelineInputAssemblyStateCreateInfo input_assembly_state{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = VK_FALSE,
        };
    //viewport state(to say it uses only one viewport and one scissor rectangle)
        VkPipelineViewportStateCreateInfo viewport_state{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .scissorCount = 1,
        };
    //configure rasterizer to rasterize only backfaces (cuz we shadow mappin) 
        VkPipelineRasterizationStateCreateInfo rasterization_state{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable = VK_FALSE,//Maybe I have to enable it for sun
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_BACK_BIT,
            .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
            .depthBiasEnable = VK_TRUE,
            .lineWidth = 1.0f,
        };
    //disable multisampling
        VkPipelineMultisampleStateCreateInfo multisample_state{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			.sampleShadingEnable = VK_FALSE,
		};
    //depth attatchment
        VkPipelineDepthStencilStateCreateInfo depth_stencil_state{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = VK_TRUE,
            .depthWriteEnable = VK_TRUE,
            .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
			.depthBoundsTestEnable = VK_FALSE,
			.stencilTestEnable = VK_FALSE,
		};

    //create da pipeline
        VkGraphicsPipelineCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = (uint32_t)stages.size(),
            .pStages = stages.data(),
            .pVertexInputState = &Vertex::array_input_state,
            .pInputAssemblyState = &input_assembly_state,
            .pViewportState = &viewport_state,
            .pRasterizationState = &rasterization_state,
			.pMultisampleState = &multisample_state,
			.pDepthStencilState = &depth_stencil_state,
            .pColorBlendState = nullptr,
			.pDynamicState = &dynamic_state,
			.layout = layout,
			.renderPass = render_pass,
			.subpass = subpass,
        };
        VK(vkCreateGraphicsPipelines(rtg.device, VK_NULL_HANDLE, 1, &create_info, nullptr, &handle));
    }
    //deallocate shader modules now that they are no longer used
    vkDestroyShaderModule(rtg.device, vert_module, nullptr);
}

void Tutorial::Shadow2DPipeline::destroy(RTG &rtg){
    if(handle != VK_NULL_HANDLE){
        vkDestroyPipeline(rtg.device, handle, nullptr);
        handle = VK_NULL_HANDLE;
    }
    if(layout != VK_NULL_HANDLE){
        vkDestroyPipelineLayout(rtg.device, layout, nullptr);
        layout = VK_NULL_HANDLE;
    }

    if (set0_Transforms != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(rtg.device, set0_Transforms, nullptr);
		set0_Transforms = VK_NULL_HANDLE;
	}

}