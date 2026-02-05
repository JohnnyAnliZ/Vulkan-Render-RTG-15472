#include "Tutorial.hpp"
#include "Helpers.hpp"
#include "VK.hpp"


static uint32_t vert_code[] = 
#include "spv/objects.vert.inl"
;

static uint32_t frag_code[] = 
#include "spv/objects.frag.inl"
;

void Tutorial::ObjectsPipeline::create(RTG &rtg, VkRenderPass render_pass, uint32_t subpass){
    VkShaderModule vert_module = rtg.helpers.create_shader_module(vert_code);
    VkShaderModule frag_module = rtg.helpers.create_shader_module(frag_code);

    { //the set0_World layout holds world info in a uniform buffer used in the fragment shader:
		std::array< VkDescriptorSetLayoutBinding, 1 > bindings{
			VkDescriptorSetLayoutBinding{
				.binding = 0,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT
			},
		};
		
		VkDescriptorSetLayoutCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = uint32_t(bindings.size()),
			.pBindings = bindings.data(),
		};

		VK( vkCreateDescriptorSetLayout(rtg.device, &create_info, nullptr, &set0_World) );
	}

    {//the set1_Transform layout holds an array of Transform structures in a storage buffer used in the vertex shader:
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
        VK(vkCreateDescriptorSetLayout(rtg.device, &create_info, nullptr, &set1_Transforms));
    }

    {//the set2_TEXTURE layout has a single descriptor for a sample2D used in the fragment shader
        std::array<VkDescriptorSetLayoutBinding, 1> bindings{
            VkDescriptorSetLayoutBinding{
                .binding = 0,
                .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .descriptorCount = 1,
                .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
            }
        };

        VkDescriptorSetLayoutCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = uint32_t(bindings.size()),
            .pBindings = bindings.data(),
        };
        VK(vkCreateDescriptorSetLayout(rtg.device, &create_info, nullptr, &set2_TEXTURE));
    }    

    {//create pipeline layout
        std::array< VkDescriptorSetLayout, 3 > layouts{
			set0_World,
            set1_Transforms,
            set2_TEXTURE,
		};

        VkPipelineLayoutCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = uint32_t(layouts.size()),
            .pSetLayouts = layouts.data(),
            .pushConstantRangeCount = 0,
            .pPushConstantRanges = nullptr,
        };

        VK(vkCreatePipelineLayout(rtg.device, &create_info, nullptr, &layout));
    }

    {//create pipeline
        std::array<VkPipelineShaderStageCreateInfo, 2> stages{
            VkPipelineShaderStageCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage =VK_SHADER_STAGE_VERTEX_BIT,
                .module = vert_module,
                .pName = "main",
            },
            VkPipelineShaderStageCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage =VK_SHADER_STAGE_FRAGMENT_BIT,
                .module = frag_module,
                .pName = "main",
            },            
        };
    //dynamic states (viewport and scissor will be set dynamically through state commands)
        std::vector<VkDynamicState> dynamic_states{
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR,
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
    //configure rasterizer to do back face culling
        VkPipelineRasterizationStateCreateInfo rasterization_state{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable = VK_FALSE,
            .rasterizerDiscardEnable = VK_FALSE,
            .polygonMode = VK_POLYGON_MODE_FILL,
            .cullMode = VK_CULL_MODE_BACK_BIT,
            .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
            .depthBiasEnable = VK_FALSE,
            .lineWidth = 1.0f,
        };
    //disable multisampling
        VkPipelineMultisampleStateCreateInfo multisample_state{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			.sampleShadingEnable = VK_FALSE,
		};
    //disable depth test for now
        VkPipelineDepthStencilStateCreateInfo depth_stencil_state{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = VK_TRUE,
            .depthWriteEnable = VK_TRUE,
            .depthCompareOp = VK_COMPARE_OP_LESS,
			.depthBoundsTestEnable = VK_FALSE,
			.stencilTestEnable = VK_FALSE,
		};
    //color attatchment
        std::array<VkPipelineColorBlendAttachmentState, 1> attachment_states{
            VkPipelineColorBlendAttachmentState{
				.blendEnable = VK_FALSE,
				.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
			},
        };
        VkPipelineColorBlendStateCreateInfo color_blend_state{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
			.logicOpEnable = VK_FALSE,
			.attachmentCount = uint32_t(attachment_states.size()),
			.pAttachments = attachment_states.data(),
			.blendConstants{0.0f, 0.0f, 0.0f, 0.0f},
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
			.pColorBlendState = &color_blend_state,
			.pDynamicState = &dynamic_state,
			.layout = layout,
			.renderPass = render_pass,
			.subpass = subpass,
        };
        VK(vkCreateGraphicsPipelines(rtg.device, VK_NULL_HANDLE, 1, &create_info, nullptr, &handle));
    }
    //deallocate shader modules now that they are no longer used
    vkDestroyShaderModule(rtg.device, vert_module, nullptr);
    vkDestroyShaderModule(rtg.device, frag_module, nullptr);
}

void Tutorial::ObjectsPipeline::destroy(RTG &rtg){
    if(handle != VK_NULL_HANDLE){
        vkDestroyPipeline(rtg.device, handle, nullptr);
        handle = VK_NULL_HANDLE;
    }
    if(layout != VK_NULL_HANDLE){
        vkDestroyPipelineLayout(rtg.device, layout, nullptr);
        layout = VK_NULL_HANDLE;
    }
    if (set0_World != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(rtg.device, set0_World, nullptr);
		set0_World = VK_NULL_HANDLE;
	}
    if (set1_Transforms != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(rtg.device, set1_Transforms, nullptr);
		set1_Transforms = VK_NULL_HANDLE;
	}
    if (set2_TEXTURE != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(rtg.device, set2_TEXTURE, nullptr);
		set2_TEXTURE = VK_NULL_HANDLE;
	}

}