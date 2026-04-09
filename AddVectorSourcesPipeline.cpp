#include "Tutorial.hpp"
#include "Helpers.hpp"
#include "VK.hpp"



static uint32_t comp_code[] = 
#include "spv/addVectorSources.comp.inl"
;

void AddVectorSourcesPipeline::create(RTG &rtg, VkRenderPass render_pass, uint32_t subpass){
    VkShaderModule comp_module = rtg.helpers.create_shader_module(comp_module);


    { //the set0_vector_volume layout holds the vector_volume
		std::array< VkDescriptorSetLayoutBinding, 1 > bindings{
			VkDescriptorSetLayoutBinding{
				.binding = 0,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
			},
		};
		
		VkDescriptorSetLayoutCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = uint32_t(bindings.size()),
			.pBindings = bindings.data(),
		};

		VK( vkCreateDescriptorSetLayout(rtg.device, &create_info, nullptr, &set0_vector_volume) );
	}

    {//create pipeline layout
        VkPushConstantRange range{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
            .offset = 0,
            .size = sizeof(Push),
        };
        std::array< VkDescriptorSetLayout, 4 > layouts{
			set0_vector_volume
		};

        VkPipelineLayoutCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 0,
            .pSetLayouts = layouts.data(),
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &range
        };

        VK(vkCreatePipelineLayout(rtg.device, &create_info, nullptr, &layout));
    }

    {//create pipeline
        std::array<VkPipelineShaderStageCreateInfo, 1> stages{
            VkPipelineShaderStageCreateInfo{
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage =VK_SHADER_STAGE_COMPUTE_BIT,
                .module = comp_module,
                .pName = "main",
            },            
        };

    //create da pipeline
        VkGraphicsPipelineCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = (uint32_t)stages.size(),
            .pStages = stages.data(),
			.layout = layout,
			.renderPass = render_pass,
			.subpass = subpass,
        };
        VK(vkCreateGraphicsPipelines(rtg.device, VK_NULL_HANDLE, 1, &create_info, nullptr, &handle));
    }

    vkDestroyShaderModule(rtg.device, comp_module, nullptr);
}

void AddVectorSourcesPipeline::destroy(RTG &rtg){
    if(layout != VK_NULL_HANDLE){
        vkDestroyPipelineLayout(rtg.device, layout, nullptr);
        layout = VK_NULL_HANDLE;
    }
    if(handle != VK_NULL_HANDLE){
        vkDestroyPipeline(rtg.device, handle, nullptr);
        handle = VK_NULL_HANDLE;
    }
    if (set0_vector_volume != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(rtg.device, set0_vector_volume, nullptr);
		set0_vector_volume = VK_NULL_HANDLE;
	}
}