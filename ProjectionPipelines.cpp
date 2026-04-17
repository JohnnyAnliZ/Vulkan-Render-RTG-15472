#include "Tutorial.hpp"
#include "Helpers.hpp"
#include "VK.hpp"



static uint32_t div_code[] = 
#include "spv/divergence.comp.inl"
;

void DivergencePipeline::create(RTG &rtg){
    VkShaderModule comp_module = rtg.helpers.create_shader_module(div_code);
    { //velocity
		std::array< VkDescriptorSetLayoutBinding, 2 > bindings{
			VkDescriptorSetLayoutBinding{
				.binding = 0,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
			},
            VkDescriptorSetLayoutBinding{
				.binding = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
			}
		};
		
		VkDescriptorSetLayoutCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = uint32_t(bindings.size()),
			.pBindings = bindings.data(),
		};

		VK( vkCreateDescriptorSetLayout(rtg.device, &create_info, nullptr, &set0_velocity_volume) );
	}
    
    { //pressure
		std::array< VkDescriptorSetLayoutBinding, 2 > bindings{
			VkDescriptorSetLayoutBinding{
				.binding = 0,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
			},
            VkDescriptorSetLayoutBinding{
				.binding = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
			},
		};
		
		VkDescriptorSetLayoutCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = uint32_t(bindings.size()),
			.pBindings = bindings.data(),
		};

		VK( vkCreateDescriptorSetLayout(rtg.device, &create_info, nullptr, &set1_pressure_volume) );
	}

    { //divergence
		std::array< VkDescriptorSetLayoutBinding, 1 > bindings{
			VkDescriptorSetLayoutBinding{
				.binding = 0,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
			},
		};
		
		VkDescriptorSetLayoutCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = uint32_t(bindings.size()),
			.pBindings = bindings.data(),
		};

		VK( vkCreateDescriptorSetLayout(rtg.device, &create_info, nullptr, &set2_divergence_volume) );
	}

    {//create pipeline layout
        VkPushConstantRange range{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(Push),
        };
        std::array< VkDescriptorSetLayout, 3 > layouts{
			set0_velocity_volume,
            set1_pressure_volume,
            set2_divergence_volume,
		};

        VkPipelineLayoutCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = (uint32_t)layouts.size(),
            .pSetLayouts = layouts.data(),
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &range
        };

        VK(vkCreatePipelineLayout(rtg.device, &create_info, nullptr, &layout));
    }

    {//create pipeline

        VkPipelineShaderStageCreateInfo stage{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage =VK_SHADER_STAGE_COMPUTE_BIT,
            .module = comp_module,
            .pName = "main",
        };

    //create da pipeline
        VkComputePipelineCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .flags = 0,
            .stage = stage,
			.layout = layout,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1,
        };
        VK(vkCreateComputePipelines(rtg.device, VK_NULL_HANDLE, 1, &create_info, nullptr, &handle));
    }

    vkDestroyShaderModule(rtg.device, comp_module, nullptr);
}

void DivergencePipeline::destroy(RTG &rtg){
    if(layout != VK_NULL_HANDLE){
        vkDestroyPipelineLayout(rtg.device, layout, nullptr);
        layout = VK_NULL_HANDLE;
    }
    if(handle != VK_NULL_HANDLE){
        vkDestroyPipeline(rtg.device, handle, nullptr);
        handle = VK_NULL_HANDLE;
    }
    if (set0_velocity_volume != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(rtg.device, set0_velocity_volume, nullptr);
		set0_velocity_volume = VK_NULL_HANDLE;
	}
    if (set1_pressure_volume != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(rtg.device, set1_pressure_volume, nullptr);
		set1_pressure_volume = VK_NULL_HANDLE;
	}
    if (set2_divergence_volume != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(rtg.device, set2_divergence_volume, nullptr);
		set2_divergence_volume = VK_NULL_HANDLE;
	}
    
}


static uint32_t pressure_solve_code[] = 
#include "spv/pressureSolve.comp.inl"
;

void PressureSolvePipeline::create(RTG &rtg){
    VkShaderModule comp_module = rtg.helpers.create_shader_module(pressure_solve_code);
    { //divergence
		std::array< VkDescriptorSetLayoutBinding, 1 > bindings{
			VkDescriptorSetLayoutBinding{
				.binding = 0,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
			},
		};
		
		VkDescriptorSetLayoutCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = uint32_t(bindings.size()),
			.pBindings = bindings.data(),
		};

		VK( vkCreateDescriptorSetLayout(rtg.device, &create_info, nullptr, &set0_divergence_volume) );
	}
    
    { //pressure
		std::array< VkDescriptorSetLayoutBinding, 2 > bindings{
			VkDescriptorSetLayoutBinding{
				.binding = 0,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
			},
            VkDescriptorSetLayoutBinding{
				.binding = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
			},
		};
		
		VkDescriptorSetLayoutCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = uint32_t(bindings.size()),
			.pBindings = bindings.data(),
		};

		VK( vkCreateDescriptorSetLayout(rtg.device, &create_info, nullptr, &set1_pressure_volume) );
	}



    {//create pipeline layout
        VkPushConstantRange range{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(Push),
        };
        std::array< VkDescriptorSetLayout, 2 > layouts{
			set0_divergence_volume,
            set1_pressure_volume,            
		};

        VkPipelineLayoutCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = (uint32_t)layouts.size(),
            .pSetLayouts = layouts.data(),
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &range
        };

        VK(vkCreatePipelineLayout(rtg.device, &create_info, nullptr, &layout));
    }

    {//create pipeline

        VkPipelineShaderStageCreateInfo stage{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage =VK_SHADER_STAGE_COMPUTE_BIT,
            .module = comp_module,
            .pName = "main",
        };

    //create da pipeline
        VkComputePipelineCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .flags = 0,
            .stage = stage,
			.layout = layout,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1,
        };
        VK(vkCreateComputePipelines(rtg.device, VK_NULL_HANDLE, 1, &create_info, nullptr, &handle));
    }

    vkDestroyShaderModule(rtg.device, comp_module, nullptr);
}

void PressureSolvePipeline::destroy(RTG &rtg){
    if(layout != VK_NULL_HANDLE){
        vkDestroyPipelineLayout(rtg.device, layout, nullptr);
        layout = VK_NULL_HANDLE;
    }
    if(handle != VK_NULL_HANDLE){
        vkDestroyPipeline(rtg.device, handle, nullptr);
        handle = VK_NULL_HANDLE;
    }
    if (set0_divergence_volume != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(rtg.device, set0_divergence_volume, nullptr);
		set0_divergence_volume = VK_NULL_HANDLE;
	}
    if (set1_pressure_volume != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(rtg.device, set1_pressure_volume, nullptr);
		set1_pressure_volume = VK_NULL_HANDLE;
	}

}

static uint32_t grad_sub_code[] = 
#include "spv/gradientSubtract.comp.inl"
;

void GradientSubtractPipeline::create(RTG &rtg){
    VkShaderModule comp_module = rtg.helpers.create_shader_module(grad_sub_code);
    { //velocity
		std::array< VkDescriptorSetLayoutBinding, 2 > bindings{
			VkDescriptorSetLayoutBinding{
				.binding = 0,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
			},
            VkDescriptorSetLayoutBinding{
				.binding = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
			}
		};
		
		VkDescriptorSetLayoutCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = uint32_t(bindings.size()),
			.pBindings = bindings.data(),
		};

		VK( vkCreateDescriptorSetLayout(rtg.device, &create_info, nullptr, &set0_velocity_volume) );
	}
    
    { //pressure
		std::array< VkDescriptorSetLayoutBinding, 2 > bindings{
			VkDescriptorSetLayoutBinding{
				.binding = 0,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
			},
            VkDescriptorSetLayoutBinding{
				.binding = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT
			},
		};
		
		VkDescriptorSetLayoutCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
			.bindingCount = uint32_t(bindings.size()),
			.pBindings = bindings.data(),
		};

		VK( vkCreateDescriptorSetLayout(rtg.device, &create_info, nullptr, &set1_pressure_volume) );
	}


    {//create pipeline layout
        VkPushConstantRange range{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(Push),
        };
        std::array< VkDescriptorSetLayout, 2 > layouts{
			set0_velocity_volume,
            set1_pressure_volume,
		};

        VkPipelineLayoutCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = (uint32_t)layouts.size(),
            .pSetLayouts = layouts.data(),
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &range
        };

        VK(vkCreatePipelineLayout(rtg.device, &create_info, nullptr, &layout));
    }

    {//create pipeline

        VkPipelineShaderStageCreateInfo stage{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage =VK_SHADER_STAGE_COMPUTE_BIT,
            .module = comp_module,
            .pName = "main",
        };

    //create da pipeline
        VkComputePipelineCreateInfo create_info{
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .flags = 0,
            .stage = stage,
			.layout = layout,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex = -1,
        };
        VK(vkCreateComputePipelines(rtg.device, VK_NULL_HANDLE, 1, &create_info, nullptr, &handle));
    }

    vkDestroyShaderModule(rtg.device, comp_module, nullptr);
}

void GradientSubtractPipeline::destroy(RTG &rtg){
    if(layout != VK_NULL_HANDLE){
        vkDestroyPipelineLayout(rtg.device, layout, nullptr);
        layout = VK_NULL_HANDLE;
    }
    if(handle != VK_NULL_HANDLE){
        vkDestroyPipeline(rtg.device, handle, nullptr);
        handle = VK_NULL_HANDLE;
    }
    if (set0_velocity_volume != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(rtg.device, set0_velocity_volume, nullptr);
		set0_velocity_volume = VK_NULL_HANDLE;
	}
    if (set1_pressure_volume != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(rtg.device, set1_pressure_volume, nullptr);
		set1_pressure_volume = VK_NULL_HANDLE;
	}
    
}