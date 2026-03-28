#include "Tutorial.hpp"
#include "Helpers.hpp"
#include "VK.hpp"


static uint32_t vert_code[] = 
#include "spv/shadows2D.vert.inl"
;

static uint32_t frag_code[] = 
#include "spv/shadows2D.frag.inl"
;


void Tutorial::init_shadow_mapping(){
    { //create shadow pass
		//specify attachments
		std::array< VkAttachmentDescription, 1 > attachments{
			VkAttachmentDescription{ //0 - depth attachment:
				.format = VK_FORMAT_D32_SFLOAT,
				.samples = VK_SAMPLE_COUNT_1_BIT,
				.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
				.storeOp = VK_ATTACHMENT_STORE_OP_STORE,
				.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
				.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
				.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
				.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
			},
		};
		//subpass
		VkAttachmentReference depth_attachment_ref{
			.attachment = 0,
			.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
		};

		VkSubpassDescription subpass{
			.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
			.inputAttachmentCount = 0,
			.pInputAttachments = nullptr,
			.colorAttachmentCount = 0,
			.pColorAttachments = nullptr,
			.pDepthStencilAttachment = &depth_attachment_ref,
		};

		//dependencies
		std::array< VkSubpassDependency, 2 > dependencies {
			// BEFORE render pass (read → write)
            VkSubpassDependency{
                .srcSubpass = VK_SUBPASS_EXTERNAL,
                .dstSubpass = 0,
                .srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                .srcAccessMask = VK_ACCESS_SHADER_READ_BIT,
                .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            },

            // AFTER render pass (write → read)
            VkSubpassDependency{
                .srcSubpass = 0,
                .dstSubpass = VK_SUBPASS_EXTERNAL,
                .srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                .dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                .srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
            }
		};

		VkRenderPassCreateInfo create_info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
			.attachmentCount = uint32_t(attachments.size()),
			.pAttachments = attachments.data(),
			.subpassCount = 1,
			.pSubpasses = &subpass,
			.dependencyCount = uint32_t(dependencies.size()),
			.pDependencies = dependencies.data(),
		};
		VK( vkCreateRenderPass(rtg.device, &create_info, nullptr, &shadow_pass) );
	}	

	shadow_2D_pipeline.create(rtg, shadow_pass, 0);
    std::cout<<"created shadow pass pipeline"<<std::endl;

	{//go through the lights to count the lights and get a suitable shadow atlas size
		std::deque<Item> current_nodes;
		for(auto n : scene72.scene.roots){//start with the root nodes
			current_nodes.emplace_back(Item{n,mat4::identity()});
		}
		while(!current_nodes.empty()){//go through the graph using this queue bfs setup (this creates two instances of a child if two nodes have it as one of the children)
			auto [node, world_from_parent] = current_nodes.front();
			current_nodes.pop_front();

			//this node's world transform
			mat4 world_from_local = world_from_parent * node->parent_from_local();//accumulate transform
	
			if(node->light != nullptr){
				std::variant< S72::Light::Sun, S72::Light::Sphere, S72::Light::Spot > &v = node->light->source;

				uint32_t faces = 0;
				if(std::holds_alternative<S72::Light::Sun>(v)){
					S72::Light::Sun &sun = get<S72::Light::Sun>(v);
					faces = sun.shadow_map_num;
				}
				else if(std::holds_alternative<S72::Light::Sphere>(v)){
					S72::Light::Sphere &sphere = get<S72::Light::Sphere>(v);
					faces = sphere.shadow_map_num;
				}
				else if(std::holds_alternative<S72::Light::Spot>(v)){
					S72::Light::Spot &spot = get<S72::Light::Spot>(v);
					faces = spot.shadow_map_num;
				}

				uint32_t shadow_map_size = node->light->shadow;
				light_shadow_map_sizes.emplace_back(shadow_map_size);
				total_shadow_map_size += shadow_map_size * shadow_map_size * faces;	
			}
			for(S72::Node *child : node->children){
				current_nodes.emplace_back(child, world_from_local);
			}
		}
		//find the fitting atlas size
		for(uint32_t power = 1; power < 14; power++){//2 to the power of 12 ( 16384 side lenght ) should be large enough
			if(((uint32_t) 1 << power) * ((uint32_t) 1 << power) > total_shadow_map_size){
				atlas_size = 1<<power;
				break;
			}
		}
	}


    for (Workspace &workspace : workspaces){
        {//create image for shadow atlas
            workspace.Shadow_Atlas = rtg.helpers.create_image(
                VkExtent2D{.width = atlas_size, .height = atlas_size},
                VK_FORMAT_D32_SFLOAT, 
                VK_IMAGE_TILING_OPTIMAL,
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            );
        }
    
        {//create image view for it 
            VkImageViewCreateInfo view_info{
                .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                .image = workspace.Shadow_Atlas.handle,
                .viewType = VK_IMAGE_VIEW_TYPE_2D,
                .format = VK_FORMAT_D32_SFLOAT,
                .subresourceRange = {
                    VK_IMAGE_ASPECT_DEPTH_BIT,
                    0, 1,
                    0, 1
                }
            };
            VK( vkCreateImageView(rtg.device, &view_info, nullptr, &workspace.Shadow_Atlas_view));
        }

        {//framebuffer 
            VkFramebufferCreateInfo fb_info{
                .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                .renderPass = shadow_pass,
                .attachmentCount = 1,
                .pAttachments = &workspace.Shadow_Atlas_view,
                .width = atlas_size,
                .height = atlas_size,
                .layers = 1
            };
            VK(vkCreateFramebuffer(rtg.device, &fb_info, nullptr, &workspace.Shadow_Atlas_FB));
        }
        {
            workspace.debug_buffer = rtg.helpers.create_buffer(workspace.Shadow_Atlas.allocation.size,
                VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT|VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, Helpers::Mapped
            );
        }

        {//create sampler
            VkSamplerCreateInfo sampler_info{
                .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                .magFilter = VK_FILTER_LINEAR,
                .minFilter = VK_FILTER_LINEAR,               
                .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                .compareEnable = VK_TRUE,
                .compareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
            };
            VK( vkCreateSampler(rtg.device, &sampler_info, nullptr, &depth_texture_sampler) );
        }

		{//allocate descriptor set for Shadow Atlas descriptor
			VkDescriptorSetAllocateInfo alloc_info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = texture_descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &lambertian_objects_pipeline.set2_Shadows,
			};
			VK(vkAllocateDescriptorSets(rtg.device, &alloc_info, &workspace.Shadow_Atlas_descriptors));
		}

		
        {//update the descriptor set for this mtfk
            VkDescriptorImageInfo Shadow_Atlas_info{
				.sampler = depth_texture_sampler,
                .imageView = workspace.Shadow_Atlas_view,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
			};
			std::array<VkWriteDescriptorSet, 1> writes{
				VkWriteDescriptorSet{
					.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
					.dstSet = workspace.Shadow_Atlas_descriptors,
					.dstBinding = 0,
					.dstArrayElement = 0,
					.descriptorCount = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.pImageInfo = &Shadow_Atlas_info,
				},
			};

			vkUpdateDescriptorSets(
				rtg.device, //device
				uint32_t(writes.size()), //descriptorWriteCount
				writes.data(), //pDescriptorWrites
				0, //descriptorCopyCount
				nullptr //pDescriptorCopies
			);
        }

    }
}

void Tutorial::draw_all_objects(VkCommandBuffer const &cmd, mat4 const &LIGHTS_CLIP_FROM_WORLD, vec4 const &_shadow_atlas){
    //push constants
    Shadow2DPipeline::Push push{
        .LIGHT_CLIP_FROM_WORLD = LIGHTS_CLIP_FROM_WORLD,
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
    {//depth bias
        vkCmdSetDepthBias(cmd, 1.75f, 0.0f, 1.75f);
    }
    
    //literally draw all the objects
    uint32_t index_offset = 0;
    if(!lambertian_object_instances.empty()){
        for (LambertianObjectInstance const &inst : lambertian_object_instances) {
            uint32_t index = uint32_t(&inst - &lambertian_object_instances[0]);

            vkCmdDraw(cmd, inst.vertices.count, 1, inst.vertices.first, (index+index_offset));
        }
    }
    index_offset = (uint32_t)lambertian_object_instances.size();
    
    if(!env_mirror_object_instances.empty()){
        for (EnvMirrorObjectInstance const &inst : env_mirror_object_instances) {
            uint32_t index = uint32_t(&inst - &env_mirror_object_instances[0]);

            vkCmdDraw(cmd, inst.vertices.count, 1, inst.vertices.first, (index+index_offset));
        }
    }
    index_offset += (uint32_t)env_mirror_object_instances.size();

    if(!pbr_object_instances.empty()){
        for (PbrObjectInstance const &inst : pbr_object_instances) {
            uint32_t index = uint32_t(&inst - &pbr_object_instances[0]);

            vkCmdDraw(cmd, inst.vertices.count, 1, inst.vertices.first, (index+index_offset));
        }
    }
}


void Tutorial::Shadow2DPipeline::create(RTG &rtg, VkRenderPass render_pass, uint32_t subpass){
    VkShaderModule vert_module = rtg.helpers.create_shader_module(vert_code);
    VkShaderModule frag_module = rtg.helpers.create_shader_module(frag_code);

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
            .depthCompareOp = VK_COMPARE_OP_LESS,
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