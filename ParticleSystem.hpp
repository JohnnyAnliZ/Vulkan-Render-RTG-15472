#pragma once
#include "RTG.hpp"
#include "Helpers.hpp"
#include "ComputeSystem.hpp"
#include "mat4.hpp"

struct ParticleComputePipeline {
    VkDescriptorSetLayout set0_layout = VK_NULL_HANDLE;
    struct Push {
        uint32_t num_particles;
        float    dt;
        uint32_t N;
        uint32_t frame;
    };
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline       handle = VK_NULL_HANDLE;
    void create(RTG &);
    void destroy(RTG &);
};

struct ParticleRenderPipeline {
    struct Push {
        mat4     CLIP_FROM_WORLD;     // 64 bytes
        vec4     volume_center_cell;  // xyz=center WS, w=cell_size_ws  (16 bytes)
        uint32_t N;                   // 4 bytes — total 84 bytes
    };
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline       handle = VK_NULL_HANDLE;
    void create(RTG &, VkRenderPass render_pass, uint32_t subpass);
    void destroy(RTG &);
};

struct ParticleSystem {
    static constexpr uint32_t NUM_PARTICLES = 200'000;

    void init(RTG &, ComputeContext &, VkDescriptorPool particle_pool,
              VkImageView velocity_views[2]);
    void destroy(RTG &);

    void update(ComputeContext &, float dt, uint32_t velocity_ind, uint32_t frame);
    void draw(VkCommandBuffer, mat4 const &CLIP_FROM_WORLD,
              vec3 volume_center, float cell_size_ws, uint32_t N);

    Helpers::AllocatedBuffer particle_buffer;
    VkSampler                linear_sampler  = VK_NULL_HANDLE;
    VkDescriptorSet          compute_sets[2] = {};

    ParticleComputePipeline compute_pipeline;
    ParticleRenderPipeline  render_pipeline;
};
