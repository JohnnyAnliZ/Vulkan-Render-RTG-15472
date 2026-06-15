#version 450
layout(location = 0) in vec4 in_pos_life;

layout(push_constant) uniform Push {
    mat4     CLIP_FROM_WORLD;
    vec4     volume_center_cell;
    uint     N;
} pc;

layout(location = 0) out float out_life;

void main() {
    float Nf         = float(pc.N);
    vec3  vol_center = pc.volume_center_cell.xyz;
    float cell_size  = pc.volume_center_cell.w;
    vec3  ws = vol_center + (in_pos_life.xyz - Nf * 0.5) * (cell_size / Nf);
    gl_Position  = pc.CLIP_FROM_WORLD * vec4(ws, 1.0);
    gl_PointSize = 2.0;
    out_life = in_pos_life.w;
}
