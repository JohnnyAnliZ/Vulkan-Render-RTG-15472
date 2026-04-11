#version 450

layout(set = 0, binding = 0) uniform sampler2D SHADOW_ATLAS;
layout(set = 1, binding = 0) uniform sampler3D VELOCITY_VOL;

layout(location = 0) in vec2 position;
layout(location = 0) out vec4 outColor;

void main(){
    //TODO: put shadow_atlas somewhere on the screen, and slices of VELOCITY_VOL
    
}