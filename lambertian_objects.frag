#version 450
#include "tone_map.glsl"
#include "lights.glsl"

layout(push_constant) uniform Push {
    uint light_count;
}pc;

layout(set = 0, binding=0, std140) readonly buffer Lights{
    Light LIGHTS[];
};

layout(set=2,binding=0) uniform sampler2D NORMAL;
layout(set=2,binding=1) uniform sampler2D TEXTURE;
layout(set=2,binding=2) uniform samplerCube DIFFUSE_IRRADIANCE;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 tangent;
layout(location = 3) in vec2 texCoord;

layout(location = 0) out vec4 outColor;

void main(){
    vec3 B = cross(normal, tangent.xyz) * tangent.w;
    mat3 TBN = mat3(tangent.xyz, B, normal);


    //sample normal map to get normal in tangent space
    vec3 n_tangent = texture(NORMAL, vec2(texCoord.y,-texCoord.x)).rgb * 2.0 - 1.0;
    
    vec3 n = normalize(TBN * n_tangent);

    //direction that the light is from(this should point to -z)
    vec3 albedo = texture(TEXTURE, vec2(texCoord.y,-texCoord.x)).rgb;

    
    vec3 irradiance_from_env = texture(DIFFUSE_IRRADIANCE, n).rgb;


    vec3 irradiance_from_lights = vec3(0);
    for(uint i = 0; i < pc.light_count; i++){//go through the lights to add up irradiance
        irradiance_from_lights += diffuse_lighting_irradiance(LIGHTS[i], position, n);
    }


    vec3 diffuse = (albedo / 3.14159265354 * (irradiance_from_env + irradiance_from_lights));
    outColor = vec4(diffuse, 1.0);
}