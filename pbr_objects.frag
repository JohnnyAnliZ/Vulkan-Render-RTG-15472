#version 450
#include "tone_map.glsl"


layout(push_constant) uniform Push {
    float exposure;
    int toneMapMode;
}pc;


layout(set=0,binding=0,std140) uniform Eye {
	vec3 EYE;
};

layout(set=2,binding=0) uniform sampler2D ALBEDO;
layout(set=2,binding=1) uniform sampler2D ROUGHNESS;
layout(set=2,binding=2) uniform sampler2D METALNESS;
layout(set=2,binding=3) uniform samplerCube ENVIRONMENT;    
layout(set=2,binding=4) uniform sampler2D BRDF_LUT;
layout(set=2,binding=5) uniform samplerCube DIFFUSE_IRRADIANCE;



layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 tangent;
layout(location = 3) in vec2 texCoord;


layout(location = 0) out vec4 outColor;



void main(){
    vec3 N = normalize(normal);
    vec3 V = normalize(EYE - position);

    vec3 albedo = texture(ALBEDO, texCoord).rgb;
    float roughness = texture(ROUGHNESS, texCoord).r;
    float metalness = texture(METALNESS, texCoord).r;

    // Base reflectivity
    vec3 F0 = mix(vec3(0.04), albedo, metalness);

    // === Diffuse IBL ===
    vec3 irradiance = texture(DIFFUSE_IRRADIANCE, N).rgb;
    vec3 diffuse = irradiance * albedo * (1.0 - metalness);

    // === Specular IBL ===
    vec3 R = reflect(-V, N);

    float NoV = max(dot(N, V), 0.0);
    vec3 prefiltered = textureLod(ENVIRONMENT, R, roughness * 4.0).rgb;
    vec2 brdf = texture(BRDF_LUT, vec2(NoV, roughness)).rg;

    vec3 specular = prefiltered * (F0 * brdf.x + brdf.y);

    vec3 color = diffuse + specular;

    //tone mapping
    vec3 radiance =  color * pow(2.0, pc.exposure);  // common exposure model
    vec3 mapped = apply_tone_map(radiance, pc.toneMapMode);
    outColor = vec4(mapped, 1.0);
}