#version 450
#include "tone_map.glsl"


layout(push_constant) uniform Push {
	bool is_env;
    float exposure;
    int toneMapMode;
}pc;

layout(set=0,binding=0,std140) uniform Eye {
	vec3 EYE;
};

layout(set=2,binding=0) uniform sampler2D NORMAL;
layout(set=2,binding=1) uniform samplerCube TEXTURE;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 tangent;
layout(location = 3) in vec2 texCoord;

layout(location = 0) out vec4 outColor;

void main(){

    vec3 B = cross(normal, tangent.xyz) * tangent.w;
    mat3 TBN = mat3(tangent.xyz, B, normal);

    //sample normal map to get normal in tangent space
    vec3 n_tangent = texture(NORMAL, texCoord).rgb * 2.0 - 1.0;

    vec3 n = normalize(TBN * n_tangent);


    vec3 radiance = vec3(0.0);
    if(pc.is_env){
        radiance = textureLod(TEXTURE, n, 0.0).rgb;
    }
    else{     
        vec3 i = position - EYE; //incident vector pointing toward the surface, per glsl reflect()'s requirement

        vec3 reflection = reflect(i, n); //get the reflection vector
        radiance = textureLod(TEXTURE, reflection, 0.0).rgb;
    }

    //tone mapping
    radiance *= pow(2.0, pc.exposure);  // common exposure model
    vec3 mapped = apply_tone_map(radiance, pc.toneMapMode);
    outColor = vec4(mapped, 1.0);
}