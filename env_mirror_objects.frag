#version 450


layout(push_constant) uniform Push {
	bool is_env;
};

layout(set=0,binding=0,std140) uniform Eye {
	vec3 EYE;
};

layout(set=2,binding=0) uniform samplerCube TEXTURE;

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 tangent;
layout(location = 3) in vec2 texCoord;


layout(location = 0) out vec4 outColor;

void main(){
    vec3 n = normalize(normal);
    if(is_env){
        outColor = texture(TEXTURE, n);
    }
    else{     
        vec3 i = position - EYE; //incident vector pointing toward the surface, per glsl reflect()'s requirement
       
        vec3 reflection = reflect(i, n); //get the reflection vector
        outColor = texture(TEXTURE, reflection);
    }
}