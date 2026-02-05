#version 450

layout(push_constant) uniform Push {
	float time;
};
layout(location = 0) in vec2 position;
layout(location = 0) out vec4 outColor;

float random (vec2 st) {
    return fract(sin(dot(st.xy,
                         vec2(12.9898,78.233)))*
        43758.5453123);
}

void main(){

	float cycle = (sin(6.28 * time / 5.0) + 0.8) / 1.8;
    
    vec3 daySky = vec3(0.53, 0.81, 0.92);
    vec3 nightSky = vec3(0.01, 0.01, 0.03);
    
    vec3 finalColor = mix(nightSky, daySky, cycle);
    
    outColor = vec4(finalColor, 1.0);
}


