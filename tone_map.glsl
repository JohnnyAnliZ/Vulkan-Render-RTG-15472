#ifndef TONE_MAP_GLSL
#define TONE_MAP_GLSL



vec3 tone_map_reinhard(vec3 c){
    return c / (1.0 + c);
}

vec3 apply_tone_map(vec3 c, int mode){
    //linear
    if(mode == 0) return c;
    else if(mode == 1) return tone_map_reinhard(c);
    return c;
}

#endif