#include "cube_utilities.hpp"

float cube_texel_solid_angle(uint32_t x, uint32_t y, uint32_t width){
    float inv = 1.0f / width;

    float u = (2.0f * (x + 0.5f) * inv) - 1.0f;
    float v = (2.0f * (y + 0.5f) * inv) - 1.0f;

    float texel_area = 4.0f * inv * inv;  // = 4 / width^2

    float jacobian = 1.0f / pow(1.0f + u*u + v*v, 1.5f);

    return texel_area * jacobian;
}

vec3 cube_texel_direction(uint32_t face, uint32_t x, uint32_t y, uint32_t width){
    vec3 dir;
    float u = (2.0f * (x + 0.5f) / width) - 1.0f;
    float v = (2.0f * (y + 0.5f) / width) - 1.0f;
    v = -v;//following assignments assume uv is bottom-left origin per cube map face
    assert(face < 6);
    switch(face){
        case 0: // +X (right)
            dir = normalized(vec3( 1.0f,  v, -u));
            break;

        case 1: // -X (left)
            dir = normalized(vec3(-1.0f,  v,  u));
            break;

        case 2: // +Y (front)
            dir = normalized(vec3( u, 1.0f, -v));
            break;

        case 3: // -Y (back)
            dir = normalized(vec3( u, -1.0f,  v));
            break;

        case 4: // +Z (top)
            dir = normalized(vec3( u,  v, 1.0f));
            break;

        case 5: // -Z (bottom)
            dir = normalized(vec3( u, -v, -1.0f));
            break;
    }
    return dir;
}


void direction_to_cube_texel(vec3 d, uint32_t width, uint32_t &face, uint32_t &x, uint32_t &y){
    d = normalized(d);
    vec3 a = vec3{abs(d.x), abs(d.y), abs(d.z)};

    float u, v;

    if (a.x >= a.y && a.x >= a.z)
    {
        if (d.x > 0) {
            face = 0;
            u = -d.z / a.x;
            v =  d.y / a.x;
        } else {
            face = 1;
            u =  d.z / a.x;
            v =  d.y / a.x;
        }
    }
    else if (a.y >= a.x && a.y >= a.z)
    {
        if (d.y > 0) {
            face = 2;
            u =  d.x / a.y;
            v = -d.z / a.y;
        } else {
            face = 3;
            u =  d.x / a.y;
            v =  d.z / a.y;
        }
    }
    else
    {
        if (d.z > 0) {
            face = 4;
            u =  d.x / a.z;
            v =  d.y / a.z;
        } else {
            face = 5;
            u =  d.x / a.z;
            v = -d.y / a.z;
        }
    }

    v = -v;

    float fx = (u + 1.0f) * 0.5f * width - 0.5f;
    float fy = (v + 1.0f) * 0.5f * width - 0.5f;

    x = std::clamp(uint32_t(fx), 0u, width - 1);
    y = std::clamp(uint32_t(fy), 0u, width - 1);
}

void convolve_cubemap_diffuse(uint32_t in_width, std::vector<char> const &in_cube, uint32_t out_width, std::vector<char> &out_cube){
    

    std::vector<vec3> in_cube_radiance_values;
    in_cube_radiance_values.resize(in_width * in_width * 6);
    rgbe_to_radiance_values(in_cube, in_cube_radiance_values, in_width);
    //integrate over whole input cubemap for each output texel

    //precompute input texel directions
    std::vector<Sample> input_samples;
    input_samples.reserve(in_width * in_width * 6);

    for(uint32_t face = 0; face < 6; face++){
        for(uint32_t y = 0; y < in_width; y++){
            for(uint32_t x = 0; x < in_width; x++){
                vec3 dir = cube_texel_direction(face, x, y, in_width);
                float solid_angle = cube_texel_solid_angle(x,y, in_width);
                uint32_t index = face * in_width * in_width + y * in_width + x;
                vec3 L = in_cube_radiance_values[index];
                input_samples.emplace_back(Sample{.direction = dir, .solid_angle = solid_angle, .radiance = L});
            }
        }
    }

    //integrate over all the input samples for each out texel
    std::vector<vec3> out_radiance(6 * out_width * out_width);
    for(uint32_t face = 0; face < 6; ++face){
        for(uint32_t y = 0; y < out_width; ++y){
            for(uint32_t x = 0; x < out_width; ++x){

                vec3 n = cube_texel_direction(face, x, y, out_width);

                vec3 irradiance(0.0f);
                
                for(const auto& s : input_samples){

                    float cos_theta = dot(n, s.direction);

                    if(cos_theta > 0.0f){
                        irradiance = irradiance + s.radiance * cos_theta * s.solid_angle / 3.14159265f;
                    }
                }
                
                uint32_t index = face * out_width * out_width + y * out_width + x;
                out_radiance[index] = irradiance;
            }
        }
    }

    //output the image
    assert(out_cube.size() == out_width * out_width * 6 * 4);
    
    radiance_values_to_rgbe(out_radiance, out_cube, out_width * out_width * 6);

}

//this code is also taken from https://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
vec3 importance_sample_ggx(vec2 Xi, vec3 N, float roughness)
{
    float a = roughness*roughness;
	
    float phi = 2.0f * 3.14159265f * Xi.x; //Xi.x is evenly spaced so phi just evenly goes around the hemisphere
    
    // this black magic is derived from the NDF(normal distribution function), by first converting the NDF from projected
    // area density to solid angle density, then transforming the NDF to be in terms of cosTheta, 
    // then integrating to get the CDF, then using inverse transform sampling to translate from Xi.y (uniform) to 
    // cosTheta (biased by the ndf)
    float cosTheta = sqrt((1.0f - Xi.y) / (1.0f + (a*a - 1.0f) * Xi.y)); 

    float sinTheta = sqrt(1.0f - cosTheta*cosTheta);
	
    // from spherical coordinates to cartesian coordinates
    vec3 H;
    H.x = cos(phi) * sinTheta;
    H.y = sin(phi) * sinTheta;
    H.z = cosTheta;
	
    // from tangent-space vector to world-space sample vector
    vec3 up        = abs(N.z) < 0.999f ? vec3(0.0f, 0.0f, 1.0f) : vec3(1.0f, 0.0f, 0.0f);
    vec3 tangent   = normalized(cross(up, N));
    vec3 bitangent = cross(N, tangent);
	
    vec3 sampleVec = tangent * H.x + bitangent * H.y + N * H.z;
    return normalized(sampleVec);
}  

//this implementation comes from https://learnopengl.com/PBR/IBL/Specular-IBL which, in turns, takes from
//holger's article https://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
float RadicalInverse_VanDerCorput(uint32_t bits) 
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10f; // / 0x100000000
}
// ----------------------------------------------------------------------------
vec2 Hammersley(uint32_t i, uint32_t N)
{
    return vec2(float(i)/float(N), RadicalInverse_VanDerCorput(i));
}  


//for pbr shading
void convolve_cubemap_ggx(uint32_t in_width, std::vector<vec3> const &env, uint32_t out_width, std::vector<vec3> &output, float roughness){
    //the code in the inner loop for each texel 
    //is adapted from https://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
    for(uint32_t face = 0; face < 6; ++face){
        for(uint32_t y = 0; y < out_width; ++y){
            for(uint32_t x = 0; x < out_width; ++x){
                vec3 R = cube_texel_direction(face, x, y, out_width);
                vec3 N = R;
                vec3 V = R;

                vec3 prefiltered(0.0f);//the color to be summed into
                float totalWeight = 0.0f;
                const uint32_t NumSamples = 512;
                for(uint32_t i = 0; i < NumSamples; i++){
                    vec2 Xi = Hammersley(i, NumSamples);
                    vec3 H = importance_sample_ggx(Xi, N, roughness);
                    vec3 L = normalized(2.0f * dot(V,H) * H - V);

                    float NdotL = std::max(dot(N,L), 0.0f);//weighing by cos of the angle between N and L to (somehow) reduce approximation error
                    
                    if (NdotL > 0.0f) {
                        uint32_t in_face, in_y, in_x;
                        direction_to_cube_texel(L, in_width, in_face, in_x, in_y);
                        uint32_t index = in_face * in_width * in_width + in_y * in_width + in_x;
                        vec3 radiance = env[index];
                        prefiltered += radiance * NdotL;
                        totalWeight += NdotL;
                    }
                }
                
                //write the calculated radiance to the output map
                uint32_t out_index = face * out_width * out_width + y * out_width + x;
                output[out_index] = prefiltered/totalWeight;
            }
        }
    }
}

//for the 2d brdf LUT
float GeometrySchlickGGX(float NdotV, float roughness)
{
    float a = roughness * roughness;;
    float k = a / 2.0f;

    float nom   = NdotV;
    float denom = NdotV * (1.0f - k) + k;

    return nom / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = std::max(dot(N, V), 0.0f);
    float NdotL = std::max(dot(N, L), 0.0f);
    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
} 

vec2 integrate_brdf(float roughness, float NoV){
    vec3 V;
    V.x = sqrt( 1.0f - NoV * NoV ); // sin
    V.y = 0;
    V.z = NoV; // cos
    float A = 0;
    float B = 0;
    const uint32_t NumSamples = 1024;
    vec3 N =  vec3(0.0f, 0.0f, 1.0f);
    for( uint32_t i = 0; i < NumSamples; i++ ){
        vec2 Xi = Hammersley( i, NumSamples );
        vec3 H = importance_sample_ggx( Xi, N, roughness );
        vec3 L = 2 * dot( V, H ) * H - V;
        float NoL = std::max(L.z, 0.0f);
        float NoH = std::max(H.z, 0.0f);
        float VoH = std::max(dot(V, H), 0.0f);
        if( NoL > 0 ){
            float G = GeometrySmith(N, V, L, roughness);
            float safeNoV = std::max(NoV, 0.0001f);
            float G_Vis = G * VoH / (NoH * safeNoV);
            float Fc = std::powf( 1 - VoH, 5 );
            A += (1 - Fc) * G_Vis;
            B += Fc * G_Vis;
        }
    }
    return vec2( A, B ) / NumSamples;
}