
//shader functions for light calculations
#ifndef LIGHTS_GLSL
#define LIGHTS_GLSL

struct Light {
    vec4 color;
    vec4 position;
    vec4 direction;
    int type;//0 - sun ; 1 - sphere ; 2 - spot
    float limit;
    //sphere 
    float radius;
    float power;
    //sun
    float angle;
    float strength;
    //spot light only
    float fov;
    float blend;
    vec4 shadow_atlases[6];// (offset.x, offset.y, scale.x, scale.y)
	mat4 CLIP_FROM_WORLD[6];
};


float cubic_spline_utility(
    float x,//interpolation point
    float x0, float y0, float m0,
    float x1, float y1, float m1
){//convert from f(x)
    float dx = x1 - x0;
    float t = (x - x0)/dx;
    t = clamp(t, 0.0, 1.0);

    float t_sqr = t * t;
    float t_cube = t_sqr * t;

    //hermite basis are derived from the inverse of the 4 x 4 matrix that transforms a,b,c,d into f(x=x0), f(x=x1), f'(x=x0), f'(x=x1)
    //which are, in the context of H(t), H(t=0) = y0,  H(t=1) = y1, H'(t=0)m0 * dx, H'(t=1) =m1 *dx
    //(dx here is actually the same as dx/dt, chain rule for going from f(x) to H(t))

    //hermite basis are:
    float h0 =  2 * t_cube - 3 * t_sqr + 1;
    float h1 =  -2 * t_cube + 3 * t_sqr;
    float h2 =  t_cube - 2 * t_sqr + t;
    float h3 =  t_cube -  t_sqr;

    return h0 * y0 + h1 *y1 + h2 * m0 * dx + h3 * m1 * dx;

}

float sun_sphere_horizon_and_forshortening(float cos_theta, float sin_half){
    if(cos_theta <= -sin_half) return 0.0;//if the sun is fully below horizon

    if(cos_theta >= sin_half){//if the sun is fully above horizon
        return cos_theta;
    }

    //if the sun is crossing the horizon
    float interp = cubic_spline_utility(
        cos_theta,
        -sin_half, 0, 0,
        sin_half, sin_half, 1
    );
    return interp;
}


float sphere_light_attenuation(float dist, float limit){
    float inverse_square_attenuation = 1 / (dist * dist);
    float limit_falloff = max(0, 1 - pow((dist/limit),4));
    return min(100000, inverse_square_attenuation) * limit_falloff;
}

float cone_attenuation(vec3 light_to_surface, vec3 light_dir, float fov, float blend){
    float cos_phi = dot(light_to_surface, light_dir);//phi is the angle between spotlight direction and spotlight to surface
    
    float phi = acos(cos_phi);
    float inner = fov * (1-blend) * 0.5;
    float outer = fov * 0.5;

    return clamp((outer - phi) / (outer - inner), 0.0, 1.0);
}

vec3 diffuse_lighting_irradiance(Light light, vec3 surface_point, vec3 n){//doing this in world space
    
    if(light.type == 0){//light type 0 is sun, uses angle and strength 
        float cos_theta = dot(n, -normalize(light.direction.xyz));//since theta could be 0 to 180(angle between n and l)
        float sin_half = sin(light.angle * 0.5);//sin of half of the light's subtended angle(positive)
        vec3 base_irradiance = (light.color.xyz * light.strength);
        
        return base_irradiance * sun_sphere_horizon_and_forshortening(cos_theta, sin_half);
    }

    else if(light.type == 1){//light type 1 is sphere, uses radius, power, position
        vec3 surface_to_light = light.position.xyz - surface_point;

        float dist = length(surface_to_light);
        float sin_half = min(light.radius / dist, 1.0);
        float cos_theta = dot(n, surface_to_light/dist);

        vec3 base_irradiance = (light.color.xyz * light.power) * sphere_light_attenuation(dist, light.limit);

        return base_irradiance * sun_sphere_horizon_and_forshortening(cos_theta, sin_half);

    }
    else if(light.type == 2){//spot light, uses direction, radius, power, position, fov, blend
        vec3 surface_to_light = light.position.xyz - surface_point;

        float dist = length(surface_to_light);
        float sin_half = min(light.radius / dist, 1.0);
        float cos_theta = dot(n, surface_to_light/dist);

        //get the cone attenuation
        float spotlight_attenuation = cone_attenuation(-surface_to_light/dist, normalize(light.direction.xyz), light.fov, light.blend);

        vec3 base_irradiance = (light.color.xyz * light.power) * sphere_light_attenuation(dist, light.limit) * spotlight_attenuation;
        return  base_irradiance * sun_sphere_horizon_and_forshortening(cos_theta, sin_half);
    }
    return vec3(0.0);
}


float GeometrySchlickGGX(float NdotV, float roughness)
{
    float a = (roughness + 1) * (roughness + 1);;
    float k = a / 8.0f;

    float nom   = NdotV;
    float denom = NdotV * (1.0f - k) + k;

    return nom / denom;
}

float GeometrySmith(float NdotV, float NdotL, float roughness)
{

    float ggx2 = GeometrySchlickGGX(NdotV, roughness);
    float ggx1 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
} 


float ggx_normal_distribution(float NdotH, float alpha_2){

    float denom =(NdotH * NdotH * (alpha_2-1) + 1);
    return alpha_2/(3.14159215354 * denom * denom);
}

vec3 fresnel_schlick(float VdotH, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);
}


vec3 specular_lighting_irradiance(Light light, vec3 surface_point, vec3 eye, vec3 n, float roughness, vec3 F0){//doing this in world space
    vec3 V = normalize(eye - surface_point);
    vec3 R = reflect(-V, n);
    vec3 L = light.position.xyz - surface_point;//surface to center of light(scaled)
    vec3 L_rep = vec3(0.0);//normalized direction from surface to representation point
    float l = 0;
    float light_attenuation = 1.0;//attenuation depending on light_type (spot inv_sqr, falloff, spotlight)
    float alpha = roughness * roughness;
    float alpha_2 = alpha * alpha;
    
    //-----evalutate representation point --- 
    if(light.type == 0){//light type 0 is sun, uses angle and strength 
        L_rep = -normalize(light.direction.xyz);
    }
    else if(light.type == 1 || light.type == 2){//sphere or spot light
        vec3 center_to_ray = dot(L,R) * R - L;
        vec3 closest_point = L + center_to_ray * min(1, light.radius/max(length(center_to_ray), 0.0001));
        l = length(closest_point);
        L_rep = closest_point/l;
    }
    vec3 H = normalize(L_rep + V);

    //-----specluar brdf-----
    float NdotL = max(0.0, dot(n,L_rep));
    float NdotV = max(0.0, dot(n,V));
    float NdotH = max(0.0, dot(n,H));
    float VdotH = max(0.0, dot(V,H));

    float D = ggx_normal_distribution(NdotH, alpha_2);
    float G = GeometrySmith(NdotV, NdotL, roughness);
    vec3 F = fresnel_schlick(VdotH, F0);

    vec3 specular_brdf = (D * G * F) / max(4.0 * NdotV * NdotL, 0.001);

    //-----light attenuation-----
    if(light.type == 0){//light type 0 is sun, uses angle and strength 
        float sin_half = sin(light.angle * 0.5);
        light_attenuation = sun_sphere_horizon_and_forshortening(NdotL, sin_half);
    }
    else if(light.type == 1 || light.type == 2){//sphere or spot light   
        float dist = length(L);

        float distance_attenuation = sphere_light_attenuation(dist, light.limit);

        float cone_atten = 1.0;
        if(light.type == 2){
            cone_atten = cone_attenuation(-normalize(L), normalize(light.direction.xyz), light.fov, light.blend);
        }

        float sin_half = min(light.radius / dist, 1.0);//sin of half the subtended angle
        float horizon_and_forshortening_attenuation = sun_sphere_horizon_and_forshortening(NdotL, sin_half);

        light_attenuation = distance_attenuation * cone_atten * horizon_and_forshortening_attenuation;
    }

    vec3 light_intensity = light.color.xyz * (light.type == 0 ? light.strength : light.power);

    return light_intensity * specular_brdf * light_attenuation;

}


vec3 compute_atlas_coordinates(Light light, vec3 worldPos) {
    vec4 light_clip = light.CLIP_FROM_WORLD[0] * vec4(worldPos, 1.0);

    if(light_clip.w <= 0.0)
        return vec3(0.0, 0.0, 0.0);

    vec3 ndc = light_clip.xyz / light_clip.w;

    vec2 uv = ndc.xy * 0.5 + 0.5;

    if(uv.x < 0.0 || uv.x > 1.0 ||
       uv.y < 0.0 || uv.y > 1.0)
        return vec3(0.0, 0.0, 0.0);

    vec2 atlas_uv;
    atlas_uv.x = light.shadow_atlases[0].x + (uv.x) * light.shadow_atlases[0].z;
    atlas_uv.y = light.shadow_atlases[0].y + (uv.y) * light.shadow_atlases[0].w;



    float depth = ndc.z;

    
    return vec3(atlas_uv, depth);
}

#endif
