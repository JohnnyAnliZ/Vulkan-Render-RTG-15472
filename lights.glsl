
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
};


vec3 sun_sphere_irradiance(float cos_theta, float sin_half, vec3 base_irradiance){
    if(cos_theta <= -sin_half) return vec3(0.0);//if the sun is fully below horizon

    if(cos_theta >= sin_half){//if the sun is fully above horizon
        return base_irradiance * cos_theta;
    }

    //if the sun is crossing the horizon
    float interp = cubic_spline_utility(
        cos_theta,
        -sin_half, 0, 0,
        sin_half, sin_half, 1
    );
    return base_irradiance * interp;
}



vec3 diffuse_lighting_irradiance(Light light, vec3 surface_point, vec3 n){//doing this in world space

    if(light.type == 0){//light type 0 is sun, uses angle and strength 
        float cos_theta = dot(n, -normalize(light.direction.xyz));//since theta could be 0 to 180(angle between n and l)
        float sin_half = sin(light.angle * 0.5);//sin of half of the light's subtended angle(positive)
        vec3 base_irradiance = (light.color.xyz * light.strength).xyz;

        return sun_sphere_irradiance(cos_theta, sin_half, base_irradiance);
        
    }

    else if(light.type == 1){//light type 1 is sphere, uses radius, power, position
        vec3 surface_to_light = light.position.xyz - surface_point;

        float dist = length(surface_to_light);
        float sin_half = min(light.radius / dist, 1.0);
        float cos_theta = dot(n, surface_to_light/dist);

        float inverse_square_attenuation = 1 / (dist * dist);
        float limit_falloff = max(0, 1 - pow((dist/ light.limit),4));
        vec3 base_irradiance = (light.color.xyz * light.power).xyz * min(100000, inverse_square_attenuation) * limit_falloff;

        return sun_sphere_irradiance(cos_theta, sin_half, base_irradiance);

    }
    else if(light.type == 2){//spot light, uses direction, radius, power, position, fov, blend
        vec3 surface_to_light = light.position.xyz - surface_point;

        float dist = length(surface_to_light);
        float sin_half = min(light.radius / dist, 1.0);
        float cos_theta = dot(n, surface_to_light/dist);
        float inverse_square_attenuation = 1 / (dist * dist);
        float limit_falloff = max(0, 1 - pow((dist/ light.limit),4));
        vec3 base_irradiance = (light.color.xyz * light.power).xyz * min(100000, inverse_square_attenuation) * limit_falloff;

        //get the cone attenuation
        float cos_phi = dot(-surface_to_light/dist, normalize(light.direction.xyz));//phi is the angle between spotlight direction and spotlight to surface
        
        float phi = acos(cos_phi);
        float inner = light.fov * (1-light.blend) * 0.5;
        float outer = light.fov * 0.5;



        float spotlight_attenuation = clamp((outer - phi) / (outer - inner), 0.0, 1.0);


        return sun_sphere_irradiance(cos_theta, sin_half, base_irradiance * spotlight_attenuation);
    }
    return vec3(0.0);
}


vec3 specular_lighting_irradiance(Light light, vec3 surface_point, vec3 eye, vec3 n, float alpha){//doing this in world space
    vec3 V = normalize(eye - surface_point);
    vec3 R = reflect(-V, n);
    vec3 L = light.position.xyz - surface_point;
    if(light.type == 0){//light type 0 is sun, uses angle and strength 

    }
    else if(light.type == 1){//light type 1 is sphere, uses radius, power, position
        vec3 center_to_ray = dot(L,R) * R - L;
        vec3 closest_point = L + center_to_ray * min(1, light.radius/length(center_to_ray));
        float l = length(closest_point);
        vec3 rep_dir = closest_point/l;
        
        float attenuation = 1/(alpha * alpha * l * l);

    }
    else if(light.type == 2){//spot light, uses direction, radius, power, position, fov, blend

    }
    return vec3(0.0);

}

#endif
