#pragma once

#include "mat4.hpp"
#include <array>
#include <cassert>
#include <corecrt_math_defines.h>

struct Light {
	vec4 color;
	vec4 position;
	vec4 direction;
	int type; //0 - sun ; 1 - sphere ; 2 - spot
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
	vec4 shadow_atlases[6]; // (offset.x, offset.y, scale.x, scale.y)
	mat4 CLIP_FROM_WORLD[6];

	void compute_clip_from_world_spot() {
		assert(type == 2);
		vec3 up;
		if (abs(dot(direction.xyz(), vec3(0,1,0))) > 0.99f) {
			up = vec3(0,0,1);
		} else {
			up = vec3(0,1,0);
		}
		up = vec3(0,1,0);
		CLIP_FROM_WORLD[0] = perspective(fov, 1.0f, 1.0f, limit) * look_at_free(position.xyz(), vec3(0.0f)+direction.xyz(), up);
	}

	void compute_clip_from_world_sphere() {
		assert(type == 1);
		mat4 views[6];
		views[0] = look_at_free(position.xyz(), vec3(1,0,0),  vec3(0,-1,0));
		views[1] = look_at_free(position.xyz(), vec3(-1,0,0), vec3(0,-1,0));
		views[2] = look_at_free(position.xyz(), vec3(0,1,0),  vec3(0,0,1));
		views[3] = look_at_free(position.xyz(), vec3(0,-1,0), vec3(0,0,-1));
		views[4] = look_at_free(position.xyz(), vec3(0,0,1),  vec3(0,-1,0));
		views[5] = look_at_free(position.xyz(), vec3(0,0,-1), vec3(0,-1,0));
		for (uint32_t i = 0; i < 6; i++)
			CLIP_FROM_WORLD[i] = perspective((float)M_PI / 2.0f, 1.0f, 0.1f, limit) * views[i];
	}

	std::array<vec3, 8> get_corners() const;
	std::array<vec3, 8> get_frustum_corners() const;
};
static_assert(sizeof(Light) == 3*4*4 + 8*4 + 4*4*6 + 6*16*4, "Light structure is packed");
