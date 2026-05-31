#pragma once

#include "mat4.hpp"
#include <array>
#include <cmath>

struct Transform {
	mat4 CLIP_FROM_LOCAL;
	mat4 WORLD_FROM_LOCAL;
	mat4 WORLD_FROM_LOCAL_NORMAL;
};
static_assert(sizeof(Transform) == 16*4 + 16*4 + 16*4, "Transform structure is packed");

struct AABB {
	vec3 max = vec3{-INFINITY, -INFINITY, -INFINITY};
	vec3 min = vec3{ INFINITY,  INFINITY,  INFINITY};
	void get_box_corners(mat4 WORLD_FROM_LOCAL, std::array<vec3, 8> &box_corners);
};
