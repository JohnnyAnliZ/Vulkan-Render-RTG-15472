#pragma once

#include "mat4.hpp"
#include <array>
#include <limits>
#include <corecrt_math_defines.h>

enum class CameraMode {
	Scene = 0,
	Free  = 1,
	Debug = 2,
};

struct BasicCamera {
	vec3  eye;
	vec3  dir;
	vec3  up;
	float aspect;
	float vfov;
	float near;
	float far = std::numeric_limits<float>::infinity();

	mat4 clip_from_world() { return perspective(vfov, aspect, near, far) * look_at_free(eye, dir, up); }
	mat4 world_from_clip() { return clip_from_world().inverse(); }
	std::array<vec3, 8> get_frustum_corners() const;
};

struct OrbitCamera {
	vec3  target    = vec3();
	float radius    = 5.0f;
	float azimuth   = 0.0f;
	float elevation = 0.25f * float(M_PI);
	float fov       = 60.0f / 180.0f * float(M_PI);
	float near      = 0.1f;
	float far       = 1000.0f;

	mat4 clip_from_world(float aspect) {
		return perspective(60.0f * float(M_PI) / 180.0f, aspect, near, far)
		     * orbit(target, azimuth, elevation, radius);
	}
	mat4 world_from_clip(float aspect) { return clip_from_world(aspect).inverse(); }
	vec3 get_eye() const;
	std::array<vec3, 8> get_frustum_corners(float aspect) const;
};
