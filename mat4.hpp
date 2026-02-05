#pragma once

// a small matrix library for 4x4 matrices and 3x3 stuff

#include <array>
#include <cmath>
#include <cstdint>

using mat4 = std::array<float, 16>;//column major
static_assert(sizeof(mat4) == 16 * 4 );

struct vec4{
	vec4(float _x, float _y, float _z, float _w){
		x = _x;
		y = _y;
		z = _z;
		w = _w;
	}
	vec4(){
		x = 0;
		y = 0;
		z = 0;
		w = 0;
	}
	union {
		struct {
			float x;
			float y;
			float z;
			float w;
		};
		float data[4] = {};
	};
};

static_assert(sizeof(vec4) == 4 * 4 );


struct vec3{
	vec3(float _x, float _y, float _z){
		x = _x;
		y = _y;
		z = _z;
	};
	vec3(){
		x = 0;
		y = 0;
		z = 0;
	}
	union {
		struct {
			float x;
			float y;
			float z;
		};
		float data[3] = {};
	};
};
static_assert(sizeof(vec3) == 3 * 4 );

inline float dot(vec4 const &a,vec4 const &b){
    return a.x*b.x+ a.y*b.y+ a.z*b.z+ a.w*b.w;
}




inline vec4 operator*(mat4 const &A, vec4 const &b){
    vec4 ret;
    //ret = A * b
    for(uint32_t r = 0; r<4;r++){
        //first term of the row
        ret.data[r] = A[0 * 4 + r] * b.x;
        //add the rest
		for (uint32_t k = 1; k < 4; ++k) {
			ret.data[r] += A[k * 4 + r] * b.data[k];
		}
    }
	return ret;
}

inline mat4 operator*(mat4 const &A, mat4 const &B){
    mat4 ret;
    //ret = A * B
    for(uint32_t r = 0; r < 4; r++){
        for(uint32_t c = 0; c < 4; c++){
            ret[c * 4 + r] = A[0 * 4 + r] * B[c * 4 + 0];
			for (uint32_t k = 1; k < 4; ++k) {
				ret[c * 4 + r] += A[k * 4 + r] * B[c * 4 + k];
			}
        }
    }
	return ret;
}




//-------vec3 stuff-----
inline float dot(vec3 const &a,vec3 const &b){
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
inline vec3 cross(vec3 const &a, vec3 const &b){
	return vec3{a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
inline vec3 operator-(vec3 const &a, vec3 const & b){
	return vec3{a.x-b.x, a.y-b.y, a.z-b.z};
}
inline vec3 operator+(vec3 const &a, vec3 const & b){
	return vec3{a.x+b.x, a.y+b.y, a.z+b.z};
}
inline vec3 operator*(vec3 const &a, float const &c){
	return vec3{a.x*c, a.y*c, a.z*c};
}
inline vec3 operator*(float const &c, vec3 const &a){
	return vec3{a.x*c, a.y*c, a.z*c};
}
inline vec3 operator/(vec3 const &a, float const &c){
	return vec3{a.x/c, a.y/c, a.z/c};
}
inline float length(vec3 const &a){
	return std::sqrt(dot(a,a));
}
inline vec3 normalized(vec3 const &a){
	float len = length(a);
	return a/len;
}


//----end vec3 stuff----
//----view projection stuff----


//perspective projection matrix.
// - vfov is fov *in radians*
// - near maps to 0, far maps to 1
// looks down -z with +y up and +x right
inline mat4 perspective(float vfov, float aspect, float near, float far) {
	//as per https://www.terathon.com/gdc07_lengyel.pdf
	// (with modifications for Vulkan-style coordinate system)
	//  notably: flip y (vulkan device coords are y-down)
	//       and rescale z (vulkan device coords are z-[0,1])
	const float e = 1.0f / std::tan(vfov / 2.0f);
	const float a = aspect;
	const float n = near;
	const float f = far;
	return mat4{ //note: column-major storage order!
		e/a,  0.0f,                      0.0f, 0.0f,
		0.0f,   -e,                      0.0f, 0.0f,
		0.0f, 0.0f,-0.5f - 0.5f * (f+n)/(f-n),-1.0f,
		0.0f, 0.0f,             - (f*n)/(f-n), 0.0f,
	};

    
}
//look at matrix:
// makes a camera-space-from-world matrix for a camera at eye looking toward
// target with up-vector pointing (as-close-as-possible) along up.
// That is, it maps:
//  - eye_xyz to the origin
//  - the unit length vector from eye_xyz to target_xyz to -z
//  - an as-close-as-possible unit-length vector to up to +y
inline mat4 look_at(
	vec3 eye, vec3 target, vec3 up) {
    vec3 eye_to_target = target - eye;
	eye_to_target = normalized(eye_to_target);
	vec3 up_orthognal = up - dot(eye_to_target, up) * eye_to_target;
	up_orthognal = normalized(up_orthognal);
	vec3 right = cross(eye_to_target, up_orthognal);
	
	float right_dot_eye = dot(right,eye);
	float up_dot_eye = dot(up_orthognal,eye);
	float forward_dot_eye = dot(eye_to_target,eye);

	//construct the matrix
	return mat4{
		right.x,up_orthognal.x,-eye_to_target.x,0.0f,
		right.y,up_orthognal.y,-eye_to_target.y,0.0f,
		right.z,up_orthognal.z,-eye_to_target.z,0.0f,
		-right_dot_eye, -up_dot_eye, forward_dot_eye,1.0f
	};
}



inline mat4 orbit(
		vec3 target,
		float azimuth, float elevation, float radius
	) {

	//compute right direction
	vec3 right = vec3(-std::sin(azimuth), std::cos(azimuth), 0.0f);
	//compute up direction
	vec3 up = vec3(-std::sin(elevation) * std::cos(azimuth), -std::sin(elevation) * std::sin(azimuth), std::cos(elevation));
	//compute out direction
	vec3 out = vec3(std::cos(elevation) * std::cos(azimuth), std::cos(elevation) * std::sin(azimuth), std::sin(elevation));
	//compute camera position
	vec3 c_pos = target + radius * out;
	//assemble and return camera-from-world matrix
	mat4 camera_from_world = {
		right.x, up.x, out.x, 0.0f,
		right.y, up.y, out.y, 0.0f,
		right.z, up.z, out.z, 0.0f,
		-dot(c_pos,right), -dot(c_pos,up),-dot(c_pos, out), 1.0f
	};
	return camera_from_world;
}
//----end view projection stuff----

