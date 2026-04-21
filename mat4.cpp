#include "mat4.hpp"


mat4 mat4::inverse(){
    float a[4][8] = {};

    // Build augmented matrix [M | I]
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            a[r][c] = data[c*4 + r];   // use explicit row/col accessor
        }
        a[r][4 + r] = 1.0f;
    }

    // Eliminate
    for (int col = 0; col < 4; ++col) {
        int pivot = col;
        for (int r = col + 1; r < 4; ++r) {
            if (std::abs(a[r][col]) > std::abs(a[pivot][col])) {
                pivot = r;
            }
        }

        assert(std::abs(a[pivot][col]) > 1e-8f);

        if (pivot != col) {
            for (int c = 0; c < 8; ++c) std::swap(a[col][c], a[pivot][c]);
        }

        float invPivot = 1.0f / a[col][col];
        for (int c = 0; c < 8; ++c) a[col][c] *= invPivot;

        for (int r = 0; r < 4; ++r) {
            if (r == col) continue;
            float f = a[r][col];
            for (int c = 0; c < 8; ++c) {
                a[r][c] -= f * a[col][c];
            }
        }
    }

    mat4 out;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            out[c*4 + r] = a[r][4 + c];
        }   
    }
    return out;
}



mat3 mat3::inverse() {
    float a = data[0];
    float b = data[1];
    float c = data[2];
    float d = data[3];
    float e = data[4];
    float f = data[5];
    float g = data[6];
    float h = data[7];
    float i = data[8];

    float det =
        a * (e*i - f*h) -
        b * (d*i - f*g) +
        c * (d*h - e*g);

    assert(det != 0.0f);

    float invDet = 1.0f / det;

    return mat3{{
        (e*i - f*h) * invDet,
        (c*h - b*i) * invDet,
        (b*f - c*e) * invDet,

        (f*g - d*i) * invDet,
        (a*i - c*g) * invDet,
        (c*d - a*f) * invDet,

        (d*h - e*g) * invDet,
        (b*g - a*h) * invDet,
        (a*e - b*d) * invDet
    }};
}


//taken and modified from http://number-none.com/product/Understanding%20Slerp,%20Then%20Not%20Using%20It/
quat quat::slerp(quat const &v0, quat const &v1, float t){
    // v0 and v1 should be unit length or else
    // something broken will happen.

    // Compute the cosine of the angle between the two vectors.
    float dot_product = dot(v0, v1);

    const float DOT_THRESHOLD = 0.9995f;
    if (dot_product > DOT_THRESHOLD) {
        // If the inputs are too close for comfort, linearly interpolate
        // and normalize the result.

        quat result = v0 + t*(v1 - v0);
        result.normalize();
        return result;
    }

    dot_product = std::min(dot_product, 1.0f);
    dot_product = std::max(dot_product, -1.0f);// Robustness: Stay within domain of acos()
    
    float theta_0 = acos(dot_product);  // theta_0 = angle between input vectors
    float theta = theta_0*t;    // theta = angle between v0 and result 

    quat v2 = v1 - v0*dot_product;
    v2.normalize();              // { v0, v2 } is now an orthonormal basis

    return v0*cos(theta) + v2*sin(theta);
}