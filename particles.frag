#version 450
layout(location = 0) in float in_life;
layout(location = 0) out vec4 out_color;

void main() {
    vec2  c  = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(c, c);
    if (r2 > 1.0) discard;
    float alpha = (1.0 - r2) * clamp(in_life / 2.0, 0.0, 1.0) * 0.7;
    out_color = vec4(0.8, 0.9, 1.0, alpha);
}
