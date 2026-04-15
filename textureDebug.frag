#version 450

layout(set = 0, binding = 0) uniform sampler2D SHADOW_ATLAS;
layout(set = 1, binding = 0) uniform sampler3D VELOCITY_VOL;

layout(location = 0) in vec2 position;
layout(location = 0) out vec4 outColor;

void main(){

    vec2 uv = position;

    // --- Shadow atlas (bottom right) ---
    float s = 0.25;
    vec2 minBR = vec2(1.0 - s, 0.0);
    vec2 maxBR = vec2(1.0, s);

    if (uv.x >= minBR.x && uv.y <= maxBR.y) {
        vec2 localUV = (uv - minBR) / s;
        vec3 col = texture(SHADOW_ATLAS, localUV).r * vec3(1.0,1.0,1.0);
        outColor = vec4(col, 1.0);
        return;
    }

    // --- Velocity slices (top middle) ---
    int sliceCount = 8;
    float sliceSize = 0.1;
    float spacing = 0.01;

    float totalWidth = sliceCount * sliceSize + (sliceCount - 1) * spacing;
    float startX = 0.5 - totalWidth * 0.5;
    float topY = 1.0 - sliceSize;

    for (int i = 0; i < sliceCount; i++) {

        float x0 = startX + i * (sliceSize + spacing);
        float x1 = x0 + sliceSize;
        float y0 = topY;
        float y1 = 1.0;

        if (uv.x >= x0 && uv.x <= x1 &&
            uv.y >= y0 && uv.y <= y1) {

            vec2 localUV = vec2(
                (uv.x - x0) / sliceSize,
                (uv.y - y0) / sliceSize
            );

            float z = float(i) / float(sliceCount - 1);

            vec3 samplePos = vec3(localUV, z)/2;//0-1 on all axes


            vec3 vel = texture(VELOCITY_VOL, samplePos).xyz;

            outColor = vec4(vel * 0.5 + 0.5, 1.0);
            return;
        }
    }

    //background
    discard;
}
