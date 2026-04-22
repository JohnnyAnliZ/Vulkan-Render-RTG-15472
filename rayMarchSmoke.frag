#version 450

layout(set = 0, binding = 0) uniform sampler3D DENSITY_VOL;

layout(location = 0) in vec2 position;   // [0,1]
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform Push {
    mat4 WORLD_FROM_CLIP;
    vec4 EYE;   // xyz used
    vec4 volume_center_step; // xyz=center, w=cell size
    uint N;
} pc;

bool intersectAABB(
    vec3 rayOrigin,
    vec3 rayDir,
    vec3 bmin,
    vec3 bmax,
    out float tEnter,
    out float tExit
) {
    vec3 invDir = 1.0 / rayDir;

    vec3 t0 = (bmin - rayOrigin) * invDir;
    vec3 t1 = (bmax - rayOrigin) * invDir;

    vec3 tsmaller = min(t0, t1);
    vec3 tbigger  = max(t0, t1);

    tEnter = max(max(tsmaller.x, tsmaller.y), tsmaller.z);
    tExit  = min(min(tbigger.x, tbigger.y), tbigger.z);

    return tExit >= tEnter;
}

void main() {
    vec3 eye = pc.EYE.xyz;

    float cellSize = pc.volume_center_step.w / float(pc.N);
    float stepSize = cellSize * 0.5;   // good default

    // Volume bounds:
    float size = pc.volume_center_step.w;
    vec3 halfSize = vec3(size * 0.5);

    vec3 center = pc.volume_center_step.xyz;
    vec3 boxMin = center - halfSize;
    vec3 boxMax = center + halfSize;

    // --- Ray reconstruction ---
    vec2 ndc = position * 2.0 - 1.0;

    vec4 clipNear = vec4(ndc, 0.0, 1.0);
    vec4 clipFar  = vec4(ndc, 1.0, 1.0);

    vec4 worldNear4 = pc.WORLD_FROM_CLIP * clipNear;
    vec4 worldFar4  = pc.WORLD_FROM_CLIP * clipFar;

    vec3 worldNear = worldNear4.xyz / worldNear4.w;
    vec3 worldFar  = worldFar4.xyz / worldFar4.w;

    vec3 rayOrigin = eye;
    vec3 rayDir = normalize(worldFar - worldNear);

    // --- Ray-box intersection ---
    float tEnter, tExit;
    if (!intersectAABB(rayOrigin, rayDir, boxMin, boxMax, tEnter, tExit)) {
        discard;
    }

    float tStart = max(tEnter, 0.0);
    float tEnd   = tExit;

    if (tStart >= tEnd) {
        discard;
    }

    // --- Accumulation ---
    float accum = 0.0;

    const int MAX_STEPS = 512;

    float t = tStart;

    for (int i = 0; i < MAX_STEPS; ++i) {
        if (t >= tEnd) break;

        vec3 p_ws = rayOrigin + t * rayDir;

        // world -> [0,1]^3
        vec3 uvw = (p_ws - boxMin) / (boxMax - boxMin);

        float density = texture(DENSITY_VOL, uvw).x;

        accum += density * stepSize;

        t += stepSize;
    }

    // simple visualization
    float value = accum;



    outColor = vec4(value * vec3(1.0, 1.0, 1.0), 1.0);
}


