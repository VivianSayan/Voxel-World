#version 450

// The three corners live in the shader rather than in a vertex buffer, so
// there is no buffer, no memory allocation and no vertex input state to set up.
// Swap this for a real vertex buffer when you want geometry that changes.
vec2 CORNERS[3] = vec2[](
    vec2( 0.0, -0.6),
    vec2( 0.6,  0.6),
    vec2(-0.6,  0.6)
);

vec3 COLOURS[3] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);

layout(location = 0) out vec3 tint;

void main() {
    gl_Position = vec4(CORNERS[gl_VertexIndex], 0.0, 1.0);
    tint = COLOURS[gl_VertexIndex];
}
