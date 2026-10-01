#version 450

layout(location = 0) in vec3 tint;
layout(location = 0) out vec4 colour;

void main() {
    colour = vec4(tint, 1.0);
}
