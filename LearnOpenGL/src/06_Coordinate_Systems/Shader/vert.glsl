#version 330 core
layout (location = 0) in vec3 aPos; // 位置属性
layout (location = 1) in vec2 aTexCoord; // 纹理坐标属性

out vec2 texCoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

void main()
{
    gl_Position = proj * view * model * vec4(aPos, 1.0f);
    texCoord = aTexCoord;
}