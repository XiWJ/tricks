#version 330 core
layout (location = 0) in vec3 aPos; // 位置属性
layout (location = 1) in vec3 aColor; // 颜色属性
layout (location = 2) in vec2 aTexCoord; // 纹理坐标属性

out vec3 vertexColor; // 为片段着色器指定一个颜色输出
out vec2 texCoord;

uniform float offset;
uniform mat4 transform;

void main()
{
    gl_Position = transform * vec4(aPos, 1.0f);
    vertexColor = aColor;
    texCoord = aTexCoord;
}