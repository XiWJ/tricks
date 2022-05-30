#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D screenTexture; // FBO获取的color texture

void main()
{   
    FragColor = texture(screenTexture, TexCoords);
} 