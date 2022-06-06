#version 330 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D screenTexture; // FBO获取的color texture
uniform sampler2D depthTexture;

void main()
{   
    vec4 depth = texture(depthTexture, TexCoords);
    if (depth.r < 1)
    {
        FragColor = texture(screenTexture, TexCoords);
        gl_FragDepth = depth.r;
    }
    else
    {
        discard;
    }
} 