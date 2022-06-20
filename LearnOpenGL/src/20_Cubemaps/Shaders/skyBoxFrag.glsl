#version 330 core
out vec4 FragColor;

in vec3 TexCoords;

uniform samplerCube skybox; // sampleCube -- 立方体贴图

void main()
{    
    FragColor = texture(skybox, TexCoords);
}