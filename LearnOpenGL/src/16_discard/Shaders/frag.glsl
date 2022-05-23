#version 330 core
out vec4 FragColor;

struct Material {
    sampler2D diffuse; // 原来的 vec3 ambient + diffuse => sample2D diffuse
    sampler2D specular; // 原来的 vec3 specular => sample2D specular
    float shininess;
};


uniform Material material;


in vec2 TexCoords;


void main()
{
    vec4 texColor = texture(material.diffuse, TexCoords);
    if (texColor.a < 0.1)
    {
        discard;
    }
    FragColor = texColor;
}