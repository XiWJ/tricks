#version 330 core
out vec4 FragColor;

struct Material {
    sampler2D diffuse;
    sampler2D specular;
    float shininess;
};


uniform Material material;


in vec2 TexCoords;


void main()
{
    vec3 result = vec3(texture(material.diffuse, TexCoords));
    FragColor = vec4(result, 1.0f);
}