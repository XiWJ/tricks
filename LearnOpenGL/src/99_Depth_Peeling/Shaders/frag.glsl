#version 330 core
out vec4 FragColor;

struct Material {
    sampler2D diffuse;
    sampler2D specular;
    float shininess;
};


uniform Material material;
uniform sampler2D depthTexture;
uniform bool first_pass;


in vec2 TexCoords;


void main()
{
    FragColor = texture(material.diffuse, TexCoords);
    if (!first_pass)
    {
        float max_depth = texture(depthTexture, TexCoords).r;
        if (gl_FragCoord.z <= max_depth)
        {
            discard;
        }
    }
}