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
uniform float width; // 屏幕尺寸
uniform float height;


in vec2 TexCoords;


void main()
{
    FragColor = texture(material.diffuse, TexCoords);
    if (!first_pass)
    {
        vec2 tex_coord = vec2(gl_FragCoord.x / width, gl_FragCoord.y / height); // 计算该像素点对应depthTexture上的uv坐标
        float max_depth = texture(depthTexture, tex_coord).r;
        if (gl_FragCoord.z <= max_depth)
        {
            discard;
        }
    }
}