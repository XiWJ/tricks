#version 330 core
out vec4 FragColor;

in vec3 vertexColor;
in vec2 texCoord;

uniform sampler2D ourTexture1;
uniform sampler2D ourTexture2;
uniform float coeff;

void main()
{
    // FragColor = texture(ourTexture, texCoord); // 根据UV在texture上找对应颜色
    // FragColor = vec4(vertexColor, 1.0f);
    // FragColor = vec4(vertexColor, 1.0f) * texture(ourTexture, texCoord);
    FragColor = mix(texture(ourTexture1, texCoord), texture(ourTexture2, vec2(1.0f - texCoord.x, texCoord.y)), coeff); // 两个纹理叠加
}