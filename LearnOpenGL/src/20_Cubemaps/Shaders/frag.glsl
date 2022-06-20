#version 330 core
out vec4 FragColor;

struct Material {
    sampler2D diffuse; // 原来的 vec3 ambient + diffuse => sample2D diffuse
    sampler2D specular; // 原来的 vec3 specular => sample2D specular
    float shininess;
};


uniform Material material;
uniform vec3 cameraPos;
uniform samplerCube skybox;


in vec3 Normal;
in vec3 Position;


void main()
{
    // 反射
    // vec3 I = normalize(Position - cameraPos);
    // vec3 R = reflect(I, normalize(Normal));
    // FragColor = vec4(texture(skybox, R).rgb, 1.0);

    // 折射
    float ratio = 1.00 / 1.52;
    vec3 I = normalize(Position - cameraPos);
    vec3 R = refract(I, normalize(Normal), ratio);
    FragColor = vec4(texture(skybox, R).rgb, 1.0);
}