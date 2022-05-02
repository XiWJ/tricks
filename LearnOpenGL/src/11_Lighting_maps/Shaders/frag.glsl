#version 330 core
struct Material {
    sampler2D diffuse; // 原来的 vec3 ambient + diffuse => sample2D diffuse
    sampler2D specular; // 原来的 vec3 specular => sample2D specular
    float shininess;
}; 
struct Light {
    vec3 position;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

out vec4 FragColor;

uniform vec3 viewPos; // 相机位置
uniform Material material; // 材质
uniform Light light; // 环境光

in vec3 Normal;
in vec3 FragPos;
in vec2 TexCoords; // UV

void main()
{
    // 环境光
    vec3 ambient = light.ambient * vec3(texture(material.diffuse, TexCoords));

    // 漫反射
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(light.position - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = light.diffuse * diff * vec3(texture(material.diffuse, TexCoords));

    // 镜面反射
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 halfVec = normalize(viewDir + lightDir);
    float spec = pow(max(dot(halfVec, norm), 0.0), material.shininess);
    vec3 specular = light.specular * spec * vec3(texture(material.specular, TexCoords));
    
    vec3 result = ambient + diffuse + specular;
    FragColor = vec4(result, 1.0);
}