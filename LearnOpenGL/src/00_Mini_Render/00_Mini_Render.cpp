#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <tools/shader_s.h>
#include <tools/Camera.h>
#define STB_IMAGE_IMPLEMENTATION
#include <tools/stb_image.h>
#include <tools/OBJ_Loader.h>

#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;

// 回调
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow *window);

unsigned int loadTexture(const char *path);
void load_model(const std::string model_path, std::vector<float> & vertices, std::vector<unsigned int> & indices);

// 场景参数
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

float deltaTime = 0.0f;
float lastFrame = 0.0f;

Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
glm::vec3 lightDir(-1.0f, -1.0f, -1.0f);

// 模型参数 modify
std::string model_path = fs::current_path().string() + "\\..\\..\\..\\..\\src\\00_Mini_Render\\Models\\spot_triangulated_good.obj";
std::string texture_path = fs::current_path().string() + "\\..\\..\\..\\..\\src\\00_Mini_Render\\Textures\\spot_texture.png";

int main()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }


    // 构建 & 编译 shader
    std::string cube_vert_path = fs::current_path().string() + "\\..\\..\\..\\..\\src\\00_Mini_Render\\Shaders\\vert.glsl";
    std::string cube_frag_path = fs::current_path().string() + "\\..\\..\\..\\..\\src\\00_Mini_Render\\Shaders\\frag.glsl";
    Shader cubeShader(cube_vert_path.c_str(), cube_frag_path.c_str());


    // 数据输入
    std::vector<float> vertices;
    std::vector<unsigned int> indices;
    load_model(model_path, vertices, indices);

    
    // VBO, VAO
    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), &vertices[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0); // Position
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float))); // Normal
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float))); // UV
    glEnableVertexAttribArray(2);
    glBindVertexArray(0);


    // 纹理输入
    unsigned int diffuseMap = loadTexture(texture_path.c_str());
    unsigned int specularMap = loadTexture(texture_path.c_str());
    
    cubeShader.use(); // 不要忘记在设置uniform变量之前激活着色器程序！
    cubeShader.setInt("material.diffuse", 0);
    cubeShader.setInt("material.specular", 1);
    cubeShader.setFloat("material.shininess", 32.0f);

    // 定向光
    cubeShader.setVec3("light.direction", lightDir);
    cubeShader.setVec3("light.ambient",  0.05f, 0.05f, 0.05f);
    cubeShader.setVec3("light.diffuse",  0.4f, 0.4f, 0.4f);
    cubeShader.setVec3("light.specular", 0.5f, 0.5f, 0.5f);

    // 开启深度测试
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    // 渲染循环
    while (!glfwWindowShouldClose(window))
    {
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        processInput(window);

        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        // 渲染参数 modify
        cubeShader.use();
        // model - view - projection
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::rotate(model, glm::radians(180.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        glm::mat4 view = camera.GetViewMatrix();
        cubeShader.setMat4("model", model);
        cubeShader.setMat4("projection", projection);
        cubeShader.setMat4("view", view);
        cubeShader.setVec3("viewPos", camera.Position);


        glBindVertexArray(VAO);
        // --- texture ---
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, diffuseMap);
        glActiveTexture(GL_TEXTURE0 + 1);
        glBindTexture(GL_TEXTURE_2D, specularMap);
        // glDrawArrays(GL_TRIANGLES, 0, triangle_num);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);


        glfwSwapBuffers(window);
        glfwPollEvents();
    }


    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);

    glfwTerminate();
    return 0;
}


void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(UP, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(DOWN, deltaTime);
}


void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}


void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}


void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}


unsigned int loadTexture(char const * path)
{
    unsigned int textureID;
    glGenTextures(1, &textureID);
    
    int width, height, nrComponents;
    stbi_set_flip_vertically_on_load(true); // 沿Y轴翻转
    unsigned char *data = stbi_load(path, &width, &height, &nrComponents, 0);
    if (data)
    {
        GLenum format;
        if (nrComponents == 1)
            format = GL_RED;
        else if (nrComponents == 3)
            format = GL_RGB;
        else if (nrComponents == 4)
            format = GL_RGBA;

        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexImage2D(GL_TEXTURE_2D, 0, format, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        stbi_image_free(data);
    }
    else
    {
        std::cout << "Texture failed to load at path: " << path << std::endl;
        stbi_image_free(data);
    }

    return textureID;
}

void load_model(const std::string model_path, std::vector<float> & vertices, std::vector<unsigned int> & indices)
{
    objl::Loader loader;
    bool loadOut = loader.LoadFile(model_path);
    
    vertices = {};
    indices = {};

    if (loadOut)
    {
        for (int i = 0; i < loader.LoadedMeshes.size(); i ++)
        {
            objl::Mesh curMesh = loader.LoadedMeshes[i];
            for (int j = 0; j < curMesh.Vertices.size(); j ++)
            {
                vertices.emplace_back(curMesh.Vertices[j].Position.X);
                vertices.emplace_back(curMesh.Vertices[j].Position.Y);
                vertices.emplace_back(curMesh.Vertices[j].Position.Z);

                vertices.emplace_back(curMesh.Vertices[j].Normal.X);
                vertices.emplace_back(curMesh.Vertices[j].Normal.Y);
                vertices.emplace_back(curMesh.Vertices[j].Normal.Z);

                vertices.emplace_back(curMesh.Vertices[j].TextureCoordinate.X);
                vertices.emplace_back(curMesh.Vertices[j].TextureCoordinate.Y);
            }
            for (int j = 0; j < curMesh.Indices.size(); j += 3)
            {
                indices.emplace_back(curMesh.Indices[j]);
                indices.emplace_back(curMesh.Indices[j + 1]);
                indices.emplace_back(curMesh.Indices[j + 2]);
            }
        }
    }
}