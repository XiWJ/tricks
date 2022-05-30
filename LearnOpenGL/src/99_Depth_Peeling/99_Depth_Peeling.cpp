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

void setFbo(unsigned int & fbo, unsigned int & textureColorBuffer, unsigned int & rbo);

// 场景参数
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;

float deltaTime = 0.0f;
float lastFrame = 0.0f;

glm::vec3 lightPos(1.2f, 1.0f, 2.0f);

const int k = 4;
int kt = 0;

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
    std::string cube_vert_path = fs::current_path().string() + "\\..\\..\\..\\..\\src\\99_Depth_Peeling\\Shaders\\vert.glsl";
    std::string cube_frag_path = fs::current_path().string() + "\\..\\..\\..\\..\\src\\99_Depth_Peeling\\Shaders\\frag.glsl";
    Shader cubeShader(cube_vert_path.c_str(), cube_frag_path.c_str());

    std::string screen_vert_path = fs::current_path().string() + "\\..\\..\\..\\..\\src\\99_Depth_Peeling\\Shaders\\screen_vert.glsl";
    std::string screen_frag_path = fs::current_path().string() + "\\..\\..\\..\\..\\src\\99_Depth_Peeling\\Shaders\\screen_frag.glsl";
    Shader screenShader(screen_vert_path.c_str(), screen_frag_path.c_str());


    // 数据输入
    std::string spot_model_path = fs::current_path().string() + "\\..\\..\\..\\..\\src\\99_Depth_Peeling\\Models\\spot_triangulated_good.obj";
    objl::Loader loader;
    bool loadOut = loader.LoadFile(spot_model_path);
    
    std::vector<float> vertices = {};
    int triangle_num = 0;
    if (loadOut)
    {
        for (int i = 0; i < loader.LoadedMeshes.size(); i ++)
        {
            objl::Mesh curMesh = loader.LoadedMeshes[i];
            triangle_num = curMesh.Vertices.size();
            for (int j = 0; j < curMesh.Vertices.size(); j ++)
            {
                vertices.emplace_back(curMesh.Vertices[j].Position.X);
                vertices.emplace_back(curMesh.Vertices[j].Position.Y);
                vertices.emplace_back(curMesh.Vertices[j].Position.Z);

                // vertices.emplace_back(curMesh.Vertices[j].Normal.X);
                // vertices.emplace_back(curMesh.Vertices[j].Normal.Y);
                // vertices.emplace_back(curMesh.Vertices[j].Normal.Z);

                vertices.emplace_back(curMesh.Vertices[j].TextureCoordinate.X);
                vertices.emplace_back(curMesh.Vertices[j].TextureCoordinate.Y);
            }
        }
    }

    float quadVertices[] = { // vertex attributes for a quad that fills the entire screen in Normalized Device Coordinates.
        // positions   // texCoords
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };
    
    
    // cube VBO, VAO
    unsigned int spotVBO, spotVAO;
    glGenVertexArrays(1, &spotVAO);
    glGenBuffers(1, &spotVBO);

    glBindBuffer(GL_ARRAY_BUFFER, spotVBO);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), &vertices[0], GL_STATIC_DRAW);

    glBindVertexArray(spotVAO);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0); // Position
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float))); // UV
    glEnableVertexAttribArray(1);

    // screen quad VBO, VAO
    unsigned int quadVBO, quadVAO;
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), quadVertices, GL_STATIC_DRAW);

    glBindVertexArray(quadVBO);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0); // Position
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float))); // UV
    glEnableVertexAttribArray(1);


    // 生成纹理
    std::string spot_texture_path = fs::current_path().string() + "\\..\\..\\..\\..\\src\\99_Depth_Peeling\\Textures\\spot_texture.png";
    unsigned int spotMap = loadTexture(spot_texture_path.c_str());
    
    cubeShader.use(); // 不要忘记在设置uniform变量之前激活着色器程序！
    cubeShader.setInt("material.diffuse", 0);
    screenShader.use();
    screenShader.setInt("screenTexture", 0);

    unsigned int fbos[k], texs[k], dtexs[k];
    for (int i = 0; i < k; i ++)
    {
        setFbo(fbos[i], texs[i], dtexs[i]);
    }


    // 渲染循环
    while (!glfwWindowShouldClose(window))
    {
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        processInput(window);

        /* pass 1 */
        // 绑定framebuffer，绘制场景到绑定的framebuffer color附件，作为color texture
        glEnable(GL_DEPTH_TEST);
        glDisable(GL_BLEND);
        
        cubeShader.use();
        // --- model - view - projection
        glm::mat4 model = glm::mat4(1.0f);
        model = glm::rotate(model, glm::radians(180.0f + 30.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        glm::mat4 view = camera.GetViewMatrix();
        cubeShader.setMat4("model", model);
        cubeShader.setMat4("projection", projection);
        cubeShader.setMat4("view", view);


        glBindVertexArray(spotVAO);
        // --- texture ---
        for (int pass = 0; pass < k; pass ++)
        {
            const bool first_pass = pass == 0;
            cubeShader.setBool("first_pass", first_pass);
            if (!first_pass)
            {
                cubeShader.setInt("depthTexture", 1);
                glActiveTexture(GL_TEXTURE0 + 1);
                glBindTexture(GL_TEXTURE_2D, dtexs[pass - 1]);
            }
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, spotMap);
            glBindFramebuffer(GL_FRAMEBUFFER, fbos[pass]);
            glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glDrawArrays(GL_TRIANGLES, 0, triangle_num);
        }
        

        // 清理
        glBindVertexArray(0);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, 0);


        /* pass 2 */
        // 绑定回默认framebuffer(main windows) & 绘制with附加的color纹理
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        screenShader.use();
        glBindVertexArray(quadVAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texs[kt % k]); // 使用color附件纹理
        glDrawArrays(GL_TRIANGLES, 0, 6);


        glfwSwapBuffers(window);
        glfwPollEvents();
    }


    glDeleteVertexArrays(1, &spotVAO);
    glDeleteVertexArrays(1, &quadVAO);
    glDeleteBuffers(1, &spotVBO);
    glDeleteBuffers(1, &quadVBO);
    for (int i = 0; i < k; i ++)
    {
        glDeleteFramebuffers(1, &fbos[i]);
    }


    glfwTerminate();
    return 0;
}


void setFbo(unsigned int & fbo, unsigned int & textureColorBuffer, unsigned int & rbo)
{
    /*
        设置FBO
    */
    // 1. 生成FBO
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    // 2. 创建 color 附件纹理 & 绑定
    glGenTextures(1, &textureColorBuffer);
    glBindTexture(GL_TEXTURE_2D, textureColorBuffer);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL); // NULL -- 当前无数据，先空着
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorBuffer, 0);
    glBindTexture(GL_TEXTURE_2D, 0);

    // 3. 创建 renderbuffer & 绑定
    // glGenRenderbuffers(1, &rbo);
    // glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glGenTextures(1, &rbo);
    glBindTexture(GL_TEXTURE_2D, rbo);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32, SCR_WIDTH, SCR_HEIGHT); // 开辟 renderbuffer
    // glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo); // 附加 depth
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, SCR_WIDTH, SCR_HEIGHT, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, NULL); // NULL -- 当前无数据，先空着
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, rbo, 0);

    // 4. 检查framebuffer是否完整
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cout << "ERROR::fbo:: fbo is not complete!" << std::endl;
    glBindFramebuffer(GL_FRAMEBUFFER, 0); // 解绑
    // glBindRenderbuffer(GL_RENDERBUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
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
    if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS)
        kt = 0;
    if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS)
        kt = 1;
    if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS)
        kt = 2;
    if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS)
        kt = 3;
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