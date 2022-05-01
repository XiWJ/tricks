#include <glad/glad.h>
#include <GLFW/glfw3.h>

// --- shader & texture ---
#include <tools/shader_s.h>
#define STB_IMAGE_IMPLEMENTATION
#include <tools/stb_image.h>

// --- matrix computation ---
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <iostream>
#include <string>
#include <filesystem>
namespace fs = std::filesystem;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

float mixValue = 0.2f;

int main()
{
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // 顶点 & 片元 着色器
    std::string vert_path = fs::current_path().string() + "\\..\\..\\..\\..\\src\\05_Transformations\\Shader\\vert.glsl";
    std::string frag_path = fs::current_path().string() + "\\..\\..\\..\\..\\src\\05_Transformations\\Shader\\frag.glsl";

    // 加载 & 编译 shader
    Shader ourShader(vert_path.c_str(), frag_path.c_str());

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    float vertices[] = {
    //     ---- 位置 ----       ---- 颜色 ----     - 纹理坐标 -
        0.5f,  0.5f, 0.0f,   1.0f, 0.0f, 0.0f,   2.0f, 2.0f,   // 右上
        0.5f, -0.5f, 0.0f,   0.0f, 1.0f, 0.0f,   2.0f, 0.0f,   // 右下
        -0.5f, -0.5f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f,   // 左下
        -0.5f,  0.5f, 0.0f,   1.0f, 1.0f, 0.0f,   0.0f, 2.0f    // 左上
    };
    unsigned int indices[] = {
        0, 1, 3, // first triangle
        1, 2, 3  // second triangle
    };

    // 生成 VAO VBO
    unsigned int VBOs[2], VAOs[2], EBO;
    glGenVertexArrays(1, VAOs);
    glGenBuffers(1, VBOs);
    glGenBuffers(1, &EBO);
    // 绑定 VAO
    glBindVertexArray(VAOs[0]);
    // 绑定 VBO & 复制数据到 VBO
    glBindBuffer(GL_ARRAY_BUFFER, VBOs[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // 指定顶点属性
    // 1. pos
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // 2. color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // 3. UV texcroods
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    // 绑定EBO, 同时拷贝数据到GL_ELEMENT_ARRAY_BUFFER
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // 解绑VBO
    glBindBuffer(GL_ARRAY_BUFFER, 0); 
    // 解绑VAO
    glBindVertexArray(0);

    // 绑定 VAO
    glBindVertexArray(VAOs[1]);
    // 绑定 VBO & 复制数据到 VBO
    glBindBuffer(GL_ARRAY_BUFFER, VBOs[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // 指定顶点属性
    // 1. pos
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // 2. color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // 3. UV texcroods
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);

    // 绑定EBO, 同时拷贝数据到GL_ELEMENT_ARRAY_BUFFER
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // 解绑VBO
    glBindBuffer(GL_ARRAY_BUFFER, 0); 
    // 解绑VAO
    glBindVertexArray(0);

    // 生成纹理
    unsigned int texture1;
    glGenTextures(1, &texture1);
    // 绑定纹理
    glBindTexture(GL_TEXTURE_2D, texture1);
    // 沿Y轴翻转图片
    stbi_set_flip_vertically_on_load(true);
    // 为当前绑定的纹理对象设置 环绕
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);   
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // 为当前绑定的纹理对象设置 过滤方式
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // 加载纹理图片
    int width, height, nrChannels;
    std::string texture1_image_path = fs::current_path().string() + "\\..\\..\\..\\..\\src\\04_Textures\\container.jpg";
    unsigned char *data = stbi_load(texture1_image_path.c_str(), &width, &height, &nrChannels, 0);

    if (data)
    {
        // 载入图片成为纹理
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D); // 自动生成mipmap
    }
    else
    {
        std::cout << "Failed to load texture1" << std::endl;
    }

    // 释放内存
    stbi_image_free(data);

    // 生成纹理
    unsigned int texture2;
    glGenTextures(1, &texture2);
    // 绑定纹理
    glBindTexture(GL_TEXTURE_2D, texture2);
    // 沿Y轴翻转图片
    stbi_set_flip_vertically_on_load(true);
    // 为当前绑定的纹理对象设置 环绕
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);   
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
    // 为当前绑定的纹理对象设置 过滤方式
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    // 加载纹理图片
    std::string texture2_image_path = fs::current_path().string() + "\\..\\..\\..\\..\\src\\04_Textures\\awesomeface.png";
    data = stbi_load(texture2_image_path.c_str(), &width, &height, &nrChannels, 0);

    if (data)
    {
        // 载入图片成为纹理
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D); // 自动生成mipmap
    }
    else
    {
        std::cout << "Failed to load texture1" << std::endl;
    }

    // 释放内存
    stbi_image_free(data);

    // 在渲染循环之前 告诉着色采样器属于哪个纹理单元
    ourShader.use(); // 不要忘记在设置uniform变量之前激活着色器程序！
    glUniform1i(glGetUniformLocation(ourShader.ID, "ourTexture1"), 0); // 手动设置，"ourShader.setInt("texture2", 0);"这样写也ok
    ourShader.setInt("ourTexture2", 1); // 或者使用着色器类设置

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        // input
        // -----
        processInput(window);

        float timeValue = glfwGetTime();
        float movingValue = sin(timeValue) + 1.0f;
        float rotateValue = (sin(timeValue) / 2.0f) + 0.5f;
        float scaleValue = (sin(timeValue) / 2.0f) + 0.5f;

        // render
        // ------
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture1);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, texture2);

        /*
        注意：实际的变换顺序应该与阅读顺序相反，比如，代码中我们先位移再旋转，实际的变换却是先应用旋转再是位移的
        */
        // 创建变换矩阵 trans
        glm::mat4 trans = glm::mat4(1.0f);
        // 平移 (1, 0, 0)
        trans = glm::translate(trans, glm::vec3(-1.0f + movingValue, -1.0f + movingValue, 0.0f));
        // 绕 (0, 0, 1) 旋转 90 度
        trans = glm::rotate(trans, glm::radians(2 * 360.0f * rotateValue), glm::vec3(0.0, 0.0, 1.0));
        // 缩放 各方向缩小到原来 0.5
        trans = glm::scale(trans, glm::vec3(0.5, 0.5, 0.5));

        // 使用 Shader 类对象，方便高效
        ourShader.use();
        ourShader.setFloat("coeff", mixValue);
        // 给 shader 传递 uniform 变量值
        unsigned int transformLoc = glGetUniformLocation(ourShader.ID, "transform"); // 首先查询uniform变量的地址
        glUniformMatrix4fv(transformLoc, 1, GL_FALSE, glm::value_ptr(trans));

        glBindVertexArray(VAOs[0]);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        trans = glm::mat4(1.0f);
        trans = glm::translate(trans, glm::vec3(-0.5f, 0.5f, 0.0f));
        trans = glm::scale(trans, glm::vec3(scaleValue, scaleValue, scaleValue));
        // 给 shader 传递 uniform 变量值
        ourShader.setFloat("coeff", 1.0f - mixValue);
        transformLoc = glGetUniformLocation(ourShader.ID, "transform"); // 首先查询uniform变量的地址
        glUniformMatrix4fv(transformLoc, 1, GL_FALSE, glm::value_ptr(trans));
        glBindVertexArray(VAOs[1]);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // 删除VAO, VBO, EBO
    glDeleteVertexArrays(1, VAOs);
    glDeleteBuffers(1, VBOs);
    glDeleteBuffers(1, &EBO);

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
    {
        mixValue += 0.001f; // change this value accordingly (might be too slow or too fast based on system hardware)
        if(mixValue >= 1.0f)
            mixValue = 1.0f;
    }
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
    {
        mixValue -= 0.001f; // change this value accordingly (might be too slow or too fast based on system hardware)
        if (mixValue <= 0.0f)
            mixValue = 0.0f;
    }
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}