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

// --- Camera ---
#include <tools/Camera.h>

#include <iostream>
#include <string>
#include <filesystem>
namespace fs = std::filesystem;

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);
void mouse_callback(GLFWwindow* window, double xpos, double ypos); // 鼠标旋转回调
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset); // 鼠标滚轮回调

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;

float mixValue = 0.2f;
// 相机位置，朝向，向上向量
// glm::vec3 cameraPos   = glm::vec3(0.0f, 0.0f,  3.0f);
// glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
// glm::vec3 cameraUp    = glm::vec3(0.0f, 1.0f,  0.0f);
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f));

float deltaTime = 0.0f; // 当前帧与上一帧的时间差
float lastFrame = 0.0f; // 上一帧的时间
float lastX = (float)SCR_WIDTH / 2.0f, lastY = (float)SCR_HEIGHT / 2.0f; // 上一帧鼠标位置，初始状态在屏幕正中心

bool firstMouse = true;
float pitch = 0.0f;
float yaw = -90.0f; // NOTE: 初始化 yaw 是为 -90° !!!

float fov = 45.0f;

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
    glfwSetCursorPosCallback(window, mouse_callback); // 鼠标位置回调
    glfwSetScrollCallback(window, scroll_callback); // 鼠标滚轮回调

    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); // 隐藏光标 & 捕捉光标

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // 启动 深度测试
    glEnable(GL_DEPTH_TEST);

    // 顶点 & 片元 着色器
    std::string vert_path = fs::current_path().string() + "\\..\\..\\..\\..\\src\\06_Coordinate_Systems\\Shader\\vert.glsl";
    std::string frag_path = fs::current_path().string() + "\\..\\..\\..\\..\\src\\06_Coordinate_Systems\\Shader\\frag.glsl";

    // 加载 & 编译 shader
    Shader ourShader(vert_path.c_str(), frag_path.c_str());

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    float vertices[] = {
        -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,
        0.5f, -0.5f, -0.5f,  1.0f, 0.0f,
        0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
        0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,

        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
        0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
        0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
        0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
        -0.5f,  0.5f,  0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,

        -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

        0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
        0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
        0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
        0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
        0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
        0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
        0.5f, -0.5f, -0.5f,  1.0f, 1.0f,
        0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
        0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,

        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
        0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
        0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
        0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f
    };
    unsigned int indices[] = {
        0, 1, 3, // first triangle
        1, 2, 3  // second triangle
    };
    glm::vec3 cubePositions[] = {
        glm::vec3( 0.0f,  0.0f,  0.0f), 
        glm::vec3( 2.0f,  5.0f, -15.0f), 
        glm::vec3(-1.5f, -2.2f, -2.5f),  
        glm::vec3(-3.8f, -2.0f, -12.3f),  
        glm::vec3( 2.4f, -0.4f, -3.5f),  
        glm::vec3(-1.7f,  3.0f, -7.5f),  
        glm::vec3( 1.3f, -2.0f, -2.5f),  
        glm::vec3( 1.5f,  2.0f, -2.5f), 
        glm::vec3( 1.5f,  0.2f, -1.5f), 
        glm::vec3(-1.3f,  1.0f, -1.5f)  
    };

    // 生成 VAO VBO
    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    // 绑定 VAO
    glBindVertexArray(VAO);
    // 绑定 VBO & 复制数据到 VBO
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // 指定顶点属性
    // 1. pos
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // 2. UV texcroods
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

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
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);   
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
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

        float radius = 10.0f;
        float camX = sin(glfwGetTime()) * radius;
        float camZ = cos(glfwGetTime()) * radius;

        // render
        // ------
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // 清除颜色缓冲 & 深度缓存

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture1);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, texture2);

        /*
        注意：实际的变换顺序应该与阅读顺序相反，比如，代码中我们先位移再旋转，实际的变换却是先应用旋转再是位移的
        */
        // // 创建变换矩阵
        glm::mat4 model = glm::mat4(1.0f); // model
        glm::mat4 view = glm::mat4(1.0f); // view
        glm::mat4 proj = glm::mat4(1.0f); // projection

        // 绕 x-axis 旋转 -55°
        // model = glm::rotate(model, (float)glfwGetTime() * glm::radians(50.0f), glm::vec3(0.5f, 1.0f, 0.0f));
        // lookAt - 相机位置, 注视点, Up向量
        // view = glm::lookAt(glm::vec3(camX, 0.0, camZ), glm::vec3(0.0, 0.0, 0.0), glm::vec3(0.0, 1.0, 0.0)); 
        // WSADQR 表征相机 前后左右前后 平移
        // view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
        view = camera.GetViewMatrix(); // 直接从Camera对象获取view矩阵
        // perspective - fov, aspect radio, near, far
        // proj = glm::perspective(glm::radians(fov), 800.0f / 600.0f, 0.1f, 100.0f);
        proj = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f); // // 直接从Camera对象获取 Zoom == fov

        // 使用 Shader 类对象，方便高效
        ourShader.use();
        ourShader.setFloat("coeff", mixValue);
        // 给 shader 传递 uniform 变量值: model, view, projection matrix
        // unsigned int modelLoc = glGetUniformLocation(ourShader.ID, "model");
        // glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
        unsigned int viewLoc = glGetUniformLocation(ourShader.ID, "view");
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, &view[0][0]);
        // unsigned int projLoc = glGetUniformLocation(ourShader.ID, "proj"); // 这两行等价于 setMat4("name", mat4)
        // glUniformMatrix4fv(projLoc, 1, GL_FALSE, glm::value_ptr(proj));
        ourShader.setMat4("proj", proj);

        glBindVertexArray(VAO);
        // 10 个不同位置的 box 在同一场景下显示
        for(unsigned int i = 0; i < 10; i++)
        {
            glm::mat4 model(1.0f), view(1.0f), proj(1.0f);
            model = glm::translate(model, cubePositions[i]);
            float angle = 20.0f * i; 
            model = glm::rotate(model, glm::radians(angle), glm::vec3(1.0f, 0.3f, 0.5f));
            ourShader.setMat4("model", model);

            glDrawArrays(GL_TRIANGLES, 0, 36);
        }
        glDrawArrays(GL_TRIANGLES, 0, 36);

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // 删除VAO, VBO, EBO
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);

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

    float currentFrame = glfwGetTime();
    deltaTime = currentFrame - lastFrame;
    lastFrame = currentFrame;

    float cameraSpeed = 2.5f * deltaTime; // adjust accordingly
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(UP, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(DOWN, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    // 防止第一帧突然跳变
    if(firstMouse) // 这个bool变量初始时是设定为true的
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // 注意这里是相反的，因为y坐标是从底部往顶部依次增大的
    lastX = xpos;
    lastY = ypos;

    // float sensitivity = 0.05f;
    // xoffset *= sensitivity;
    // yoffset *= sensitivity;

    // yaw   += xoffset;
    // pitch += yoffset;

    // if(pitch > 89.0f) // 在90度时视角会发生逆转
    //     pitch =  89.0f;
    // if(pitch < -89.0f)
    //     pitch = -89.0f;

    // // 计算 俯仰角（绕 X 轴）, 偏航角（绕 Y 轴）
    // glm::vec3 front;
    // front.x = cos(glm::radians(pitch)) * cos(glm::radians(yaw));
    // front.y = sin(glm::radians(pitch));
    // front.z = cos(glm::radians(pitch)) * sin(glm::radians(yaw));
    // cameraFront = glm::normalize(front);

    // 上述 注释 的代码 等价于 下面这一行
    camera.ProcessMouseMovement(xoffset, yoffset);
}

// 修改 fov 实现 Zoom 效果
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    // if(fov >= 1.0f && fov <= 60.0f)
    //     fov -= yoffset;
    // if(fov <= 1.0f)
    //     fov = 1.0f;
    // if(fov >= 60.0f)
    //     fov = 60.0f;
    
    // 上述 注释 的代码 等价于 下面这一行
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}