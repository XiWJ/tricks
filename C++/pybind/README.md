# Python using C++ library with Pybin11
## 0 Why pybind11?
pybind11与Boost::Python一样可以作为C++ => Python的桥梁，但是Boost太重了，引用pybin11官方说法"Boost is an enormously large and complex suite of utility libraries that works with almost every C++ compiler in existence", 这带来的代价就是，Boost::Python里有很多不必要、不实用的代码，因此pybind11这类轻量级（仅头文件库只需要大约 4K 行代码）就有了用武之地。

## 1 build environment
1. 安装pybind11

    ```bash
    git clone https://github.com/pybind/pybind11.git
    ```

2. 构建开发文件夹

    在`pybind11`同级目录下，构建如果工程文件结构
    ```bash
    - root
        - src
            main.cpp
            *.cpp
        - inc
            *.h
        - pybind11
        CMakeLists.txt
    ```

3. 设置CMakeLists.txt

    ```cmake
    cmake_minimum_required(VERSION 3.4...3.18)
    project(example)

    # C++ 17
    set(CMAKE_CXX_STANDARD 17)

    # include
    set(INCLUDE_H ${CMAKE_CURRENT_SOURCE_DIR}/inc)
    include_directories(${INCLUDE_H})

    # src
    aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src DIR_SRCS)

    # pybind11
    add_subdirectory(pybind11) # 包含子目录 pybind11 https://github.com/pybind/pybind11.git
    pybind11_add_module(example ${DIR_SRCS})

    # version setup
    set(EXAMPLE_VERSION_INFO 1.0) # 设定版本号
    target_compile_definitions(example
                            PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
    ```

## 2 C++开发 - Python使用
以下全部代码见[examplePybind](./examplePybind/)

### 2.1 say hello
    
`say hello`是`pybind11`官方文档中一个hello world测试案例
    
    
```c++
#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

int add(int i, int j) // add func, pybind hello world.
{
    return i + j;
}

PYBIND11_MODULE(example, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cmake_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc"; // 说明文档

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc"); // 绑定 add 函数

#ifdef VERSION_INFO // 版本号
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
```

### 2.2 build py lib
采用CMake方式build

- windows
```bash
mkdir build
cd build
cmake "Visual Studio 15 2022" -A x64 ..
cmake --build . --config Release -- /maxcpucount:8
```

- osx
```bash
mkdir build
cd build
cmake ..
make -j8
```

在`build/Release/`文件夹下可以找到`example.pyd`或者`example.so`文件

### 2.3 python import
```python
>>> import example
>>> print(example.add(1, 2))
>>> 3
```

### 2.4 template function
当然仅仅只是跑通官方测试案例还是不过瘾的，下面对模板函数进行构建

`main.cpp`
```c++
template <typename T> // 模板函数
T mul(T x, T y)
{
    return x * y;
}

PYBIND11_MODULE(example, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cmake_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    m.def("mul", mul<int>, py::arg("x")=1, py::arg("y")=2); // 模板函数 int
    m.def("mul", mul<float>, py::arg("x")=2.0, py::arg("y")=3.0); // 模板函数 float
    m.def("mul", mul<double>, py::arg("x")=3.0, py::arg("y")=4.0); // 模板函数 double

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
```

`examplePybind.py`
```python
import example
print("example version: ", example.__version__)

def test_func():
    print("mul() = {}".format(example.mul())) # 默认参数 int x, y = 1, 2
    print("mul(12, 3) = {}".format(example.mul(12, 3))) # int x, y = 12, 3
    print("mul(12.0, 2.0) = {}".format(example.mul(12.0, 2.0))) # float x, y = 12.0, 2.0

if __name__ == "__main__":
    test_func()
```

结果
```bash
mul() = 2
mul(12, 3) = 36     
mul(12.0, 2.0) = 24.0
```

### 2.5 template class
更进一步的，模板 + 类 + 分文件编写

分别支持`int`和`float`类型的微型数值计算的类。

`numeric.h`
```c++
#include <pybind11/pybind11.h>

#define FLT_MAX 3.402823466e+38F
#define FLT_MIN 1.175494351e-38F

template <typename T>
class SimpleNum // 类声明
{
public:
    T m_x;

    SimpleNum();
    SimpleNum(T x): m_x(x) {}
    T Add(T y);
    T Sub(T y);
    T Mul(T y);
    T Div(T y);
    void Set(T x);
    T Get();

private:
    bool _IsZero(T y);

};
```

`numeric.cpp`
```c++
#include "../inc/numeric.h"

namespace py = pybind11;

template <typename T>
T SimpleNum<T>::Add(T y) // 具体实现
{
    return this->m_x + y;
}

template <typename T>
T SimpleNum<T>::Sub(T y)
{
    return this->m_x - y;
}

template <typename T>
T SimpleNum<T>::Mul(T y)
{
    return this->m_x * y;
}

template <typename T>
T SimpleNum<T>::Div(T y)
{
    if (_IsZero(y))
    {
        return FLT_MAX;
    }
    return this->m_x / y;
}

template <typename T>
void SimpleNum<T>::Set(T x)
{
    this->m_x = x;
}

template <typename T>
T SimpleNum<T>::Get()
{
    return this->m_x;
}

template <typename T>
bool SimpleNum<T>::_IsZero(T y)
{
    return y == 0;
}

// 就地绑定, 会在main.cpp对class_ext也进行绑定
void class_ext(py::module &m) {
    py::class_<SimpleNum<int> > (m, "SimpleNum")
        .def(py::init<int>())
        .def_readwrite("x", &SimpleNum<int>::m_x) // 并不建议直接访问成员变量x, 安全起见用Set(), Get()
        .def("Add", &SimpleNum<int>::Add)
        .def("Sub", &SimpleNum<int>::Sub)
        .def("Mul", &SimpleNum<int>::Mul)
        .def("Div", &SimpleNum<int>::Div)
        .def("Set", &SimpleNum<int>::Set)
        .def("Get", &SimpleNum<int>::Get);
    
    py::class_<SimpleNum<float> > (m, "SimpleNumFLT")
        .def(py::init<float>())
        .def_readwrite("x", &SimpleNum<float>::m_x) // 并不建议直接访问成员变量x, 安全起见用Set(), Get()
        .def("Add", &SimpleNum<float>::Add)
        .def("Sub", &SimpleNum<float>::Sub)
        .def("Mul", &SimpleNum<float>::Mul)
        .def("Div", &SimpleNum<float>::Div)
        .def("Set", &SimpleNum<float>::Set)
        .def("Get", &SimpleNum<float>::Get);
}
```

`main.cpp`
```c++
void class_ext(py::module &m); // 分文件绑定，先声明，其他源文件会定义

PYBIND11_MODULE(example, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: cmake_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";

    class_ext(m); // 绑定class

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
```

`examplePybind.py`
```python
import example
print("example version: ", example.__version__)

def test_class():
    num = example.SimpleNum(3) # 类 -> 对象
    print("num = {}".format(num.x))
    print("{} + 4 = {}".format(num.x, num.Add(4))) # 成员函数 Add
    print("{} - 4 = {}".format(num.x, num.Sub(4))) # 成员函数 Sub
    print("{} * 4 = {}".format(num.x, num.Mul(4))) # 成员函数 Mul
    print("{} / 4 = {}".format(num.x, num.Div(4))) # 成员函数 Div
    print("num.Set(99)")
    num.Set(99) # 成员函数 设置成员变量x
    print("num.Get(): {}".format(num.Get())) # Get函数 获取成员变量x

    numFlt = example.SimpleNumFLT(30.0) # 类 -> 对象，float类型
    print("numFlt = {}".format(numFlt.x))
    print("{} + 4.0 = {}".format(numFlt.x, numFlt.Add(4.0)))
    print("{} - 4.0 = {}".format(numFlt.x, numFlt.Sub(4.0)))
    print("{} * 4.0 = {}".format(numFlt.x, numFlt.Mul(4.0)))
    print("{} / 4.0 = {}".format(numFlt.x, numFlt.Div(4.0)))
    print("numFlt.Set(199.0)")
    numFlt.Set(199.0)
    print("numFlt.Get(): {}".format(numFlt.Get()))

if __name__ == "__main__":
    test_class()
```

结果
```bash
num = 3
3 + 4 = 7
3 - 4 = -1
3 * 4 = 12
3 / 4 = 0
num.Set(99)
num.Get(): 99
numFlt = 30.0
30.0 + 4.0 = 34.0
30.0 - 4.0 = 26.0
30.0 * 4.0 = 120.0
30.0 / 4.0 = 7.5
numFlt.Set(199.0)
numFlt.Get(): 199.0
```

### 2.6 C++ vector - Python list
C++中的vector与python list实际上可以做一个对应

实现一个针对数组的平方函数，支持`int`和`float`类型

`vectorTest.h`
```c++
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <vector>

using namespace std;
namespace py = pybind11;

template <typename T>
vector<T> vecSquare(vector<T> vec);
```

`vectorTest.cpp`
```c++
#include "../inc/vectorTest.h"

template <typename T>
vector<T> vecSquare(vector<T> vec)
{
    vector<T> ret;
    for (int i=0; i < vec.size(); i++)
    {
        ret.emplace_back(vec[i] * vec[i]);
    }
    return ret;
}

// 就地绑定, 会在main.vector_ext
void vector_ext(py::module &m) {
    m.def("Square", &vecSquare<int>);
    m.def("Square", &vecSquare<float>);
    m.def("Square", &vecSquare<double>);
}
```

`examplePybind.py`
```python
def test_vector():
    b = [1, 2, 3]
    print("b^2 = {}".format(example.Square(b)))
    c = [4.0, 5.0, 6.0]
    print("c^2 = {}".format(example.Square(c)))

if __name__ == "__main__":
    test_vector()
```

结果
```
b^2 = [1, 4, 9]
c^2 = [16.0, 25.0, 36.0]
```