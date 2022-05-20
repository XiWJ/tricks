#include <pybind11/pybind11.h>

#include <../inc/numeric.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

int add(int i, int j) // add func, pybind hello world.
{
    return i + j;
}

template <typename T> // 模板函数
T mul(T x, T y)
{
    return x * y;
}

void class_ext(py::module &m); // 分文件绑定，先声明，其他源文件会定义
void vector_ext(py::module &m); // 对vector -> python list


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

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    m.def("mul", mul<int>, py::arg("x")=1, py::arg("y")=2); // 模板函数 int
    m.def("mul", mul<float>, py::arg("x")=2.0, py::arg("y")=3.0); // 模板函数 float
    m.def("mul", mul<double>, py::arg("x")=3.0, py::arg("y")=4.0); // 模板函数 double

    class_ext(m); // 绑定class
    vector_ext(m); // 绑定vector

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}