#include "../inc/numeric.h"

namespace py = pybind11;

template <typename T>
T SimpleNum<T>::Add(T y)
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