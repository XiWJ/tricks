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