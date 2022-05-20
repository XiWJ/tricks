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