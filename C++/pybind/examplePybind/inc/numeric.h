#include <pybind11/pybind11.h>

#define FLT_MAX 3.402823466e+38F
#define FLT_MIN 1.175494351e-38F

template <typename T>
class SimpleNum
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