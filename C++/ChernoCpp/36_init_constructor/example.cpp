#include <iostream>

using namespace std;

#define USE_INIT_CONSTRUCTOR 0

class Pos
{
public:
    float m_x, m_y, m_z;
    Pos(float x, float y, float z)
    {
        m_x = x;
        m_y = y;
        m_z = z;
        cout << "created pos" << " x: " << m_x << ", y:" << m_y << ", z:" << m_z << endl;
    }
    Pos()
    {
        m_x = 0.0;
        m_y = 0.0;
        m_z = 0.0;
        cout << "created pos use default"<< endl;
    }
};

class Example
{
public:
    Pos m_pos;

    #if USE_INIT_CONSTRUCTOR == 1
        Example() : m_pos(Pos(1, 2, 3)) {}
    #else 
        Example() 
        {
            m_pos = Pos(1, 2, 3);
        }
    #endif
};

void test()
{
    Example example;
}

int main()
{
    test();
    return 0;
}