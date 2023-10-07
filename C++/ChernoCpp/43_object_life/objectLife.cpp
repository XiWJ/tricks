#include <iostream>
using namespace std;

class Entity
{
public:
    Entity()
    {
        cout << "created Entity" << endl;
    }

    ~Entity()
    {
        cout << "destroyed Entity" << endl;
    }
};

class ScopedPtr
{
private:
    Entity* m_Ptr;
public:
    ScopedPtr(Entity* e) : m_Ptr(e) {}
    ~ScopedPtr()
    {
        delete m_Ptr;
    }
};

int main()
{
    // {
    //     Entity e;
    // }

    // {
    //     Entity* e = new Entity();
    // }

    {
        ScopedPtr e = new Entity();
    }
    return 0;
}