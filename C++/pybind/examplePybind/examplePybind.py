import sys, os
sys.path.append("./build/Release/")

import example
print("example version: ", example.__version__)

def say_hello():
    print("add(1, 2) = {}".format(example.add(1, 2)))

def test_func():
    print("mul() = {}".format(example.mul())) # 默认参数 int x, y = 1, 2
    print("mul(12, 3) = {}".format(example.mul(12, 3))) # int x, y = 12, 3
    print("mul(12.0, 2.0) = {}".format(example.mul(12.0, 2.0))) # float x, y = 12.0, 2.0

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

def test_vector():
    b = [1, 2, 3]
    print("b^2 = {}".format(example.Square(b)))
    c = [4.0, 5.0, 6.0]
    print("c^2 = {}".format(example.Square(c)))

if __name__ == "__main__":
    say_hello()
    print("*"*20 + "\n")
    test_func()
    print("*"*20 + "\n")
    test_class()
    print("*"*20 + "\n")
    test_vector()