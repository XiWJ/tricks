import PyBoostTest

def test_pyBoost():
    print(PyBoostTest.greet()) # say hello world
    num_a = PyBoostTest.generateZero() # generate num 0 -> num_a

    num_A = PyBoostTest.Num(2.5) # Class Num() -> num_A object
    PyBoostTest.printNum(num_A) # print object num_A

    num_A.set(10.0) # set
    print(num_A.get()) # get

    num_A.clear() # clear data
    PyBoostTest.printNum(num_A)


if __name__ == "__main__":
    test_pyBoost()