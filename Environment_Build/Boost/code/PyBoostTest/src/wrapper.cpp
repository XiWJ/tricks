#define BOOST_PYTHON_STATIC_LIB
#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>

#include <iostream>
#include "Num.h"

using namespace boost::python;
using namespace std;

char const* greet() 
{
	return "hello, world";
}

int generateZero()
{
	return 0;
}

void printNum(Num & num)
{
	std::cout << num.get() << std::endl;
}

BOOST_PYTHON_MODULE(PyBoostTest)
{
	// 普通 python 函数
	def("greet", greet);
	def("generateZero", &generateZero);
	def("printNum", &printNum);

	// 类
	class_<Num>("Num", init<>()) // init<>()默认构造
		.def(init<float>()) // 有参构造
		.def("get", &Num::get) // 成员函数
		.def("set", &Num::set)
		.def("clear", &Num::clear)
	;
}