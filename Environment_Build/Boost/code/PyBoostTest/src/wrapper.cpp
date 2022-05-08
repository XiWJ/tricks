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
	// ��ͨ python ����
	def("greet", greet);
	def("generateZero", &generateZero);
	def("printNum", &printNum);

	// ��
	class_<Num>("Num", init<>()) // init<>()Ĭ�Ϲ���
		.def(init<float>()) // �вι���
		.def("get", &Num::get) // ��Ա����
		.def("set", &Num::set)
		.def("clear", &Num::clear)
	;
}