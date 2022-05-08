#include "Num.h"

float Num::get()
{
	return m_x;
}

void Num::set(float num)
{
	m_x = num;
}

void Num::clear()
{
	m_x = 0.0f;
}