#pragma once

class Num
{
public:
	Num()=default;
	Num(float x) : m_x(x) {}

	float get();
	void set(float num);
	void clear();

private:
	float m_x;
};
