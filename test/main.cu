#include <iostream>
#include <cuda.h>
#include "../common/utility.cuh"

#define BSIZE 1024

void fun1() {
	int x = 0;
	for (int i = 0; i < 1000; ++i) {
		x++;
	}
	std::cout << "fun1" << std::endl;
}

void fun2() {
	int x = 0;
	for (int i = 0; i < 10000000; ++i) {
		x++;
	}
	std::cout << "fun2" << std::endl;
}

template <int N, class C>
float measure_time(float a[N], C fun) {
	float s = 0.0;
	int rep = 10;
	for (int i = 0; i < rep; ++i) {
		cudaEvent_t start1, stop1;
		start_clock(start1, stop1);
		{
			fun();
		}
		s += stop_clock(start1, stop1);
	}
	std::cout << N << std::endl;
	return s / rep;
}

int main(int argc, char* argv[]) {
	
	float a[10];
	float time = measure_time(a, []{
		fun1();
		fun2();}
	);
	std::cout << time << std::endl;

	return 0;
}
