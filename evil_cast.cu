#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "utility.cuh"

#define ull unsigned long long int

__global__ void evil_cast() {
	int **a = (int**)malloc(sizeof(int*));
	a[0] = (int*)malloc(sizeof(int) * 3);
	int **b = (int**)malloc(sizeof(int*));
	b[0] = nullptr;
	int *nil = nullptr;
	//b[0] = (int*)malloc(sizeof(int) * 2);
	//b[0][0] = 22; b[0][1] = 33;
	a[0][0] = 10; a[0][1] = 5; a[0][2] = 3;
	printf("before\n\ta: %p, b: %p\n", a, b);
	printf("\ta[0]: %p, b[0]: %p\n", a[0], b[0]);
	ull old = atomicCAS(* ( ull **) &b,
			    * ( ull * ) &nil,
			    * ( ull * ) &a[0]);
	//ull asd = atomicCAS(old, cmp, val);
	
	printf("after\n\ta: %p, b: %p, old: %p\n", a, b, * (int **) &old);
	printf("\ta[0][0]: %d, b[0][0]: %d\n", a[0][0], b[0][0]);
}

int main() {
	evil_cast<<<1,1>>>(); kernelCallCheck();
	return 0;
}
