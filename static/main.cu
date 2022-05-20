#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "utility.cuh"

#define ull unsigned long long int
#define BSIZE 1024
#define NB 100
#define PROB 90

struct Vec{
	T *a;
	unsigned int size;
}

__global__ void test_insert(int *a, int *size) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= *size) return;
	int idx = atomicAdd(size, 1);
	a[idx] = tid;
}

int main(int argc, char **argv){

	//cudaDeviceSetLimit(cudaLimitMallocHeapSize, INT_MAX*sizeof(int));

	int rep = 10;
	int *a, *ha;
	int size = 1024*100;
	int *dsize;
	ha = new int[size];
	for (int i = 0; i < size; ++i) {
		ha[i] = i;
	}
	gpuErrCheck( cudaMalloc(&a, 2*size*2^rep*sizeof(int)) );
	gpuErrCheck( cudaMalloc(&dsize, sizeof(int)) );
	gpuErrCheck( cudaMemcpy(a, ha, size*sizeof(int), cudaMemcpyHostToDevice)) ;
	gpuErrCheck( cudaMemcpy(dsize, &size, sizeof(int), cudaMemcpyHostToDevice) );


	float results[rep];
	float s = 0.0;
	for (int i = 0; i < rep; ++i) {
		cudaEvent_t start, stop;
		start_clock(start, stop);
		test_insert<<<gridSize(size, BSIZE), BSIZE>>>(a, dsize); kernelCallCheck();
		results[i] = stop_clock(start, stop);
		s += results[i];
		size *= 2;
	}
	for (int i = 0; i < rep-1; ++i) {
		printf("%f,", results[i]);
	}
	printf("%f\n", results[rep-1]);
	printf("%f\n", s);

	return 0;
}
