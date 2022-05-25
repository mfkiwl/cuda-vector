#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "utility.cuh"

#define ull unsigned long long int
#define BSIZE 1024
#define NB 100
#define PROB 90


inline __device__ int &at(int *a, unsigned int i) {
	return a[i];
}

__device__ void insert_atomic(int *a, int e, int *size, int q) {
	int idx = atomicAdd(size, 1);
	a[idx] = e;
}

__global__ void test_insert_atomic(int* v, int n, int *size) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= n) return;
	insert_atomic(v, at(v, tid), size, 1);
}

__global__ void test_read_write(int* v, int size) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= size) return;
	at(v, tid) += 1;
}

void run_experiment(int size, int ratio) {
	int rep = 10;
	int rw_rep = 30;
	int o_size = size;

	int *a, *ha;
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
	float results_rw[rw_rep];

	for (int i = 0; i < rep; ++i) {
		cudaEvent_t start, stop;
		start_clock(start, stop);
		test_insert_atomic<<<gridSize(size, BSIZE), BSIZE>>>(a, size, dsize);
		cudaDeviceSynchronize();
		results[i] = stop_clock(start, stop);
		cudaMemcpy(&size, dsize, sizeof(int), cudaMemcpyDeviceToHost);

		// read/write
		results_rw[i] = 0.0;
		for (int j = 0; j < rw_rep; ++j) {
			cudaEvent_t start, stop;
			start_clock(start, stop);
			test_read_write<<<gridSize(size, 1024), 1024>>>(a, size);
			cudaDeviceSynchronize();
			results_rw[i] += stop_clock(start, stop);
		}
		results_rw[i] /= rw_rep;
	}

	// print results
	printf("static,in,%d,%d,", o_size, ratio);
	for (int i = 0; i < rep-1; ++i) {
		printf("%f,", results[i]);
	}
	printf("%f\n", results[rep-1]);
	//printf("%f\n", s);
	printf("static,rw,%d,%d,", o_size, ratio);
	for (int i = 0; i < rep-1; ++i) {
		printf("%f,", results_rw[i]);
	}
	printf("%f\n", results_rw[rep-1]);
}

int main(int argc, char **argv){

	//cudaDeviceSetLimit(cudaLimitMallocHeapSize, INT_MAX*sizeof(int));

	int size = 1<<19;
	int ratio = 1;

	run_experiment(size, ratio);

	return 0;
}
