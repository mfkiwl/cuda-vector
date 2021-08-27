#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "utility.cuh"

#define ull unsigned long long int
#define BSIZE 1024
#define PROB 90


inline __device__ int log2i32(unsigned int n){
	return __clz(n) ^ 31;
}

inline __device__ int log2i64(unsigned long long int n){
	return __clzll(n) ^ 63;
}

struct LFVector {
	unsigned int size;
	int **a;
	LFVector();
	__device__ void resize(unsigned int n);
	__device__ int& at(unsigned int i);
	__device__ int get_bucket(unsigned int i);
	__device__ void new_bucket(unsigned int b);
	__device__ void push_back(int e);
};


__device__ int LFVector::get_bucket(unsigned int i) {
	return log2i32(i + 32) - log2i32(32);
}

__device__ int& LFVector::at(unsigned int i) {
	int b = get_bucket(i);
	int pos = i + 32;
	int idx = pos ^ (1 << log2i32(pos));
	return a[b][idx];
}

__device__ void LFVector::resize(unsigned int n) {
	int b1 = get_bucket(size);
	int b2 = get_bucket(n);
	for (int i = b1+1; i <= b2; ++i) {
		int bsize = 1 << (5 + i);
		a[i] = (int*)malloc(sizeof(int) * bsize);
	}
	size = n;
}

__device__ void LFVector::new_bucket(unsigned int b) {
	int bsize = 1 << (5 + b);
	int *aux = (int*)malloc(sizeof(int) * bsize);
	int *nil = nullptr;
	int **addr = &(a[b]);
	ull old = atomicCAS(* (ull **) &addr,
			    * (ull *) &nil,
			    * (ull *) &aux);
	if ((* (int **) &old) != nullptr) {
		free(aux);
	}
}

__device__ void LFVector::push_back(int e) {
	int idx = atomicAdd(&size, 1);
	int b = get_bucket(idx);
	if (a[b] == nullptr) {
		new_bucket(b);
	}
	at(idx) = e;
}

__global__ void printVec(LFVector *v) {
	printf("size: %d\n", v->size);
	return;
	for (int i = 0; i < v->size; ++i) {
		printf("%d ", v->at(i));
	}
	printf("\n");
}

__global__ void initVec(LFVector *v, int n) {
	v->size = 0;
	v->a = (int**)malloc(sizeof(int)*100);
	v->a[0] = (int*)malloc(sizeof(int*)*32);
	for (int i = 1; i < 100; ++i) {
		v->a[i] = nullptr;
	}
	v->resize(n);
	for (int i = 0; i < n; ++i) {
		v->at(i) = i;
	}
}

__global__ void test_push_back(LFVector *v) {
	int tid = threadIdx.x;
	v->push_back(tid);
}


__global__ void random_copy(LFVector *v) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= v->size) return;

	if (v->at(tid) < PROB) {
		v->push_back(v->at(tid));
	}
}

__global__ void init_random(LFVector *v, int *A, int n) {
	v->size = 0;
	v->a = (int**)malloc(sizeof(int)*100);
	v->a[0] = (int*)malloc(sizeof(int*)*32);
	for (int i = 1; i < 100; ++i) {
		v->a[i] = nullptr;
	}
	v->resize(n);
	for (int i = 0; i < n; ++i) {
		v->at(i) = A[i];
	}
}

void test_random_copy(LFVector *v, int n) {
	int *hA, *dA;
	hA = new int[n];
	for (int i = 0; i < n; ++i) {
		hA[i] = rand() % 101;
	}

	cudaMalloc(&dA, sizeof(int)*n);
	cudaMemcpy(dA, hA, sizeof(int)*n, cudaMemcpyHostToDevice);
	init_random<<<1,1>>>(v, dA, n);

	cudaEvent_t start, stop;
	start_clock(start, stop);

	random_copy<<<gridSize(n, BSIZE), BSIZE>>>(v);

	float time =stop_clock(start, stop);
	printVec<<<1,1>>>(v); kernelCallCheck();
	printf("time: %f ms\n", time);
}

int main(int argc, char **argv){
	if (argc < 2){
		fprintf(stderr, "execute as %s <n>\n", argv[0]);
		return -1;
	}

	int n = atoi(argv[1]);

	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 4*n*sizeof(int));

	LFVector *a;
	gpuErrCheck( cudaMalloc(&a, sizeof(LFVector)) );

	test_random_copy(a, n);
	//initVec<<<1,1>>>(a, n); kernelCallCheck();
	//printVec<<<1,1>>>(a); kernelCallCheck();
	//test_push_back<<<4,64>>>(a); kernelCallCheck();
	//printVec<<<1,1>>>(a); kernelCallCheck();
}


// index 0  1   2   3   4   5
// start 0  32 96  224 480 992
// bsize 32 64 128 256 512 1024
