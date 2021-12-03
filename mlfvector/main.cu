#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "utility.cuh"

#define ull unsigned long long int
#define BSIZE 64
#define NB 2
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
	int *isbucket;
	__device__ LFVector();
	__device__ void resize(unsigned int n);
	__device__ int& at(unsigned int i);
	__device__ int get_bucket(unsigned int i);
	__device__ void new_bucket(unsigned int b);
	__device__ void push_back(int e);
};

__device__ LFVector::LFVector() {
	size = 0;
	a = (int**)malloc(sizeof(int)*100);
	a[0] = (int*)malloc(sizeof(int*)*32);
	isbucket = (int*)malloc(sizeof(int)*100);
	isbucket[0] = 1;
	for (int i = 1; i < 100; ++i) {
		a[i] = nullptr;
		isbucket[i] = 0;
	}
}

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
		isbucket[i] = 1;
	}
	size = n;
}

__device__ void LFVector::new_bucket(unsigned int b) {
	//printf("inside new_bucket %d\n", b);
	int old = atomicCAS(isbucket + b, 0, 1);
	if (old == 0) {
		int bsize = 1 << (5 + b);
		a[b] = (int*)malloc(sizeof(int) * bsize);
	}
	__syncwarp();
}

__device__ void LFVector::push_back(int e) {
	int idx = atomicAdd(&size, 1);
	int b = get_bucket(idx);
	while (a[b] == nullptr) {
		new_bucket(b);
	}
	at(idx) = e;
}

struct Vector {
	unsigned int size;
	LFVector *lfv;
	unsigned int *ranges;
	__device__ Vector();
	__device__ int& at(unsigned int i);
	__device__ void insert(int e);
};

__device__ Vector::Vector() {
	size = 0;
	lfv = (LFVector*)malloc(sizeof(LFVector)*NB);
	ranges = (unsigned int*)malloc(sizeof(unsigned int)*NB);
	//LFVector a = LFVector();
	//printf("LFV: %d lfv: %d  lfv0: %d\n",
		//sizeof(LFVector) ,sizeof(lfv), sizeof(lfv[1]));
	for (int i = 0; i < NB; ++i) {
		ranges[i] = 0;
		lfv[i] = LFVector();
	}
}

__device__ int& Vector::at(unsigned int i) {
	int b = NB-1;
	while (i < ranges[b]) {
		--b;
	}
	return lfv[b].at(i-ranges[b]);
}

__device__ void Vector::insert(int e) {
	int bid = blockIdx.x;
	lfv[bid].push_back(e);
	atomicAdd(&size, 1);
	for (int i = bid+1; i < NB; ++i) {
		atomicAdd(ranges+i, 1);
	}
}

__global__ void printVec(Vector *v) {
	printf("size: %d\n", v->size);
	printf("ranges: ");
	for (int i = 0; i < NB; ++i) {
		printf("%d ", v->ranges[i]);
	}
	printf("\nsizes: ");
	for (int i = 0; i < NB; ++i) {
		printf("%d ", v->lfv[i].size);
	}
	printf("\n");
	for (int i = 0; i < v->size; ++i) {
		printf("%d ", v->at(i));
	}
	printf("\n");
}

__global__ void initVec(Vector *v) {
	*v = Vector();
	v->size = NB*BSIZE;
	for (int i = 0; i < NB; ++i) {
		v->ranges[i] = BSIZE*i;
		v->lfv[i].resize(BSIZE);
	}
	for(int i = 0; i < v->size; ++i) {
		v->at(i) = i;
	}
}

__global__ void test_insert(Vector *v) {
	int tid = threadIdx.x;
	v->insert(tid);
}


__global__ void random_copy(LFVector *v) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int n = v->size;
	for (int i = tid; i < n; i+=BSIZE) {
		if (v->at(i) < PROB)
			v->push_back(v->at(i));
	}
}

__global__ void init_random(LFVector *v, int *A, int n) {
	v->size = 0;
	v->a = (int**)malloc(sizeof(int)*100);
	v->a[0] = (int*)malloc(sizeof(int*)*32);
	v->isbucket = (int*)malloc(sizeof(int)*100);
	v->isbucket[0] = 1;
	for (int i = 1; i < 100; ++i) {
		v->a[i] = nullptr;
		v->isbucket[i] = 0;
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
	//printLFVec<<<1,1>>>(v); kernelCallCheck();

	cudaEvent_t start, stop;
	start_clock(start, stop);

	//random_copy<<<gridSize(n, BSIZE), BSIZE>>>(v);
	random_copy<<<1, BSIZE>>>(v);
	kernelCallCheck();

	float time =stop_clock(start, stop);
	//printLFVec<<<1,1>>>(v); kernelCallCheck();
	//printf("time: %f ms\n", time);
	printf("%f\n", time);
}

int main(int argc, char **argv){

	cudaDeviceSetLimit(cudaLimitMallocHeapSize, NB*BSIZE*sizeof(int));

	Vector *a;
	gpuErrCheck( cudaMalloc(&a, sizeof(Vector)) );

	initVec<<<1,1>>>(a); kernelCallCheck();
	printVec<<<1,1>>>(a); kernelCallCheck();
	test_insert<<<NB,BSIZE>>>(a); kernelCallCheck();
	printVec<<<1,1>>>(a); kernelCallCheck();
}
