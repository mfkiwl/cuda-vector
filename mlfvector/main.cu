#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "utility.cuh"

#define ull unsigned long long int
#define BSIZE 1024
#define NB 64
#define PROB 90
#define FBS 1024
#define logFBS 10

inline __device__ int log2i32(unsigned int n){
	return __clz(n) ^ 31;
}

inline __device__ int log2i64(unsigned long long int n){
	return __clzll(n) ^ 63;
}

// LFVector
template <typename T>
struct LFVector {
	unsigned int size;
	T **a;
	int *isbucket;
	__device__ LFVector();
	__device__ void resize(unsigned int n);
	__device__ T& at(unsigned int i);
	__device__ int get_bucket(unsigned int i);
	__device__ void new_bucket(unsigned int b);
	__device__ void push_back(T e);
};

template <typename T>
__device__ LFVector<T>::LFVector() {
	size = 0;
	a = (T**)malloc(sizeof(T*)*64);
	a[0] = (T*)malloc(sizeof(T)*FBS);
	isbucket = (int*)malloc(sizeof(int)*64);
	isbucket[0] = 1;
	for (int i = 1; i < 64; ++i) {
		a[i] = nullptr;
		isbucket[i] = 0;
	}
}

template <typename T>
__device__ int LFVector<T>::get_bucket(unsigned int i) {
	return log2i32(i + FBS) - log2i32(FBS);
}

template <typename T>
__device__ T& LFVector<T>::at(unsigned int i) {
	int b = get_bucket(i);
	int pos = i + FBS;
	int idx = pos ^ (1 << log2i32(pos));
	return a[b][idx];
}

template <typename T>
__device__ void LFVector<T>::resize(unsigned int n) {
	int b1 = get_bucket(size);
	int b2 = get_bucket(n);
	for (int i = b1+1; i <= b2; ++i) {
		int bsize = 1 << (logFBS + i);
		a[i] = (T*)malloc(sizeof(T) * bsize);
		isbucket[i] = 1;
	}
	size = n;
}

template <typename T>
__device__ void LFVector<T>::new_bucket(unsigned int b) {
	//printf("inside new_bucket %d\n", b);
	int old = atomicCAS(isbucket + b, 0, 1);
	if (old == 0) {
		int bsize = 1 << (logFBS + b);
		a[b] = (T*)malloc(sizeof(T) * bsize);
	}
	__syncthreads();
}

template <typename T>
__device__ void LFVector<T>::push_back(T e) {
	int idx = atomicAdd(&size, 1);
	int b = get_bucket(idx);
	while (a[b] == nullptr) {
		new_bucket(b);
	}
	at(idx) = e;
}

template <typename T>
struct Vector {
	unsigned int size;
	LFVector<T> *lfv;
	unsigned int *ranges;
	__device__ Vector();
	__device__ T& at(unsigned int i);
	__device__ void insert(T e, int q);
};

template <typename T>
__device__ Vector<T>::Vector() {
	size = 0;
	lfv = (LFVector<T>*)malloc(sizeof(LFVector<T>)*NB);
	ranges = (unsigned int*)malloc(sizeof(unsigned int)*NB);
	for (int i = 0; i < NB; ++i) {
		ranges[i] = 0;
		lfv[i] = LFVector<T>();
	}
}

template <typename T>
__device__ T& Vector<T>::at(unsigned int i) {
	// TODO use warp instructions
	int b = NB-1;
	while (i < ranges[b]) {
		--b;
	}
	return lfv[b].at(i-ranges[b]);
}

template <typename T>
__device__ void Vector<T>::insert(T e, int q) {
	__shared__ int inserted;
	if (q == 0 && threadIdx.x != 0)
		return;
	if (threadIdx.x == 0)
		inserted = 0;
	__syncthreads();
	int bid = blockIdx.x;
	lfv[bid].push_back(e);
	atomicAdd(&inserted, 1);
	__syncthreads();
	if (threadIdx.x > blockIdx.x && threadIdx.x < NB)
		atomicAdd(ranges+threadIdx.x, inserted);
	//if (threadIdx.x == 0)
		//atomicAdd(&size, inserted);
		atomicAdd(&size, 1);
}

template <typename T>
__global__ void createLFVector(Vector<T> *v) {
	*v = Vector<T>();
	return;
	v->size = 0;
	for (int i = 0; i < NB; ++i) {
		v->ranges[i] = 0;
		v->lfv[i].resize(0);
	}
}

template <typename T>
__global__ void initVec(Vector<T> *v, unsigned int n) {
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int bsize = n / NB;
	int start;
	if (bid < n % NB) {
		++bsize;
		start = bsize * bid;
	} else {
		start = bsize * bid + n%NB;
	}
	for (int i = tid; i < bsize && bid*NB+i < n; i += BSIZE) {
		v->insert(start+i, 1);
	}
}

template <typename T>
__global__ void initVec(Vector<T> *v, unsigned int n, T* in) {
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int bsize = n / NB;
	int start;
	if (bid < n % NB) {
		++bsize;
		start = bsize * bid;
	} else {
		start = bsize * bid + n%NB;
	}
	for (int i = tid; i < bsize && bid*NB+i < n; i += BSIZE) {
		v->insert(in[start+i], 1);
	}
}

template<typename T>
__global__ void get_size(int *out, Vector<T> *v) {
	*out = v->size;
}

template<typename T>
__global__ void vec2array(T *out, Vector<T> *v) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	out[tid] = v->at(tid);
}

template<typename T>
int sendToHost(T* &out, Vector<T> *v) {
	int *ds, size, *temp;
	gpuErrCheck( cudaMalloc(&ds, sizeof(int)) );
	get_size<<<1,1>>>(ds, v);
	gpuErrCheck( cudaMemcpy(&size, ds, sizeof(int), cudaMemcpyDeviceToHost) );

	out = (T*)malloc(size * sizeof(T));
	gpuErrCheck( cudaMalloc(&temp, size*sizeof(T)) );
	vec2array<<<gridSize(size, BSIZE), BSIZE>>>(temp, v);
	gpuErrCheck( cudaMemcpy(out, temp, size*sizeof(int), cudaMemcpyDeviceToHost) );
	return size;
}

// LFVector test
__global__ void printVec(Vector<int> *v) {
	printf("size: %d\n", v->size);
	//return;
	printf("ranges: ");
	for (int i = 0; i < NB; ++i) {
		printf("%d ", v->ranges[i]);
	}
	printf("\nsizes: ");
	for (int i = 0; i < NB; ++i) {
		printf("%d ", v->lfv[i].size);
	}
	printf("\n");
	return;
	for (int i = 0; i < v->size; ++i) {
		printf("%d ", v->at(i));
	}
	printf("\n");
	printf("last element: %d\n", v->at(v->size - 1));
}

__global__ void initVec(Vector<int> *v) {
	*v = Vector<int>();
	v->size = NB*BSIZE;
	for (int i = 0; i < NB; ++i) {
		v->ranges[i] = BSIZE*i;
		v->lfv[i].resize(BSIZE);
	}
	for(int i = 0; i < v->size; ++i) {
		v->at(i) = i;
	}
}

__global__ void test_insert(Vector<int> *v) {
	int tid = threadIdx.x;
	v->insert(tid, 1);
}

__global__ void test_insert2(Vector<int> *v) {
	int tid = threadIdx.x;
	int bs = v->lfv[blockIdx.x].size;
	//printf("%d %d %d\n", tid, blockIdx.x, bs);
	for (int i = tid; i < bs; i += BSIZE) {
		v->insert(i, 1);
	}
}

__global__ void test_insert2_2(Vector<int> *v) {
	int tid = threadIdx.x;
	int bs = v->lfv[blockIdx.x].size;
	//printf("%d %d %d\n", tid, blockIdx.x, bs);
	for (int i = tid; i < bs; i += BSIZE) {
		v->insert(v->lfv[blockIdx.x].at(i)+1024, 1);
	}
}

__global__ void test_insert3(Vector<int> *v) {
	int tid = threadIdx.x;
	for (int i = 0; i < 10; ++i) {
		v->insert(tid, 1);
	}
}

__global__ void test_read_write_g(Vector<int> *v, int size) {
	int rep = 30;
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= size) return;
	for (int i = 0; i < rep; ++i) {
		v->at(tid) += 1;
	}
}

__global__ void test_read_write_b(Vector<int> *v) {
	int rep = 30;
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	for (int i = tid; i < v->lfv[bid].size; i += BSIZE) {
		for (int j = 0; j < rep; ++j) {
			v->lfv[bid].at(i) += 1;
		}
	}
}

__global__ void random_copy(LFVector<int> *v) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int n = v->size;
	for (int i = tid; i < n; i+=BSIZE) {
		if (v->at(i) < PROB)
			v->push_back(v->at(i));
	}
}

__global__ void init_random(LFVector<int> *v, int *A, int n) {
	v->size = 0;
	v->a = (int**)malloc(sizeof(int*)*100);
	v->a[0] = (int*)malloc(sizeof(int)*32);
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

void test_random_copy(LFVector<int> *v, int n) {
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

void run_experiment(Vector<int> *v, int size, int ratio) {
	int *ds;
	gpuErrCheck( cudaMalloc(&ds, sizeof(int)) );

	int rep = 10;
	int rw_rep = 30;
	int o_size = size;
	createLFVector<<<1,1>>>(v); kernelCallCheck();
	initVec<<<NB,BSIZE>>>(v, size); kernelCallCheck();
	//printVec<<<1,1>>>(v); kernelCallCheck();
	float results[rep];
	float results_rw[rw_rep];
	float s = 0.0;
	// insert
	for (int i = 0; i < rep; ++i) {
		printf("%d ", i); fflush(stdout);
		cudaEvent_t start, stop;
		start_clock(start, stop);
		test_insert2<<<NB, BSIZE>>>(v); kernelCallCheck();
		cudaDeviceSynchronize();
		results[i] = stop_clock(start, stop);
		s += results[i];

		// read/write
		results_rw[i] = 0.0;
		for (int j = 0; j < rw_rep; ++j) {
			cudaEvent_t start, stop;
			start_clock(start, stop);
			// wr block
				test_read_write_b<<<NB, BSIZE>>>(v);
			// wr global - slow
				//get_size<<<1,1>>>(ds, v);
				//uErrCheck( cudaMemcpy(&size, ds, sizeof(int), cudaMemcpyDeviceToHost) );
				//st_read_write_g<<<gridSize(size, BSIZE), BSIZE>>>(v, size);
			cudaDeviceSynchronize();
			results_rw[i] += stop_clock(start, stop);
		}
		results_rw[i] /= rw_rep;
	}
	
	// print results
	printf("\n");
	printf("mlfv%d,in,%d,%d,", NB, o_size, ratio);
	for (int i = 0; i < rep-1; ++i) {
		printf("%f,", results[i]);
	}
	printf("%f\n", results[rep-1]);
	//printf("%f\n", s);
	printf("mlfv%d,rw,%d,%d,", NB, o_size, ratio);
	for (int i = 0; i < rep-1; ++i) {
		printf("%f,", results_rw[i]);
	}
	printf("%f\n", results_rw[rep-1]);
}



int main(int argc, char **argv){

	//cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1000*NB*BSIZE*sizeof(int));
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, INT_MAX*sizeof(int));

	int *a, *ha;
	int size = 1<<19;
	ha = new int[size];
	for (int i = 0; i < size; ++i) {
		ha[i] = i;
	}
	gpuErrCheck( cudaMalloc(&a, size*sizeof(int)) );
	gpuErrCheck( cudaMemcpy(a, ha, size*sizeof(int), cudaMemcpyHostToDevice)) ;
	
	// LFV
	Vector<int> *lfv;
	gpuErrCheck( cudaMalloc(&lfv, sizeof(Vector<int>)) );

	//createLFVector<<<1,1>>>(lfv); kernelCallCheck();
	//initVec<<<NB,BSIZE>>>(lfv, size, a); kernelCallCheck();
	//printVec<<<1,1>>>(lfv); kernelCallCheck();

	// TODO use ratio (3rd arg) in insertion
	run_experiment(lfv, size, 2);

	return 0;

	//initVec<<<1,1>>>(lfv); kernelCallCheck();
	createLFVector<<<1,1>>>(lfv); kernelCallCheck();
	initVec<<<NB,BSIZE>>>(lfv, size, a); kernelCallCheck();
	//printVec<<<1,1>>>(lfv); kernelCallCheck();
	int *r;
	int final_size = sendToHost(r, lfv);
	printf("%d\n", final_size);
	//print_array(r, final_size);
	for (int i = 0; i < 5; ++i) {
		printf("%d\n", i);
		test_insert2_2<<<NB,BSIZE>>>(lfv); kernelCallCheck();
		printVec<<<1,1>>>(lfv); kernelCallCheck();
	}
	//printVec<<<1,1>>>(lfv); kernelCallCheck();
	
	return 0;
}
