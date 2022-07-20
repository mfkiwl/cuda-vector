#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <vector>
#include "../common/utility.cuh"
#include "experiments.cuh"

#define BSIZE 1024
#ifndef NUM_BLOCKS
#define NUM_BLOCKS 1024
#endif


void run_mlfv_NB() {
	//int size = 1 << 25;
	int size = 1e7;
	int *a, *ha;
	ha = new int[size];
	for (int i = 0; i < size; ++i) {
		ha[i] = i;
	}
	gpuErrCheck( cudaMalloc(&a, size*sizeof(int)) );
	gpuErrCheck( cudaMemcpy(a, ha, size*sizeof(int), cudaMemcpyHostToDevice)) ;


	Vector<int, NUM_BLOCKS> *lfv;
	gpuErrCheck( cudaMalloc(&lfv, sizeof(Vector<int, NUM_BLOCKS>)) );
	optimal_NB<NUM_BLOCKS>(lfv, size);
}

void run_mlfv64(int size, int r1, int r2) {
	int *a, *ha;
	ha = new int[size];
	for (int i = 0; i < size; ++i) {
		ha[i] = i;
	}
	gpuErrCheck( cudaMalloc(&a, size*sizeof(int)) );
	gpuErrCheck( cudaMemcpy(a, ha, size*sizeof(int), cudaMemcpyHostToDevice)) ;
	
	// LFV
	const int NB = 32;
	Vector<int, NB> *lfv;
	gpuErrCheck( cudaMalloc(&lfv, sizeof(Vector<int, NB>)) );

	run_experiment<NB>(lfv, size, r1, r2);
}

void run_mlfv1024(int size, int r1, int r2) {
	int *a, *ha;
	ha = new int[size];
	for (int i = 0; i < size; ++i) {
		ha[i] = i;
	}
	gpuErrCheck( cudaMalloc(&a, size*sizeof(int)) );
	gpuErrCheck( cudaMemcpy(a, ha, size*sizeof(int), cudaMemcpyHostToDevice)) ;
	
	// LFV
	const int NB = 512;
	Vector<int, NB> *lfv;
	gpuErrCheck( cudaMalloc(&lfv, sizeof(Vector<int, NB>)) );

	
	run_experiment<NB>(lfv, size, r1, r2);
}

void run_memMap(int size, int r1, int r2) {

	cudaSetDevice(0);
	CUcontext ctx;
	cuDevicePrimaryCtxRetain(&ctx, 0);
	cuCtxSetCurrent(ctx);
	
	run_experiment(ctx, size, r1, r2);

	cuDevicePrimaryCtxRelease(0);
}

void run_static(int size, int r1, int r2) {
	run_experiment(size, r1, r2);
}

void run_size_test1024(int size) {
	int *a, *ha;
	ha = new int[size];
	for (int i = 0; i < size; ++i) {
		ha[i] = i;
	}
	gpuErrCheck( cudaMalloc(&a, size*sizeof(int)) );
	gpuErrCheck( cudaMemcpy(a, ha, size*sizeof(int), cudaMemcpyHostToDevice)) ;
	
	// LFV
	const int NB = 512;
	Vector<int, NB> *lfv;
	gpuErrCheck( cudaMalloc(&lfv, sizeof(Vector<int, NB>)) );

	single_run_experiment(lfv, size);
}

void run_phases_app_mlfv(int size, int k, int q) {
	cudaSetDevice(0);
	CUcontext ctx;
	cuDevicePrimaryCtxRetain(&ctx, 0);
	cuCtxSetCurrent(ctx);

	int *a, *ha;
	ha = new int[size];
	for (int i = 0; i < size; ++i) {
		ha[i] = i;
	}
	gpuErrCheck( cudaMalloc(&a, size*sizeof(int)) );
	gpuErrCheck( cudaMemcpy(a, ha, size*sizeof(int), cudaMemcpyHostToDevice)) ;
	
	// LFV
	const int NB = 512;
	Vector<int, NB> *lfv;
	gpuErrCheck( cudaMalloc(&lfv, sizeof(Vector<int, NB>)) );

	phases_app_experiment(lfv, ctx, size, k, q);

	cuDevicePrimaryCtxRelease(0);
}

void run_phases_app_memMap(int size, int k, int q) {
	cudaSetDevice(0);
	CUcontext ctx;
	cuDevicePrimaryCtxRetain(&ctx, 0);
	cuCtxSetCurrent(ctx);

	int *a, *ha;
	ha = new int[size];
	for (int i = 0; i < size; ++i) {
		ha[i] = i;
	}
	gpuErrCheck( cudaMalloc(&a, size*sizeof(int)) );
	gpuErrCheck( cudaMemcpy(a, ha, size*sizeof(int), cudaMemcpyHostToDevice)) ;

	phases_app_experiment(ctx, size, k, q);

	cuDevicePrimaryCtxRelease(0);
}

int main(int argc, char **argv){
	if (argc < 2) {
		fprintf(stderr,"error, run as ./prog struct\n");
		fprintf(stderr,"\tstructs: static(0) memMap(1) mlfv64(2) mlfv1024(3)\n");
		return -1;
	}

	cudaSetDevice(0);
	
	int structure = atoi(argv[1]);
	int r1 = 1;
	int r2 = 1;
	//int size = 1<<19;
	int size = 1e6;

	cudaDeviceSetLimit(cudaLimitMallocHeapSize, INT_MAX*sizeof(int));

	if (structure == 10) {
		run_mlfv_NB();
		return 0;
	}
	if (structure == 11) {
		if (argc < 3) {
			fprintf(stderr,"error, run as ./prog 11 n\n");
			return -1;
		}
		int n = atoi(argv[2]);
		run_size_test1024(n);
		return 0;
	}
	if (structure > 11) {
		if (argc < 4) {
			fprintf(stderr,"error, run as ./prog 12/13 k q\n");
			return -1;
		}
		int k = atoi(argv[2]);
		int q = atoi(argv[3]);
		int nf = 1e9;
		switch (structure) {
			case 12: run_phases_app_mlfv(nf, k, q); break;
			case 13: run_phases_app_memMap(nf, k, q); break;
		}
		return 0;
	}

	switch (structure) {
		case 0: run_static(size, r1, r2); break;
		case 1: run_memMap(size, r1, r2); break;
		case 2: run_mlfv1024(size, r1, r2); break;
		case 3: run_mlfv64(size, r1, r2); break;
	}

	kernelCallCheck();
	return 0;
}
