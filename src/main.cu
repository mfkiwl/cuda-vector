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
#define NUM_BLOCKS 514
#endif


void run_mlfv_NB() {
	int size = 1 << 25;
	int *a, *ha;
	ha = new int[size];
	for (int i = 0; i < size; ++i) {
		ha[i] = i;
	}
	gpuErrCheck( cudaMalloc(&a, size*sizeof(int)) );
	gpuErrCheck( cudaMemcpy(a, ha, size*sizeof(int), cudaMemcpyHostToDevice)) ;


	Vector<int, NUM_BLOCKS> *lfv;
	gpuErrCheck( cudaMalloc(&lfv, sizeof(Vector<int, NUM_BLOCKS>)) );
	optimal_NB<NUM_BLOCKS>(lfv, size, 1);
}

void run_mlfv64(int size) {
	int *a, *ha;
	ha = new int[size];
	for (int i = 0; i < size; ++i) {
		ha[i] = i;
	}
	gpuErrCheck( cudaMalloc(&a, size*sizeof(int)) );
	gpuErrCheck( cudaMemcpy(a, ha, size*sizeof(int), cudaMemcpyHostToDevice)) ;
	
	// LFV
	const int NB = 64;
	Vector<int, NB> *lfv;
	gpuErrCheck( cudaMalloc(&lfv, sizeof(Vector<int, NB>)) );

	run_experiment<NB>(lfv, size, 1);
}

void run_mlfv1024(int size) {
	int *a, *ha;
	ha = new int[size];
	for (int i = 0; i < size; ++i) {
		ha[i] = i;
	}
	gpuErrCheck( cudaMalloc(&a, size*sizeof(int)) );
	gpuErrCheck( cudaMemcpy(a, ha, size*sizeof(int), cudaMemcpyHostToDevice)) ;
	
	// LFV
	const int NB = 1024;
	Vector<int, NB> *lfv;
	gpuErrCheck( cudaMalloc(&lfv, sizeof(Vector<int, NB>)) );

	run_experiment<NB>(lfv, size, 1);
}

void run_memMap(int size) {

	int ratio = 1;
	cudaSetDevice(0);
	CUcontext ctx;
	cuDevicePrimaryCtxRetain(&ctx, 0);
	cuCtxSetCurrent(ctx);
	
	run_experiment(ctx, size, ratio);

	cuDevicePrimaryCtxRelease(0);
}

void run_static(int size) {
	int ratio = 1;
	run_experiment(size, ratio);
}

int main(int argc, char **argv){
	if (argc < 2) {
		fprintf(stderr,"error, run as ./prog struct\n");
		fprintf(stderr,"\tstructs: static(0) memMap(1) mlfv64(2) mlfv1024(3)\n");
		return -1;
	}
	
	int structure = atoi(argv[1]);
	int size = 1<<19;

	cudaDeviceSetLimit(cudaLimitMallocHeapSize, INT_MAX*sizeof(int));

	if (structure == 4) {
		run_mlfv_NB();
		return 0;
	}

	switch (structure) {
		case 0: run_static(size); break;
		case 1: run_memMap(size); break;
		case 2: run_mlfv1024(size); break;
		case 3: run_mlfv64(size); break;
	}

	kernelCallCheck();
	return 0;
}
