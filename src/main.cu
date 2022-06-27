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
#define NUM_BLOCKS  1024
#endif


void run_mlfv_NB() {
	int size = 1 << 20;
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

void run_mlfv64(int op, int size) {
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

	switch (op) {
		case 0: growth_experiment<NB>(lfv, size, 1); break;
		case 1: insertion_experiment<NB>(lfv, size, 1); break;
		case 2: rw_experiment<NB>(lfv, size, 1, 0); break;
		case 3: rw_experiment<NB>(lfv, size, 1, 1); break;
	}
}

void run_mlfv1024(int op, int size) {
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

	switch (op) {
		case 0: growth_experiment<NB>(lfv, size, 1); break;
		case 1: insertion_experiment<NB>(lfv, size, 1); break;
		case 2: rw_experiment<NB>(lfv, size, 1, 0); break;
		case 3: rw_experiment<NB>(lfv, size, 1, 1); break;
	}
}

void run_memMap(int op, int size) {

	int ratio = 1;
	cudaSetDevice(0);
	CUcontext ctx;
	cuDevicePrimaryCtxRetain(&ctx, 0);
	cuCtxSetCurrent(ctx);
	
	switch (op) {
		case 0: growth_experiment(ctx, size, ratio); break;
		case 1: insertion_experiment(ctx, size, ratio); break;
		case 2: rw_experiment(ctx, size, ratio); break;
	}
	cuDevicePrimaryCtxRelease(0);
}

void run_static(int op, int size) {
	int ratio = 1;
	switch (op) {
		case 1: insertion_experiment(size, ratio); break;
		case 2: rw_experiment(size, ratio); break;
	}
}

int main(int argc, char **argv){
	if (argc < 3) {
		fprintf(stderr,"error, run as ./prog struct op\n");
		fprintf(stderr,"\tstructs: static(0) memMap(1) mlfv64(2) mlfv1024(3)\n");
		fprintf(stderr,"\tops: grow(0) insert(1) rw_g(2) rw_b(3)\n");
		return -1;
	}
	
	int structure = atoi(argv[1]);
	int op = atoi(argv[2]);
	int size = 1<<19;

	cudaDeviceSetLimit(cudaLimitMallocHeapSize, INT_MAX*sizeof(int));

	if (op == 4) {
		run_mlfv_NB();
		return 0;
	}

	switch (structure) {
		case 0: run_static(op, size); break;
		case 1: run_memMap(op, size); break;
		case 2: run_mlfv64(op, size); break;
		case 3: run_mlfv1024(op, size); break;
	}

	kernelCallCheck();
	return 0;
}
