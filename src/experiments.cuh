#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "../common/utility.cuh"
#include "tests.cuh"

#define BSIZE 1024
#define REP 10
#define RW_REP 30


// test NB
template<int NB>
void optimal_NB(Vector<int, NB> *v, int size) {
	int *ds;
	gpuErrCheck( cudaMalloc(&ds, sizeof(int)) );
	//fprintf(stderr, "NB  %d \n", NB); 

	createLFVector<<<1,1>>>(v); kernelCallCheck();
	initVec<<<NB,BSIZE>>>(v, size); kernelCallCheck();

	cudaEvent_t start, stop;
	start_clock(start, stop);

	growVec<<<1,NB>>>(v, 2*size);
	test_insert2<<<NB, BSIZE>>>(v, 1, 1); //kernelCallCheck();
	cudaDeviceSynchronize();

	float time = stop_clock(start, stop);
	printf("%d,%d,%f,", size, NB, time);


	//printVec<<<1,1>>>(v);
	get_size<<<1,1>>>(ds, v);
	cudaMemcpy(&size, ds, sizeof(int), cudaMemcpyDeviceToHost);
	//fprintf(stderr, "size  %d \n", size); 
	start_clock(start, stop);
	test_read_write_g<<<gridSize(size, BSIZE), BSIZE>>>(v, size, RW_REP); //kernelCallCheck();
	cudaDeviceSynchronize();
	time = stop_clock(start, stop);
	printf("%f,", time);

	start_clock(start, stop);
	test_read_write_b<<<NB, BSIZE>>>(v, RW_REP); //kernelCallCheck();
	cudaDeviceSynchronize();
	time = stop_clock(start, stop);
	printf("%f\n", time);
}

template<int NB>
void single_run_experiment(Vector<int, NB> *v, int size) {
	int *ds;
	gpuErrCheck( cudaMalloc(&ds, sizeof(int)) );
	//fprintf(stderr, "NB  %d \n", NB); 

	createLFVector<<<1,1>>>(v); kernelCallCheck();
	initVec<<<NB,BSIZE>>>(v, size); kernelCallCheck();

	growVec<<<1,NB>>>(v, 2*size);

	cudaEvent_t start, stop;
	start_clock(start, stop);
	test_insert2<<<NB, BSIZE>>>(v, 1, 1); //kernelCallCheck();
	cudaDeviceSynchronize();
	float time = stop_clock(start, stop);
	printf("%d,%d,%f,", size, NB, time);


	//printVec<<<1,1>>>(v);
	get_size<<<1,1>>>(ds, v);
	cudaMemcpy(&size, ds, sizeof(int), cudaMemcpyDeviceToHost);
	//fprintf(stderr, "size  %d \n", size); 

	start_clock(start, stop);
	test_read_write_g<<<gridSize(size, BSIZE), BSIZE>>>(v, size, RW_REP); //kernelCallCheck();
	cudaDeviceSynchronize();
	time = stop_clock(start, stop);
	printf("%f,", time);

	start_clock(start, stop);
	test_read_write_b<<<NB, BSIZE>>>(v, RW_REP); //kernelCallCheck();
	cudaDeviceSynchronize();
	time = stop_clock(start, stop);
	printf("%f\n", time);
}

template<int NB>
void run_experiment(Vector<int, NB> *v, int size, int r1, int r2) {
	int *ds;
	gpuErrCheck( cudaMalloc(&ds, sizeof(int)) );

		fprintf(stderr, "start experiment\n");
	int rep = 10;
	int rw_rep = 10;
	//size = 1000;
	int o_size = size;
	createLFVector<<<1,1>>>(v); kernelCallCheck();
		//fprintf(stderr, "creat lfv \n");
	initVec<<<NB,BSIZE>>>(v, size); kernelCallCheck();
		//fprintf(stderr, "init lfv \n");
	//printVec<<<1,1>>>(v); kernelCallCheck();
	float results[rep];
	float results_grow[rep];
	float results_rw[rw_rep];
	int rw_kernel_rep = RW_REP;

	for (int i = 0; i < rep; ++i) {
		// grow
		//fprintf(stderr, "grow %i\n", i);
		cudaEvent_t start, stop;
		start_clock(start, stop);
		growVec<<<1,NB>>>(v, 2*size);
		cudaDeviceSynchronize();
		results_grow[i] = stop_clock(start, stop);
		//printVec<<<1,1>>>(v); kernelCallCheck();

		// insertion
		//fprintf(stderr, "insert %i\n", i);
		start_clock(start, stop);
		test_insert2<<<NB, BSIZE>>>(v ,r1, r2); //kernelCallCheck();
		cudaDeviceSynchronize();
		results[i] = stop_clock(start, stop);

		// read/write
		//fprintf(stderr, "rw %i\n", i);
		results_rw[i] = 0.0;
		for (int j = 0; j < rw_rep; ++j) {
			cudaEvent_t start, stop;
				get_size<<<1,1>>>(ds, v);
				cudaMemcpy(&size, ds, sizeof(int), cudaMemcpyDeviceToHost);
			start_clock(start, stop);
			// wr block
				test_read_write_b<<<NB, BSIZE>>>(v, rw_kernel_rep);
			// wr global - slow
				//test_read_write_g<<<gridSize(size, BSIZE), BSIZE>>>(v, size, rw_kernel_rep);
			cudaDeviceSynchronize();
			results_rw[i] += stop_clock(start, stop);
		}
		results_rw[i] /= rw_rep;
		size *= 2;
	}
	//printVec<<<1,1>>>(v); kernelCallCheck();
	
	// print results
	// grow
	printf("mlfv%d,grow,%d,%d/%d,", NB, o_size, r1, r2);
	for (int i = 0; i < rep-1; ++i) {
		printf("%f,", results_grow[i]);
	}
	printf("%f\n", results_grow[rep-1]);
	// insert
	printf("mlfv%d,in,%d,%d/%d,", NB, o_size, r1, r2);
	for (int i = 0; i < rep-1; ++i) {
		printf("%f,", results[i]);
	}
	printf("%f\n", results[rep-1]);
	//printf("%f\n", s);
	// read-write
	printf("mlfv%d,rw%d,%d,%d/%d,", NB, rw_kernel_rep, o_size, r1, r2);
	for (int i = 0; i < rep-1; ++i) {
		printf("%f,", results_rw[i]);
	}
	printf("%f\n", results_rw[rep-1]);
}


void run_experiment(CUcontext ctx, int size, int r1, int r2) {
	int rep = 10;
	int rw_rep = 30;
	int o_size = size;
	int rw_kernel_rep = RW_REP;
	int *ds;
	cudaMalloc(&ds, sizeof(int));
	cudaMemcpy(ds, &size, sizeof(int), cudaMemcpyHostToDevice);

	VectorMemMap a = VectorMemMap(ctx);

	a.grow(size*sizeof(int));
	initVec<<<gridSize(size, 1024), 1024>>>(a.getPointer(), size); kernelCallCheck();

	float results[rep];
	float results_grow[rep];
	float results_rw[rw_rep];
	
	for (int i = 0; i < rep; ++i) {

		// grow
		cudaEvent_t start, stop;
		start_clock(start, stop);
		a.grow(size*2*sizeof(int));
		cudaDeviceSynchronize();
		results_grow[i] = stop_clock(start, stop);
		
		// insertion
		start_clock(start, stop);
		test_insert_atomic<<<gridSize(size, 1024), 1024>>>(a.getPointer(), size, ds, r1, r2);
		cudaDeviceSynchronize();
		results[i] = stop_clock(start, stop);
		cudaMemcpy(&size, ds, sizeof(int), cudaMemcpyDeviceToHost);

		// read/write
		results_rw[i] = 0.0;
		CUdeviceptr dp = a.getPointer();
		for (int j = 0; j < rw_rep; ++j) {
			cudaEvent_t start, stop;
			start_clock(start, stop);
			test_read_write<<<gridSize(size, 1024), 1024>>>(dp, size, rw_kernel_rep);
			cudaDeviceSynchronize();
			results_rw[i] += stop_clock(start, stop);
		}
		results_rw[i] /= rw_rep;
	}

	// print results
	printf("memMap,grow,%d,%d/%d,", o_size, r1, r2);
	for (int i = 0; i < rep-1; ++i) {
		printf("%f,", results_grow[i]);
	}
	printf("%f\n", results_grow[rep-1]);
	printf("memMap,in,%d,%d/%d,", o_size, r1, r2);
	for (int i = 0; i < rep-1; ++i) {
		printf("%f,", results[i]);
	}
	printf("%f\n", results[rep-1]);
	//printf("%f\n", s);
	printf("memMap,rw%d,%d,%d/%d,", rw_kernel_rep, o_size, r1, r2);
	for (int i = 0; i < rep-1; ++i) {
		printf("%f,", results_rw[i]);
	}
	printf("%f\n", results_rw[rep-1]);
}

void run_experiment(int size, int r1, int r2) {
	int rep = 10;
	int rw_rep = 30;
	int o_size = size;
	int rw_kernel_rep = RW_REP;

	int *a, *ha;
	int *dsize;
	ha = new int[size];
	for (int i = 0; i < size; ++i) {
		ha[i] = i;
	}
	//gpuErrCheck( cudaMalloc(&a, 2*size*2^rep*sizeof(int)) );
	gpuErrCheck( cudaMalloc(&a, (1<<29)*sizeof(int)) );
	gpuErrCheck( cudaMalloc(&dsize, sizeof(int)) );
	gpuErrCheck( cudaMemcpy(a, ha, size*sizeof(int), cudaMemcpyHostToDevice)) ;
	gpuErrCheck( cudaMemcpy(dsize, &size, sizeof(int), cudaMemcpyHostToDevice) );


	float results[rep];
	float results_rw[rw_rep];

	for (int i = 0; i < rep; ++i) {
		cudaEvent_t start, stop;
		start_clock(start, stop);
		test_insert_atomic<<<gridSize(size, BSIZE), BSIZE>>>(a, size, dsize, r1, r2); //kernelCallCheck();
		cudaDeviceSynchronize();
		results[i] = stop_clock(start, stop);
		cudaMemcpy(&size, dsize, sizeof(int), cudaMemcpyDeviceToHost);
		//fprintf(stderr, "size %d\n", size);

		// read/write
		results_rw[i] = 0.0;
		for (int j = 0; j < rw_rep; ++j) {
			cudaEvent_t start, stop;
			start_clock(start, stop);
			test_read_write<<<gridSize(size, 1024), 1024>>>(a, size, rw_kernel_rep);
			//kernelCallCheck();
			cudaDeviceSynchronize();
			results_rw[i] += stop_clock(start, stop);
		}
		results_rw[i] /= rw_rep;
	}

	// print results
	printf("static,in,%d,%d/%d,", o_size, r1, r2);
	for (int i = 0; i < rep-1; ++i) {
		printf("%f,", results[i]);
	}
	printf("%f\n", results[rep-1]);
	//printf("%f\n", s);
	printf("static,rw%d,%d,%d/%d,", rw_kernel_rep, o_size, r1, r2);
	for (int i = 0; i < rep-1; ++i) {
		printf("%f,", results_rw[i]);
	}
	printf("%f\n", results_rw[rep-1]);
}
