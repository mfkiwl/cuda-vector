#pragma once

#define gpuErrCheck(ans) { gpuAssert((ans), __FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUASSERT: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#define kernelCallCheck() \
	{ gpuErrCheck( cudaPeekAtLastError() ); \
	gpuErrCheck( cudaDeviceSynchronize() ); } 

void inline start_clock(cudaEvent_t &start, cudaEvent_t &stop){
	gpuErrCheck( cudaEventCreate(&start) );
	gpuErrCheck( cudaEventCreate(&stop) );
	gpuErrCheck( cudaEventRecord(start, 0) );
}

float inline stop_clock(cudaEvent_t &start, cudaEvent_t &stop){
	float time;
	gpuErrCheck( cudaEventRecord(stop, 0) );
	gpuErrCheck( cudaEventSynchronize(stop) );
	gpuErrCheck( cudaEventElapsedTime(&time, start, stop) );
	gpuErrCheck( cudaEventDestroy(start) );
	gpuErrCheck( cudaEventDestroy(stop) );
	return time;
}

void print_array(float *m, int n, const char *msg){
	printf("%s\n", msg);
	for (int i = 0; i < n; ++i){
		printf("%.2f ", m[i]);
	}
	printf("\n");
}

void print_array(int *m, int n) {
	for (int i = 0; i < n; ++i){
		printf("%d ", m[i]);
	}
	printf("\n");
}
void print_array(int *m, int n, const char *msg) {
	printf("%s\n", msg);
	for (int i = 0; i < n; ++i){
		printf("%d ", m[i]);
	}
	printf("\n");
}

int inline gridSize(int n, int bsize) {
	return (n + bsize - 1) / bsize;
}
