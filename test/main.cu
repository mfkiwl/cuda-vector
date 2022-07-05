#include <iostream>
#include <cuda.h>
#include <thrust/device_vector.h>
#include "../common/utility.cuh"
#include "../stdgpu/src/stdgpu/vector.cuh"

void run_experiment_thrist() {
	int size = 1 << 19;
	int rep = 10;
	int o_size = size;
	int ratio = 1;
	thrust::device_vector<int> d_vec(size);

	float results[rep];

	for (int i = 0; i < rep; ++i) {
		cudaEvent_t start, stop;
		start_clock(start, stop);
		d_vec.resize(2*size);
		cudaDeviceSynchronize();
		results[i] = stop_clock(start, stop);


		size *= 2;
	}

	printf("thrust,gr,%d,%d,", o_size, ratio);
	for (int i = 0; i < rep-1; ++i) {
		printf("%f,", results[i]);
	}
	printf("%f\n", results[rep-1]);
}

void test() {
	stdgpu::vector vec = sdtgpu::vector();
}

int main(int argc, char* argv[]) {
	
	//run_experiment_thrust();
	test();

	return 0;
}
