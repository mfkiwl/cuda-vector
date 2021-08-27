#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include "utility.cuh"

#define ull unsigned long long int
#define BSIZE 1024
#define PROB 90

struct Vec {
	CUdeviceptr d_p;
	CUmemGenericAllocationHandle allocHandle;
	CUmemAllocationProp prop;
	CUmemAccessDesc accessDesc;
	size_t chunk_sz;
	size_t size;
	size_t reserved;
	void init(int n);
	__device__ int &at(int i);
};

void Vec::init(int n) {
	CUresult res;
	int dev = 0;
	std::cout << res << std::endl;
	size = n;
	prop = {};
	prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
	prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	prop.location.id = dev;
	res = cuMemGetAllocationGranularity(&chunk_sz, &prop,
				CU_MEM_ALLOC_GRANULARITY_MINIMUM);
	std::cout << res << std::endl;

	reserved = ((n + chunk_sz - 1) / chunk_sz) * chunk_sz;
	res = cuMemCreate(&allocHandle, reserved, &prop, 0);
	std::cout << res << std::endl;
	res = cuMemAddressReserve(&d_p, reserved, 0, 0, 0);
	std::cout << res << std::endl;
	res = cuMemMap(d_p, reserved, 0, allocHandle, 0);
	std::cout << res << std::endl;

	accessDesc = {};
	accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	accessDesc.location.id = dev;
	accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
	cuMemSetAccess(d_p, reserved, &accessDesc, 1);
}

__device__ int &Vec::at(int i) {
	return (*(int**)&d_p)[i];
}

__global__ void test(Vec *v) {
	for (int i = 0; i < v->size; ++i) {
		v->at(i) = i;
	}
}

__global__ void printVec(Vec *v) {
	for (int i = 0; i < v->size; ++i) {
		printf("%d ", v->at(i));
	}
	printf("\n");
}

int main(int argc, char **argv){
	int n = 100;
	Vec *a = new Vec();
	a->init(n);
	//cudaMalloc(&a, sizeof(Vec));

	//test<<<1,1>>>(a);
	//printVec<<<1,1>>>(a);
	
	return 0;
}

