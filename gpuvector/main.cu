#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include "utility.cuh"

#define ull unsigned long long int
#define BSIZE 1024
#define PROB 90


class VectorMemMap {
	private:
		CUdeviceptr d_p;
		CUmemAllocationProp prop;
		CUmemAccessDesc accessDesc;
		struct Range {
			CUdeviceptr start;
			size_t sz;
		};
		std::vector<Range> va_ranges;
		std::vector<CUmemGenericAllocationHandle> handles;
		std::vector<size_t> handle_sizes;
		size_t alloc_sz;
		size_t reserve_sz;
		size_t chunk_sz;
	public:
		int *d_size;
		VectorMemMap(CUcontext context);
		~VectorMemMap();

		CUdeviceptr getPointer() const {
			return d_p;
		}

		size_t getSize() const {
			return alloc_sz;
		}

		// Reserves some extra space in order to speed up grow()
		CUresult reserve(size_t new_sz);

		// Actually commits num bytes of additional memory
		CUresult grow(size_t new_sz);

		__device__ int& at(unsigned int i);
		__device__ void push_back(int e);
};

VectorMemMap::VectorMemMap(CUcontext context) : d_p(0ULL), prop(), handles(), alloc_sz(0ULL), reserve_sz(0ULL), chunk_sz(0ULL)
{
	CUdevice device;
	CUcontext prev_ctx;
	CUresult status = CUDA_SUCCESS;
	(void)status;

	status = cuCtxGetCurrent(&prev_ctx);
	assert(status == CUDA_SUCCESS);
	if (cuCtxSetCurrent(context) == CUDA_SUCCESS) {
		status = cuCtxGetDevice(&device);
		assert(status == CUDA_SUCCESS);
		status = cuCtxSetCurrent(prev_ctx);
		assert(status == CUDA_SUCCESS);
	}

	prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
	prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	prop.location.id = (int)device;
	prop.win32HandleMetaData = NULL;

	accessDesc.location = prop.location;
	accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

	status = cuMemGetAllocationGranularity(&chunk_sz, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
	assert(status == CUDA_SUCCESS);

	cudaMalloc(&d_size, sizeof(size_t));
	int t = alloc_sz;
	cudaMemcpy(&d_size, &t, sizeof(size_t), cudaMemcpyHostToDevice);
}

VectorMemMap::~VectorMemMap()
{
	CUresult status = CUDA_SUCCESS;
	(void)status;
	if (d_p != 0ULL) {
		status = cuMemUnmap(d_p, alloc_sz);
		assert(status == CUDA_SUCCESS);
		for (size_t i = 0ULL; i < va_ranges.size(); i++) {
			status = cuMemAddressFree(va_ranges[i].start, va_ranges[i].sz);
			assert(status == CUDA_SUCCESS);
		}
		for (size_t i = 0ULL; i < handles.size(); i++) {
			status = cuMemRelease(handles[i]);
			assert(status == CUDA_SUCCESS);
		}
	}
}

CUresult VectorMemMap::reserve(size_t new_sz)
{
	CUresult status = CUDA_SUCCESS;
	CUdeviceptr new_ptr = 0ULL;

	if (new_sz <= reserve_sz) {
		return CUDA_SUCCESS;
	}

	const size_t aligned_sz = ((new_sz + chunk_sz - 1) / chunk_sz) * chunk_sz;

	status = cuMemAddressReserve(&new_ptr, (aligned_sz - reserve_sz), 0ULL, d_p + reserve_sz, 0ULL);

	// Try to reserve an address just after what we already have reserved
	if (status != CUDA_SUCCESS || (new_ptr != d_p + reserve_sz)) {
		if (new_ptr != 0ULL) {
			(void)cuMemAddressFree(new_ptr, (aligned_sz - reserve_sz));
		}
		// Slow path - try to find a new address reservation big enough for us
		status = cuMemAddressReserve(&new_ptr, aligned_sz, 0ULL, 0U, 0);
		if (status == CUDA_SUCCESS && d_p != 0ULL) {
			CUdeviceptr ptr = new_ptr;
			// Found one, now unmap our previous allocations
			status = cuMemUnmap(d_p, alloc_sz);
			assert(status == CUDA_SUCCESS);
			for (size_t i = 0ULL; i < handles.size(); i++) {
				const size_t hdl_sz = handle_sizes[i];
				// And remap them, enabling their access
				if ((status = cuMemMap(ptr, hdl_sz, 0ULL, handles[i], 0ULL)) != CUDA_SUCCESS)
					break;
				if ((status = cuMemSetAccess(ptr, hdl_sz, &accessDesc, 1ULL)) != CUDA_SUCCESS)
					break;
				ptr += hdl_sz;
			}
			if (status != CUDA_SUCCESS) {
				// Failed the mapping somehow... clean up!
				status = cuMemUnmap(new_ptr, aligned_sz);
				assert(status == CUDA_SUCCESS);
				status = cuMemAddressFree(new_ptr, aligned_sz);
				assert(status == CUDA_SUCCESS);
			}
			else {
				// Clean up our old VA reservations!
				for (size_t i = 0ULL; i < va_ranges.size(); i++) {
					(void)cuMemAddressFree(va_ranges[i].start, va_ranges[i].sz);
				}
				va_ranges.clear();
			}
		}
		// Assuming everything went well, update everything
		if (status == CUDA_SUCCESS) {
			Range r;
			d_p = new_ptr;
			reserve_sz = aligned_sz;
			r.start = new_ptr;
			r.sz = aligned_sz;
			va_ranges.push_back(r);
		}
	}
	else {
		Range r;
		r.start = new_ptr;
		r.sz = aligned_sz - reserve_sz;
		va_ranges.push_back(r);
		if (d_p == 0ULL) {
			d_p = new_ptr;
		}
		reserve_sz = aligned_sz;
	}

	return status;
}

CUresult VectorMemMap::grow(size_t new_sz)
{
	CUresult status = CUDA_SUCCESS;
	CUmemGenericAllocationHandle handle;
	if (new_sz <= alloc_sz) {
		return CUDA_SUCCESS;
	}

	const size_t size_diff = new_sz - alloc_sz;
	// Round up to the next chunk size
	const size_t sz = ((size_diff + chunk_sz - 1) / chunk_sz) * chunk_sz;

	if ((status = reserve(alloc_sz + sz)) != CUDA_SUCCESS) {
		return status;
	}

	if ((status = cuMemCreate(&handle, sz, &prop, 0)) == CUDA_SUCCESS) {
		if ((status = cuMemMap(d_p + alloc_sz, sz, 0ULL, handle, 0ULL)) == CUDA_SUCCESS) {
			if ((status = cuMemSetAccess(d_p + alloc_sz, sz, &accessDesc, 1ULL)) == CUDA_SUCCESS) {
				handles.push_back(handle);
				handle_sizes.push_back(sz);
				alloc_sz += sz;
			}
			if (status != CUDA_SUCCESS) {
				(void)cuMemUnmap(d_p + alloc_sz, sz);
			}
		}
		if (status != CUDA_SUCCESS) {
			(void)cuMemRelease(handle);
		}
	}

	return status;
}

__device__ int& VectorMemMap::at(unsigned int i) {
	//cudaPointerAttributes attr;
	//cudaPointerGetAttributes(&attr, d_p);
	int *ptr = *(int **) &d_p;
	//cuPointerGetAttribute(ptr, DevicePointer, d_p);
	return *(ptr + i);
}

__device__ void VectorMemMap::push_back(int e) {
	int idx = atomicAdd(d_size, 1);
	at(idx) = e;
}

__global__ void printVec(VectorMemMap *v) {
	printf("size: %d\n", *(v->d_size));
	return;
	for (int i = 0; i < *(v->d_size); ++i) {
		printf("%d ", v->at(i));
	}
	printf("\n");
}

/*
__global__ void initVec(LFVector *v, int n) {
	v->d_size = 0;
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
*/
int main(int argc, char **argv){
	if (argc < 2){
		fprintf(stderr, "execute as %s <n>\n", argv[0]);
		return -1;
	}

	int n = atoi(argv[1]);

	cudaDeviceSetLimit(cudaLimitMallocHeapSize, 4*n*sizeof(int));

	VectorMemMap *a;

	//gpuErrCheck( cudaMalloc(&a, sizeof(LFVector)) );

	//test_random_copy(a, n);
	//initVec<<<1,1>>>(a, n); kernelCallCheck();
	//printVec<<<1,1>>>(a); kernelCallCheck();
	//test_push_back<<<4,64>>>(a); kernelCallCheck();
	//printVec<<<1,1>>>(a); kernelCallCheck();
}

