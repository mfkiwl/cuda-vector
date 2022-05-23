#include <cuda.h>
#include <assert.h>
#include <iostream>
#include <vector>
#include "utility.cuh"
using namespace std;


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
    VectorMemMap(CUcontext context);
    ~VectorMemMap();

    CUdeviceptr getPointer() const {
        return d_p;
    }

    size_t getSize() const {
        return alloc_sz;
    }
    size_t getReserve() const {
        return reserve_sz;
    }

    // Reserves some extra space in order to speed up grow()
    CUresult reserve(size_t new_sz);

    // Actually commits num bytes of additional memory
    CUresult grow(size_t new_sz);
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

CUresult
VectorMemMap::reserve(size_t new_sz)
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

CUresult
VectorMemMap::grow(size_t new_sz)
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

__device__ int &at(CUdeviceptr d_p, unsigned int i) {
	return (*(int**)&d_p)[i];
}

__device__ void insert_atomic(CUdeviceptr d_p, int e, int *size, int q) {
	int idx = atomicAdd(size, 1);
	at(d_p, idx) = e;
}

__global__ void initVec(CUdeviceptr d_p, unsigned int n) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= n) return;
	at(d_p, tid) = tid;
}

__global__ void initVec(CUdeviceptr d_p, unsigned int n, int* in) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= n) return;
	at(d_p, tid) = in[tid];
}

__global__ void test(CUdeviceptr d_p, size_t n) {
	for (size_t i = 0; i < n; ++i) {
		at(d_p, i) = i;
	}
}

__global__ void printVec(CUdeviceptr d_p, size_t n) {
	for (size_t i = 0; i < n; ++i) {
		printf("%d ", at(d_p, i));
	}
	printf("\n");
}

__global__ void test_insert_atomic(CUdeviceptr v, int n, int *size) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= n) return;
	insert_atomic(v, at(v, tid), size, 1);
}

__global__ void test_read_write(CUdeviceptr v, int size) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= size) return;
	at(v, tid) += 1;
}

// low level api test
void run_experiment(CUcontext ctx, int size, int ratio) {
	int rep = 10;
	int rw_rep = 30;
	int o_size = size;
	int *ds;
	cudaMalloc(&ds, sizeof(int));
	cudaMemcpy(ds, &size, sizeof(int), cudaMemcpyHostToDevice);

	VectorMemMap a = VectorMemMap(ctx);
	CUresult status;

	status = a.grow(size*sizeof(int));
	initVec<<<gridSize(size, 1024), 1024>>>(a.getPointer(), size); kernelCallCheck();

	float results[rep];
	float results_rw[rw_rep];
	
	for (int i = 0; i < rep; ++i) {
		cudaEvent_t start, stop;
		start_clock(start, stop);
		status = a.grow(size*2*sizeof(int));
		test_insert_atomic<<<gridSize(size, 1024), 1024>>>(a.getPointer(), size, ds);
		results[i] = stop_clock(start, stop);
		cudaMemcpy(&size, ds, sizeof(int), cudaMemcpyDeviceToHost);

		// read/write
		results_rw[i] = 0.0;
		for (int j = 0; j < rw_rep; ++j) {
			cudaEvent_t start, stop;
			start_clock(start, stop);
			test_read_write<<<gridSize(size, 1024), 1024>>>(a.getPointer(), size); kernelCallCheck();
			results_rw[i] += stop_clock(start, stop);
		}
		results_rw[i] /= rw_rep;
	}

	// print results
	printf("memMap,%d,%d,", o_size, ratio);
	for (int i = 0; i < rep-1; ++i) {
		printf("%f,", results[i]);
	}
	printf("%f\n", results[rep-1]);
	//printf("%f\n", s);
	printf("memMap,%d,%d,", o_size, ratio);
	for (int i = 0; i < rep-1; ++i) {
		printf("%f,", results_rw[i]);
	}
	printf("%f\n", results_rw[rep-1]);
}

int main(int argc, char **argv){
	int size = 1e6;
	int ratio = 2;
	cudaSetDevice(0);
	//cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1e7*sizeof(int));
	CUcontext ctx;
	cuDevicePrimaryCtxRetain(&ctx, 0);
	cuCtxSetCurrent(ctx);
	
	//size_t free;
	//cuMemGetInfo(&free, NULL);
	//cout << "Total Free Memory: " <<
		//(float)free << endl;
	
	run_experiment(ctx, size, ratio);

	/*
	VectorMemMap a = VectorMemMap(ctx);
	CUresult status;

	cout << "size: " << a.getSize() << endl;
	status = a.reserve(n*sizeof(int));
	cout << "size: " << a.getSize() << endl;
	status = a.grow(n*sizeof(int));
	cout << "size: " << a.getSize() << " reserve: " << a.getReserve() << endl;

	test<<<1,1>>>(a.getPointer(), n); kernelCallCheck();
	//printVec<<<1,1>>>(a.getPointer(), n); kernelCallCheck();
	
	//status = a.grow(1000000);
	//cout << "size: " << a.getSize() << " reserve: " << a.getReserve() << endl;

	n = 1e7;
	status = a.grow(n*sizeof(int));
	cout << "status: " << status << endl;
	cout << "size: " << a.getSize() << " reserve: " << a.getReserve() << endl;
	test<<<1,1>>>(a.getPointer(), n); kernelCallCheck();
	*/
	
	cuDevicePrimaryCtxRelease(0);
	return 0;
}

