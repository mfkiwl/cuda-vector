#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "../common/utility.cuh"

#define ull unsigned long long int
#define BSIZE 1024
#define PROB 90
#define FBS 1024
#define logFBS 10
using namespace std;

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
	__device__ T& at(unsigned int i);
	__device__ int get_bucket(unsigned int i);
	__device__ void new_bucket(unsigned int b);
	__device__ void push_back(T e);
	__device__ void grow(unsigned int n);
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
__device__ void LFVector<T>::new_bucket(unsigned int b) {
	//printf("inside new_bucket %d\n", b);
	int old = atomicCAS(isbucket + b, 0, 1);
	if (old == 0) {
		int bsize = 1 << (logFBS + b);
		a[b] = (T*)malloc(sizeof(T) * bsize);
	}
}

template <typename T>
__device__ void LFVector<T>::push_back(T e) {
	int idx = atomicAdd(&size, 1);
	int b = get_bucket(idx);
	if (a[b] == nullptr) {
		new_bucket(b);
	}
	__syncthreads();
	at(idx) = e;
}

template <typename T>
__device__ void LFVector<T>::grow(unsigned int n) {
	int b1 = get_bucket(size);
	int b2 = get_bucket(n);
	for (int b = b1+1; b <= b2; ++b) {
		new_bucket(b);
		isbucket[b] = 1;
	}
}

template <typename T, int NB>
struct Vector {
	unsigned int size;
	LFVector<T> *lfv;
	unsigned int *ranges;
	__device__ Vector();
	__device__ T& at(unsigned int i);
	__device__ void insert(T e, int q);
	__device__ void grow(unsigned int n);
};

template <typename T, int NB>
__device__ Vector<T, NB>::Vector() {
	size = 0;
	lfv = (LFVector<T>*)malloc(sizeof(LFVector<T>)*NB);
	ranges = (unsigned int*)malloc(sizeof(unsigned int)*NB);
	for (int i = 0; i < NB; ++i) {
		ranges[i] = 0;
		lfv[i] = LFVector<T>();
	}
}

template <typename T, int NB>
__device__ T& Vector<T, NB>::at(unsigned int i) {
	// TODO use warp instructions
	int b = NB-1;
	while (i < ranges[b]) {
		--b;
	}
	return lfv[b].at(i-ranges[b]);
}

template <typename T, int NB>
__device__ void Vector<T, NB>::insert(T e, int q) {
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

template <typename T, int NB>
__device__ void Vector<T, NB>:: grow(unsigned int n) {
	int tid = threadIdx.x;
	int sub_size = (n + NB - 1) / NB;
	lfv[tid].grow(sub_size);
}

template <typename T, int NB>
__global__ void growVec(Vector<T, NB> *v, unsigned int n) {
	v->grow(n);
}


template <typename T, int NB>
__global__ void createLFVector(Vector<T, NB> *v) {
	*v = Vector<T, NB>();
	return;
	v->size = 0;
	for (int i = 0; i < NB; ++i) {
		v->ranges[i] = 0;
		v->lfv[i].grow(0);
	}
}

template <typename T, int NB>
__global__ void initVec(Vector<T, NB> *v, unsigned int n) {
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

template <typename T, int NB>
__global__ void initVec(Vector<T, NB> *v, unsigned int n, T* in) {
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

template<typename T, int NB>
__global__ void get_size(int *out, Vector<T, NB> *v) {
	*out = v->size;
}

template<typename T, int NB>
__global__ void vec2array(T *out, Vector<T, NB> *v) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	out[tid] = v->at(tid);
}

template<typename T, int NB>
int sendToHost(T* &out, Vector<T, NB> *v) {
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

template<int NB>
__global__ void initVec(Vector<int, NB> *v) {
	*v = Vector<int, NB>();
	v->size = NB*BSIZE;
	for (int i = 0; i < NB; ++i) {
		v->ranges[i] = BSIZE*i;
		v->lfv[i].grow(BSIZE);
	}
	for(int i = 0; i < v->size; ++i) {
		v->at(i) = i;
	}
}

template<int NB>
__global__ void printVec(Vector<int, NB> *v) {
	printf("size: %d\n", v->size);
	// pint lfv[0]
	//for (int i = 0; i < 64; ++i) {
		//printf("%d ", v->lfv[0].isbucket[i]);
	//}
	
	printf("ranges: ");
	for (int i = 0; i < NB; ++i) {
		printf("%d ", v->ranges[i]);
	}
	printf("\nsizes: ");
	for (int i = 0; i < NB; ++i) {
		printf("%d ", v->lfv[i].size);
	}
	printf("\n");
	//return;
	for (int i = 0; i < v->size; ++i) {
		printf("%d ", v->at(i));
	}
	printf("\n");
	printf("last element: %d\n", v->at(v->size - 1));
}

// low level memMap
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

inline __device__ int &at(CUdeviceptr d_p, int i) {
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


// static
inline __device__ int &at(int *a, unsigned int i) {
	return a[i];
}

__device__ void insert_atomic(int *a, int e, int *size, int q) {
	int idx = atomicAdd(size, 1);
	a[idx] = e;
}
