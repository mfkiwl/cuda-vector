#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <mma.h>
#include "../common/utility.cuh"

#define ull unsigned long long int
#define WARPSIZE 32
#define BSIZE 1024
#define NB 100
#define PROB 90
#define DEBUG 1

using namespace nvcuda;


// atomic
__global__ void atomic(int *C, int *A, int *s, int n) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= n) return;
        C[tid] = atomicAdd(s, A[tid]);
}

__device__ void insert_atomic(int *a, int e, int *size, int q) {
	int idx = atomicAdd(size, 1);
	a[idx] = e;
}


// scan
__inline__ __device__ int warp_scan(int val, int lane){
	for (int offset = 1; offset < WARPSIZE; offset <<= 1) {
                int n = __shfl_up_sync(0xffffffff, val, offset, WARPSIZE);
		if ((lane & 31) >= offset)
			val += n;
	}
	return val;
}

__inline__ __device__ int block_scan(int val){
        static __shared__ int shared[WARPSIZE];
        int tid = threadIdx.x;
        int lane = tid & (WARPSIZE-1);
        int wid = tid/WARPSIZE;
        val = warp_scan(val, lane);
        if(lane == WARPSIZE-1)
                shared[wid] = val;

        __syncthreads();
        if(wid == 0){
                int t = (tid < blockDim.x/WARPSIZE) ? shared[lane] : 0;
                t = warp_scan(t, lane);
                shared[lane] = t;
        }
        __syncthreads();
        if (wid > 0){
                val += shared[wid-1];
        }
        return val;
}

__global__ void scan(int *C, int *A, int *s, int n) {
        __shared__ int ss;
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= n) return;
        int val = block_scan(A[tid]);
        if (threadIdx.x == BSIZE - 1 || tid == n - 1) {
                ss = atomicAdd(s, val);
        }
        __syncthreads();
        C[tid] = val + ss - A[tid];
}

__device__ void insert_scan(int *a, int e, int *size, int q) {
        __shared__ int ss;
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int n = *size;
	if (tid >= n) return;
        int val = block_scan(q);
        if (threadIdx.x == BSIZE - 1 || tid == n - 1) {
                ss = atomicAdd(size, val);
        }
        __syncthreads();
        int idx = val + ss - q;
	//printf("tid %d: %d", tid, idx);
	if (q)
		a[idx] = e;
}


// tensor core scan
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

static __device__ half upper_triang[256];
static __device__ half lower_triang[256];

__global__ void load_matrices() {
	int tid = threadIdx.x;
	int i = tid / 16;
	int j = tid % 16;
	if (i <= j) {
		upper_triang[tid] = 1;
		lower_triang[tid] = 0;
	} else {
		upper_triang[tid] = 0;
		lower_triang[tid] = 1;
	}
}

__inline__ __device__ float tensor_block_scan(half val) {
	__shared__ half vals[BSIZE];
	//__shared__ half upper_triang[256];
	//__shared__ half lower_triang[256];
	__shared__ half add[4];
	int tid = threadIdx.x;
	int wid = tid / WARPSIZE;
	vals[tid] = val;

	if (wid < 4) {
		wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
		wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
		wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
		wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> au_frag;
		wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> la_frag;

		// AU = A * UPPER + 0
		wmma::fill_fragment(c_frag, 0.0f);
		wmma::load_matrix_sync(a_frag, vals + 256*wid, 16);
		wmma::load_matrix_sync(b_frag, upper_triang, 16);

		wmma::mma_sync(au_frag, a_frag, b_frag, c_frag);

		// LA = LOWER * A + 0
		wmma::load_matrix_sync(a_frag, lower_triang, 16);
		wmma::load_matrix_sync(b_frag, vals, 16);

		wmma::mma_sync(la_frag, a_frag, b_frag, c_frag);

		// R = LA * 1 + AU
		wmma::store_matrix_sync(vals + wid*256, la_frag, 16,  wmma::mem_row_major);
		wmma::load_matrix_sync(a_frag, vals + wid*256, 16);
		wmma::fill_fragment(b_frag, 1.0f);

		wmma::mma_sync(c_frag, a_frag, b_frag, au_frag);

		wmma::store_matrix_sync(vals + wid*256, c_frag, 16,  wmma::mem_row_major);

	}
	__syncthreads();

	// combine warps
	if (tid == 0) {
		half i = 0;
		add[0] = 0;
		for (int j = 1; j < 4; ++j) {
			i += vals[j*256 - 1];
			add[j] = i;
		}
	}
	__syncthreads();

	vals[tid] += add[tid / 256];

	return vals[tid];
}

__global__ void tensor_scan(int *C, int *A, int *s, int n) {
	__shared__ int ss;
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
	float pval = tid < n ? A[tid] : 0;
	printf("n  tid %d: %f\n", tid, pval);
	__syncthreads();
	int val = tensor_block_scan(pval);
	__syncthreads();
	printf("tid %d: %d\n", tid, val);
	if (tid >= n) return;
        if (threadIdx.x == BSIZE - 1 || tid == n - 1) {
                C[blockIdx.x] = atomicAdd(s, val);
        }
        __syncthreads();
	int asd = C[blockIdx.x];
	C[tid] = val + asd - A[tid];
}

__global__ void test_block(int *out) {
	int tid = threadIdx.x;
	int val = tensor_block_scan(1);
	out[tid] = (int)val;
}

__device__ void insert_tensor_scan(int *a, int e, int *size, int q) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int n = *size;
	float pval = tid < n ? q : 0;
	printf("n  tid %d: %f\n", tid, pval);
	__syncthreads();
	int val = tensor_block_scan(pval);
	__syncthreads();
	printf("tid %d: %d\n", tid, val);
	if (tid >= n) return;
        if (threadIdx.x == BSIZE - 1 || tid == n - 1) {
                a[n + blockIdx.x] = atomicAdd(size, val);
		printf("tid %d: %d\n", tid, *size);
        }
        __syncthreads();
	int asd = a[n + blockIdx.x];
	int idx = val + asd - q;
        __syncthreads();
	//printf("tid %d: %d\n", tid, idx);
	if (q)
		a[idx] = e;
}

// test
__global__ void test_insert(int* v, int n, int *size) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= n) return;
	insert_tensor_scan(v, v[tid], size, 1);
}

void test_scan(int n) {
	int *hA, *dA, *hC, *dC, *ds, hs;
        hA = new int[n];
	hC = new int[n];
        for (int i = 0; i < n; ++i) {
                hA[i] = 1;
                //hA[i] = i%2;
        }
        hs = 0;
        cudaMalloc(&dA, sizeof(int)*n);
        cudaMalloc(&dC, sizeof(int)*n);
        cudaMalloc(&ds, sizeof(int));
        cudaMemcpy(dA, hA, sizeof(int)*n, cudaMemcpyHostToDevice);
        cudaMemcpy(ds, &hs, sizeof(int), cudaMemcpyHostToDevice);

	load_matrices<<<1,256>>>(); kernelCallCheck();

	//test_block<<<1,1024>>>(dC); kernelCallCheck();
	//cudaMemcpy(hC, dC, 1024*sizeof(int), cudaMemcpyDeviceToHost);
	//print_array(hC, 1024);


	//return 0;
        cudaEvent_t start1, stop1;
        start_clock(start1, stop1);
	{
		//atomic<<<gridSize(n, BSIZE), BSIZE>>>(dC, dA, ds, n); kernelCallCheck();
		//scan<<<gridSize(n, BSIZE), BSIZE>>>(dC, dA, ds, n); //kernelCallCheck();
		tensor_scan<<<gridSize(n, BSIZE), BSIZE>>>(dC, dA, ds, n); kernelCallCheck();
	}
        float time = stop_clock(start1, stop1);

	if (DEBUG) {
		gpuErrCheck( cudaMemcpy(hC, dC, n*sizeof(int), cudaMemcpyDeviceToHost) );
		print_array(hA, n, "array A:");
		print_array(hC, n, "array C:");
	}


        printf("time: %f ms\n", time);
        printf("%f,", time);

}

void run_experiment() {
	int size = 1<<19;
	int ratio = 1;
	int rep = 10;
	int o_size = size;

	int *a, *ha;
	int *dsize;
	ha = new int[size];
	for (int i = 0; i < size; ++i) {
		ha[i] = i;
	}
	gpuErrCheck( cudaMalloc(&a, 2*size*2^rep*sizeof(int)) );
	gpuErrCheck( cudaMalloc(&dsize, sizeof(int)) );
	gpuErrCheck( cudaMemcpy(a, ha, size*sizeof(int), cudaMemcpyHostToDevice)) ;
	gpuErrCheck( cudaMemcpy(dsize, &size, sizeof(int), cudaMemcpyHostToDevice) );


	float results[rep];

	for (int i = 0; i < rep; ++i) {
		cudaEvent_t start, stop;
		start_clock(start, stop);
		test_insert<<<gridSize(size, BSIZE), BSIZE>>>(a, size, dsize);
		cudaDeviceSynchronize();
		results[i] = stop_clock(start, stop);
		cudaMemcpy(&size, dsize, sizeof(int), cudaMemcpyDeviceToHost);
	}

	// print results
	printf("static,in,%d,%d,", o_size, ratio);
	for (int i = 0; i < rep-1; ++i) {
		printf("%f,", results[i]);
	}
	printf("%f\n", results[rep-1]);
	//printf("%f\n", s);
}


int main(int argc, char **argv){
	if (argc < 2) {
                fprintf(stderr, "Ejecutar como ./prog n\n");
                return -1;
        }
        int size = atoi(argv[1]);

	test_scan(size);
	//run_experiment();
	return 0;

	int *a, *ha;
	int *dsize;
	ha = new int[size];
	for (int i = 0; i < size; ++i) {
		ha[i] = i;
	}
	gpuErrCheck( cudaMalloc(&a, 2*size*sizeof(int)) );
	gpuErrCheck( cudaMalloc(&dsize, sizeof(int)) );
	gpuErrCheck( cudaMemcpy(a, ha, size*sizeof(int), cudaMemcpyHostToDevice)) ;
	gpuErrCheck( cudaMemcpy(dsize, &size, sizeof(int), cudaMemcpyHostToDevice) );

	test_insert<<<gridSize(size, BSIZE), BSIZE>>>(a, size, dsize); kernelCallCheck();
	cudaDeviceSynchronize();
	
	int *ha2 = new int[2*size];
	cudaMemcpy(ha2, a, sizeof(int)*size*2, cudaMemcpyDeviceToHost);
	print_array(ha2, size*2);



	return 0;
}
