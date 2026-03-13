#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cmath>
#include <string>
#include <cuda_runtime.h>
#include <chrono>

// -------------------------
// CUDA error macro
// -------------------------
#define CUDA_CHECK(stmt)                                                     \
do {                                                                         \
    cudaError_t err = (stmt);                                                \
    if (err != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA ERROR %s (%d): %s at %s:%d\n",                 \
                #stmt, int(err), cudaGetErrorString(err), __FILE__, __LINE__); \
        std::exit(EXIT_FAILURE);                                             \
    }                                                                        \
} while (0)

// -------------------------
// Helper to init input data
// -------------------------
static void init_random(float* v, long long n, unsigned long long seed = 42ULL) {
    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (long long i = 0; i < n; i++) v[i] = dist(gen);
}

// -------------------------
// Kernel to flush L2 cache by reading and writing a large buffer in global memory
// -------------------------
__global__ void flush_l2(float* dummy_buffer, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < n) {
        // volatile to prevent optimize out
        volatile float* buf = dummy_buffer;
        float val = buf[id];
        // write back to memory to ensure it goes through L2
        dummy_buffer[id] = val + 1.0f;
    }
}

// -------------------------
// Helper to force data back to CPU and flush L2 cache 
// -------------------------
void reset_system_state(float* dummy_buffer, size_t L2_SIZE, float* A, float* B, float* C, size_t sizeA, size_t sizeB, size_t sizeC) {
    //request Unified Memory to migrate these buffers to CPU memory
    cudaMemPrefetchAsync(A, sizeA, cudaCpuDeviceId);
    cudaMemPrefetchAsync(B, sizeB, cudaCpuDeviceId);
    cudaMemPrefetchAsync(C, sizeC, cudaCpuDeviceId);

    cudaDeviceSynchronize();

    // flush L2 cache using the dummy buffer
    int n = L2_SIZE / sizeof(float);
    int threads = 256; 
    int blocks = (n + threads -1) /threads;
    flush_l2<<<blocks, threads>>>(dummy_buffer, n);

    CUDA_CHECK(cudaGetLastError());

    cudaDeviceSynchronize();
}


// ------------------------------------------------------------------
// Tiled shared-memory GEMM (row-major), fixed TILE size
// ------------------------------------------------------------------
#define TILE 32
__global__ void tiled_gemm_sm(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* C,
                              const unsigned int M, const unsigned int N, const unsigned int K) {

    __shared__ float A_s[TILE][TILE];
    __shared__ float B_s[TILE][TILE]; 

    unsigned int row = blockIdx.y*blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x; 

    float sum = 0.0f;
    unsigned int numTiles = (K + TILE - 1)/TILE;

    for(unsigned int tile = 0; tile < numTiles; ++tile) {
        //Load tile to shared memory 
        unsigned int tiled_col_A = tile*TILE + threadIdx.x;
        unsigned int tiled_row_B = tile*TILE + threadIdx.y;

        //out of range loads are zero-filled
        if(row < M && tiled_col_A < K)
            A_s[threadIdx.y][threadIdx.x] = A[row*K + tiled_col_A];
        else 
            A_s[threadIdx.y][threadIdx.x] = 0.0f;
            
        if(tiled_row_B < K && col < N) 
            B_s[threadIdx.y][threadIdx.x] = B[(tiled_row_B)*N + col];
        else 
            B_s[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads(); 

        //Compute with tile
        for(unsigned int i = 0; i < TILE; ++i) {
            sum += A_s[threadIdx.y][i]*B_s[i][threadIdx.x];
        }
        __syncthreads();
    }

    //bounds check: C has dimension M × N
    if(row < M && col < N) 
        C[row * N + col] = sum;
    
}

int main() { 
    
    const int M = 4096; //num tokens in prompt
    const int N = 4096; // sequence length 
    const int K = 128; //head dimension- this is model-dependent, but many use 128 

    size_t sizeA = (size_t)M * K * sizeof(float);
    size_t sizeB = (size_t)K * N * sizeof(float);
    size_t sizeC = (size_t)M * N * sizeof(float);

    float *A, *B, *C;

    // allocate using UVM
    CUDA_CHECK(cudaMallocManaged(&A, sizeA));
    CUDA_CHECK(cudaMallocManaged(&B, sizeB));
    CUDA_CHECK(cudaMallocManaged(&C, sizeC));

    // intialize A, B on CPU 
    init_random(A, (long long)M*K);
    init_random(B, (long long)K*N);

    // streams
    cudaStream_t compute_stream;
    cudaStream_t prefetch_stream;
    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    CUDA_CHECK(cudaStreamCreate(&prefetch_stream));

    // kernel launch configuration
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1)/TILE,
              (M + TILE - 1)/TILE); 

    // L2 information
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);
    size_t L2_SIZE = deviceProp.l2CacheSize;
    size_t FLUSH_SIZE = L2_SIZE * 2; //large enough to force eviction of L2 

    // initialize dummy buffer 
    float* dummy_buffer;
    CUDA_CHECK(cudaMalloc(&dummy_buffer, FLUSH_SIZE));
    
    // warmup hardware + driver 
    reset_system_state(dummy_buffer, L2_SIZE, A, B, C, sizeA, sizeB, sizeC); 

    // begin microbenchmark experiments 
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float elapsed; 

    // =========================
    // M1 : Demand Paging
    // =========================
    // WORST CASE 
    // Data allocated via cudaMallocManaged on CPU; kernel accesses it without any prior migration

    // flush L2 and force data back to CPU
    reset_system_state(dummy_buffer, L2_SIZE, A, B, C, sizeA, sizeB, sizeC);

    CUDA_CHECK(cudaEventRecord(start, compute_stream));

    tiled_gemm_sm<<<grid, block, 0, compute_stream>>>(A, B, C, M, N, K);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop, compute_stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    printf("M1 (Demand Paging- worst case) Kernel Time: %f ms\n", elapsed);
    CUDA_CHECK(cudaDeviceSynchronize());

    // =========================
    // M2: Ideal Prefetch 
    // =========================
    // BEST CASE 
    // Same allocation, but cudaMemPrefetchAsync called on the entire buffer before the kernel launches on a separate stream

    // flush L2 and force data back to CPU
    reset_system_state(dummy_buffer, L2_SIZE, A, B, C, sizeA, sizeB, sizeC);

    // prefetch to GPU 
    CUDA_CHECK(cudaMemPrefetchAsync(A, sizeA, deviceId));
    CUDA_CHECK(cudaMemPrefetchAsync(B, sizeB, deviceId));
    CUDA_CHECK(cudaMemPrefetchAsync(C, sizeC, deviceId));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start, compute_stream));

    tiled_gemm_sm<<<grid, block, 0, compute_stream>>>(A, B, C, M, N, K);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop, compute_stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    printf("M2 (Ideal Prefetch) Kernel Time: %f ms\n", elapsed);
    CUDA_CHECK(cudaDeviceSynchronize());

    // =========================
    // M3 : Overlapped Prefetch
    // =========================
    // REALISTIC CASE
    // Issue cudaMemPrefetchAsync on a dedicated non-blocking prefetch stream, then immediately 
    // launch the kernel on the compute stream without waiting for the prefetch to complete

    // flush L2 and force data back to CPU
    reset_system_state(dummy_buffer, L2_SIZE, A, B, C, sizeA, sizeB, sizeC);

    cudaEvent_t prefetch_started;
    CUDA_CHECK(cudaEventCreate(&prefetch_started));
    
    CUDA_CHECK(cudaEventRecord(start, compute_stream));

    // launch prefetches on the nonblocking prefetch stream 
    // question- do we need to time and then substract the prefetch overhead   
    CUDA_CHECK(cudaMemPrefetchAsync(A, sizeA, deviceId, prefetch_stream));
    CUDA_CHECK(cudaMemPrefetchAsync(B, sizeB, deviceId, prefetch_stream));
    CUDA_CHECK(cudaMemPrefetchAsync(C, sizeC, deviceId, prefetch_stream));

    CUDA_CHECK(cudaEventRecord(prefetch_started, prefetch_stream));

    // the wait event
    // tell the compute stream to wait until the prefetch stream reaches the marker
    // because cudaMeMPrefetchAsync is nonblocking, the kernel can start
    // while the hardware is still streaming pages in the background
    // compute_Stream is the stream that will pause
    // prefetch_started is the event the stream is waiting for 
    CUDA_CHECK(cudaStreamWaitEvent(compute_stream, prefetch_started, 0));

    tiled_gemm_sm<<<grid, block, 0, compute_stream>>>(A, B, C, M, N, K);    

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop, compute_stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    printf("M3 (Overlapped Prefetch) Kernel Time: %f ms\n", elapsed);

    // cleanup 
    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));
    CUDA_CHECK(cudaFree(dummy_buffer));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(prefetch_started));

    CUDA_CHECK(cudaStreamDestroy(compute_stream)); 
    CUDA_CHECK(cudaStreamDestroy(prefetch_stream));

    return 0;
}