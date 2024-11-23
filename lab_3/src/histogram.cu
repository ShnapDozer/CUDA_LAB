#include <cuda_runtime.h>

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

#include "Common.h"

#define NUM_BINS 256
#define BLOCK_SIZE 256
#define RUN_COUNT 1000

__global__ void histogram(const int* data, int* histogram, int N)
 {
    __shared__ int localHist[NUM_BINS];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (threadIdx.x < NUM_BINS) localHist[threadIdx.x] = 0;
    __syncthreads();

    if (tid < N) atomicAdd(&localHist[data[tid]], 1);
    __syncthreads();

    if (threadIdx.x < NUM_BINS) atomicAdd(&histogram[threadIdx.x], localHist[threadIdx.x]);
}

int main() 
{
    srand(time(0));

    const int N = 1 << 20;
    
    size_t bytes = N * sizeof(int);
    size_t histBytes = NUM_BINS * sizeof(int);

    std::vector<int> h_A(N);
    std::vector<int> h_hist(NUM_BINS);

    int* d_A;
    int* d_hist;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_hist, histBytes);
    cudaMemset(d_hist, 0, histBytes);

    dim3 threads(BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    printf("\nРаботает histogram\n");

    for(int i = 0; i < RUN_COUNT; ++i)
    {
        for (int i = 0; i < N; i++) h_A[i] = rand() % NUM_BINS;
        cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
        
        {
            CudaTimer t(true);
            histogram<<<blocks, threads>>>(d_A, d_hist, N);
        }
    }

    printf("\nСреднее время выполнения Coalesced: %.5f мс\n", CudaTimer::avgElapsedTime);

    cudaMemcpy(h_hist.data(), d_hist, histBytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < NUM_BINS; i++) std::cout << "Элемент " << i << ": " << h_hist[i] << "\n";

    cudaFree(d_A);
    cudaFree(d_hist);

    return 0;
}
