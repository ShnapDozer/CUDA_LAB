#include <cuda_runtime.h>

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>
#include <algorithm>

#include "Common.h"

#define BLOCK_SIZE 256
#define MIN_VALUE 5
#define RUN_COUNT 10
#define N_DEF 1 << 10

__global__ void reduceMin(const int* inData, int* outData, int N) 
{ 
    __shared__ int sharedData[N_DEF];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sharedData[tid] = (idx < N) ? inData[idx] : INT_MAX;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) 
    {
        if (tid < stride) 
        {
            sharedData[tid] = min(sharedData[tid], sharedData[tid + stride]);
        }
        __syncthreads();
    }
    
    if (tid == 0) *outData = sharedData[0];
}

int main() 
{
    srand(time(0));

    const int N = N_DEF; 
    size_t bytes = N * sizeof(int);

    dim3 threads(BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2));

    int minVal;
    std::vector<int> h_A(N);
    
    int* d_A;
    int* d_out;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_out, sizeof(int));

    printf("\nРаботает reduceMin\n");

    for(int i = 0; i < RUN_COUNT; ++i)
    {
        for (int i = 0; i < N; i++) h_A[i] = rand() % 1000 + MIN_VALUE;
        cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
        
        {
            CudaTimer t(true);
            reduceMin<<<blocks, threads>>>(d_A, d_out, N);
        }

        cudaMemcpy(&minVal, d_out, sizeof(int), cudaMemcpyDeviceToHost);

        if(minVal != *std::min_element(h_A.begin(), h_A.end())) { printf("Минимальные значения не совпадают %d != %d\n", minVal, MIN_VALUE); }
    }

    printf("\nСреднее время выполнения: %.5f мс\n", CudaTimer::avgElapsedTime);

    cudaFree(d_A);
    cudaFree(d_out);

    return 0;
}