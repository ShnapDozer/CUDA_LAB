#include <cuda_runtime.h>

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

#include "Common.h"

#define BLOCK_SIZE 256
#define RUN_COUNT 1000

__global__ void vectorMultiplyCoalesced(const int* A, const int* B, int* C, int N) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] * B[idx]; 
}

__global__ void vectorMultiplyNonCoalesced(const int* A, const int* B, int* C, int N) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N - 1) 
    {
        if(idx == 4)                    C[idx+1] = A[idx+1] * B[idx+1]; 
        else if(idx == 5)               C[idx-1] = A[idx-1] * B[idx-1];  
        else if(idx > 1 << 19)          C[idx+1] = A[idx+1] * B[idx+1]; 
        else                            C[idx]   = A[idx]   * B[idx]; 
    }

    if(idx == N - 1) { C[524289] = A[524289] * B[524289];  }
}

void verifyResults(const std::vector<int>& A, const std::vector<int>& B, const std::vector<int>& C, int N) 
{
    for (int i = 0; i < N; i++) 
    {
        if (C[i] != A[i] * B[i]) 
        {
            printf("Ошибка: элемент %d не совпадает!\n", i);
            return;
        }
    }
}

int main() 
{
    srand(time(0));
    const int N = 1 << 20; 
    
    size_t bytes = N * sizeof(int);
    std::vector<int> h_A(N), h_B(N), h_C(N);

    int *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    dim3 threads(BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    printf("\nРаботает Coalesced\n");

    for(int i = 0; i < RUN_COUNT; ++i)
    {
        for (int i = 0; i < N; i++)
        {
            h_A[i] = rand() % 1000 + 1; 
            h_B[i] = rand() % 1000 + 1;
        }
        
        cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);
        
        {
            CudaTimer t(true);
            vectorMultiplyCoalesced<<<blocks, threads>>>(d_A, d_B, d_C, N);
        }

        cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);
        verifyResults(h_A, h_B, h_C, N);
    }

    printf("\nСреднее время выполнения Coalesced: %.5f мс\n", CudaTimer::avgElapsedTime);

    CudaTimer::avgElapsedTime = 0;

    printf("\nРаботает Non-Coalesced\n");
    
    for(int i = 0; i < RUN_COUNT; ++i)
    {
        for (int i = 0; i < N; i++)
        {
            h_A[i] = rand() % 1000 + 1; 
            h_B[i] = rand() % 1000 + 1;
        }
        
        cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

        {
            CudaTimer t(true);
            vectorMultiplyNonCoalesced<<<blocks, threads>>>(d_A, d_B, d_C, N);
        }

        cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);
        verifyResults(h_A, h_B, h_C, N);
    }

    printf("\nСреднее время выполнения Non-Coalesced: %.5f мс\n", CudaTimer::avgElapsedTime);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
