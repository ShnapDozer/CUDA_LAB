#include <cuda_runtime.h>

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

#include "Common.h"

#define BLOCK_SIZE 16
#define RUN_COUNT 100

__global__ void transpose(const int* inData, int* outData, int n) 
{
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    if (xIndex < n && yIndex < n) 
    {
        unsigned int inIndex = yIndex * n + xIndex;
        unsigned int outIndex = xIndex * n + yIndex;

        outData[outIndex] = inData[inIndex];
    }
}

void verifyTranspose(const std::vector<int>& input, const std::vector<int>& output, int n) 
{
    for (int i = 0; i < n; i++) 
    {
        for (int j = 0; j < n; j++) 
        {
            if (input[i * n + j] != output[j * n + i]) 
            {
                printf("Ошибка: транспонирование неверно на элементе [%d][%d]\n", i, j);
                return;
            }
        }
    }
}

int main() 
{
    const int   N       = 256;     
    size_t      bytes   = N * N * sizeof(int);

    std::vector<int> h_inMatrix(N * N);
    std::vector<int> h_outMatrix(N * N, 0);

    int* d_inMatrix;
    int* d_outMatrix;

    cudaMalloc(&d_inMatrix, bytes);
    cudaMalloc(&d_outMatrix, bytes);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    for(int i = 0; i < RUN_COUNT; ++i) 
    {
        srand(time(0));
        for (int i = 0; i < N * N; i++) h_inMatrix[i] = rand() % 1000 + 1;
        cudaMemcpy(d_inMatrix, h_inMatrix.data(), bytes, cudaMemcpyHostToDevice);

        {
            CudaTimer t(true);

            transpose<<<blocks, threads>>>(d_inMatrix, d_outMatrix, N);
            cudaDeviceSynchronize();
        }

        cudaMemcpy(h_outMatrix.data(), d_outMatrix, bytes, cudaMemcpyDeviceToHost);
        verifyTranspose(h_inMatrix, h_outMatrix, N);
    }

    printf("\nСреднее время выполнения Coalesced: %.5f мс\n", CudaTimer::avgElapsedTime);

    cudaFree(d_inMatrix);
    cudaFree(d_outMatrix);

    return 0;
}