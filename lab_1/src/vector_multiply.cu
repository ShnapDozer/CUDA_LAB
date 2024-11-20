#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define BLOCK_SIZE 256

class CudaTimer
{
public:
    CudaTimer() 
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    }

    ~CudaTimer() 
    {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, start, stop);

        printf("%.5f мс |\n", elapsedTime);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

private:
    cudaEvent_t start;
    cudaEvent_t stop;
};

__global__ void vectorMultiply(const int *A, const int *B, int *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] * B[idx];
    }
}

void verifyResults(const int *A, const int *B, const int *C, int N) {
    for (int i = 0; i < N; i++) 
    {
        if (C[i] != A[i] * B[i]) 
        {
            printf("Ошибка в элементе %d: %d * %d != %d\n", i, A[i], B[i], C[i]);
            return;
        }
    }
    printf("Результаты верны!\n");
}

int main() 
{
    printf("Размер векторов | Время выполнения\n");

    const int N = 1 << 25;
    size_t bytes = N * sizeof(int);

    printf("| %d | ", N);

    int *h_A = (int *)malloc(bytes);
    int *h_B = (int *)malloc(bytes);
    int *h_C = (int *)malloc(bytes);

    srand(time(NULL));
    for (int i = 0; i < N; i++) 
    {
        h_A[i] = rand() % 100 + 1;
        h_B[i] = rand() % 100 + 1;
    }

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 threads(BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    {
        CudaTimer t;
        vectorMultiply<<<blocks, threads>>>(d_A, d_B, d_C, N);
    }

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    verifyResults(h_A, h_B, h_C, N);

    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
