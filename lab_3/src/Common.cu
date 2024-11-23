#include "Common.h"

#include <iostream>

float CudaTimer::avgElapsedTime = 0.f;

CudaTimer::CudaTimer(bool quite) 
    : q(quite)
{
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
}

CudaTimer::~CudaTimer() 
{
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    if(avgElapsedTime != 0)
    {
        avgElapsedTime += elapsedTime;
        avgElapsedTime = avgElapsedTime / 2.f;
    }
    else avgElapsedTime += elapsedTime;
    
    if(!q) printf("Время выполнения: %.5f мс\n", elapsedTime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}