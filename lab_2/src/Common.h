#ifndef CUDATIMER_H
#define CUDATIMER_H

#include <cuda_runtime.h>

class CudaTimer
{
public:
    CudaTimer(bool quite = false); 
    ~CudaTimer();

    static float avgElapsedTime;
private:
    bool q;
    cudaEvent_t start;
    cudaEvent_t stop;

};

#endif