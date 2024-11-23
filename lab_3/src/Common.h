#ifndef CUDATIMER_H
#define CUDATIMER_H

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

template <typename T>
void printVector(const std::vector<T>& vec) 
{
    std::cout << "[";
    for (size_t i = 0; i < vec.size(); ++i) 
    {
        std::cout << vec[i];
        if (i < vec.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";
}

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