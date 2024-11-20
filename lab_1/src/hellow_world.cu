#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

#define N (1024*1024)

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

      printf("Время выполнения: %.5f мс\n", elapsedTime);

      cudaEventDestroy(start);
      cudaEventDestroy(stop);
    }

private:
    cudaEvent_t start;
    cudaEvent_t stop;
};

__global__ void kernel(float * data)
{
   int idx     = blockIdx.x * blockDim.x + threadIdx.x;
   float x     = 2.0f * 3.1415926f * (float) idx / (float) N;
   data[idx]   = sinf(sqrtf(x));
}

void getInfo()
{
   int  deviceCount;
   cudaDeviceProp devProp;

   cudaGetDeviceCount ( &deviceCount );
   printf             ( "Found %d devices\n", deviceCount );

   for ( int device = 0; device < deviceCount; device++ )
   {
      cudaGetDeviceProperties ( &devProp, device );  
      printf ( "Device %d\n", device );
      printf ( "Compute capability     : %d.%d\n", devProp.major, devProp.minor );
      printf ( "Name                   : %s\n", devProp.name );
      printf ( "Total Global Memory    : %u\n", devProp.totalGlobalMem );
      printf ( "Shared memory per block: %d\n", devProp.sharedMemPerBlock );
      printf ( "Registers per block    : %d\n", devProp.regsPerBlock );
      printf ( "Warp size              : %d\n", devProp.warpSize );
      printf ( "Max threads per block  : %d\n", devProp.maxThreadsPerBlock );
      printf ( "Total constant memory  : %d\n", devProp.totalConstMem );
   }

}

int main(int argc, char *argv[])
{
   getInfo();

   float* a    = (float*)malloc(N * sizeof(float));
   float* dev  = nullptr;

   cudaMalloc ((void**)&dev, N * sizeof(float));

   dim3 threads(1024, 1, 1);
   dim3 blocks(N / threads.x, 1, 1);

   printf("Запуск на threads: %d, blocks: %d\n", threads.x, blocks.x);

   {
      CudaTimer t;
      kernel<<<blocks, threads>>>(dev); 
   }

   cudaMemcpy(a, dev, N * sizeof(float), cudaMemcpyDeviceToHost);

   free(a);
   cudaFree(dev);

   return 0;
}

