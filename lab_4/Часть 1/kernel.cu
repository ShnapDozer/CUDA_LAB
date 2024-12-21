#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define STB_IMAGE_IMPLEMENTATION   
#include "../stb-master/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb-master/stb_image_write.h"
#include <stdio.h>

__global__ void transformKernel(float* output,
    cudaTextureObject_t texObj,
    int width, int height,
    float theta)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    float u = x / (float)width;
    float v = y / (float)height;

    u -= 0.5f;
    v -= 0.5f;
    float tu = u * cosf(theta) - v * sinf(theta) + 0.5f;
    float tv = v * cosf(theta) + u * sinf(theta) + 0.5f;

    output[y * width + x] = tex2D<float>(texObj, tu, tv);
}

char* toBinary(int n, int len)
{
    char* binary = (char*)malloc(sizeof(char) * len);
    int k = 0;
    for (unsigned i = (1 << len - 1); i > 0; i = i / 2) {
        binary[k++] = (n & i) ? '1' : '0';
    }
    binary[k] = '\0';
    return binary;
}

void print_binary(unsigned char* n)
{
    int len = 8;
    char* binary = toBinary(n[0], len);
    printf("The binary representation of %d is %s\n", n[0], binary);
}

void print_binary(float* n)
{
    int len = 32;
    char* binary = toBinary(n[0], len);
    printf("The binary representation of %d is %s\n", n[0], binary);
}

int main()
{
    int height = 320;
    int width = 213;
    int texChannels;
    float angle = 45;

    stbi_uc* pixels = stbi_load("./cat.bmp", &width, &height, &texChannels, STBI_grey);
    if(!pixels)  printf("not loaded\n");
    printf("loaded\n");

    float* h_data = (float*)malloc(sizeof(float) * width * height);
    for (int i = 0; i < height * width; ++i)
        h_data[i] = (pixels[i] & 0xff);

    print_binary(pixels);
    print_binary(h_data);

    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray_t cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);

    float* output;
    cudaMalloc(&output, width * height * sizeof(float));


    const size_t spitch = width * sizeof(float);
    cudaMemcpy2DToArray(cuArray, 0, 0, h_data, spitch, width * sizeof(float),
        height, cudaMemcpyHostToDevice);

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    
    dim3 threadsperBlock(16, 16);
    dim3 numBlocks((width + threadsperBlock.x - 1) / threadsperBlock.x, (height + threadsperBlock.y - 1) / threadsperBlock.y);
    transformKernel <<< numBlocks, threadsperBlock >>>(output, texObj, width, height, angle);
    
    cudaMemcpy(h_data, output, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    unsigned char* h_data_char = (unsigned char*)malloc(sizeof(unsigned char) * width * height);
  
    print_binary(h_data);

    for (int i = 0; i < height * width; ++i)
    {
        h_data_char[i] = (unsigned char)h_data[i];
    }
        
    print_binary(h_data_char);

    stbi_write_jpg("./cat-out.jpg", width, height, 1, h_data_char, 100);
    printf("write\n");
    
    cudaDestroyTextureObject(texObj);

    cudaFreeArray(cuArray);
    cudaFree(output);
    stbi_image_free(pixels);
    free(h_data);

    return 0;
}