#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void hello_cuda(){
    printf("Grid x: %d, y: %d\t Block x: %d, y: %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
}

int main(){
    int nx = 16, ny = 4;
    dim3 block(8,2);
    dim3 grid(nx / block.x , ny / block.y);
    hello_cuda<<<grid,block>>>();
    cudaDeviceSynchronize();
    cudaDeviceReset();
    return 0;
}