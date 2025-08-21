#include <stdio.h>

__global__ void mi_kernel() {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Soy el hilo global %d (bloque %d, hilo local %d)\n", global_id, blockIdx.x, threadIdx.x);
}

int main() {
    mi_kernel<<<2, 4>>>();
    cudaDeviceSynchronize(); // Asegura que terminen antes de cerrar
    return 0;
}