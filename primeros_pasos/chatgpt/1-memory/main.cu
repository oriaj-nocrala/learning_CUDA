#include <stdio.h>

__global__ void write_ids(int* output) {
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    output[global_id] = global_id * global_id;
}


int main(){
    int N = 8;

    // 1. Reservar memoria en la GPU
    int* device_array;
    cudaMalloc(&device_array, N * sizeof(int));

    // 2. Lanzar el kernel
    write_ids<<<2, 4>>>(device_array);  // 2 bloques de 4 = 8 hilos
    cudaDeviceSynchronize();

    // 3. Copiar resultado al host
    int host_array[N];
    cudaMemcpy(host_array, device_array, N * sizeof(int), cudaMemcpyDeviceToHost);

    // 4. Verificar
    for (int i = 0; i < N; ++i) {
        printf("host_array[%d] = %d\n", i, host_array[i]);
    }

    // 5. Liberar
    cudaFree(device_array);

}