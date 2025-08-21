#include <stdio.h>

__global__ void square_elements(int* d_A, int* d_B, int N){
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(global_thread_id < N) {
        d_B[global_thread_id] = d_A[global_thread_id] * d_A[global_thread_id];
    }
}

int main(){
    int N = 8;
    int A[N] = { 1, 2, 3, 4, 5, 6, 7, 8 };
    int B[N]; // Resultado

    int *d_A, *d_B;
    cudaMalloc(&d_A, N * sizeof(int));
    cudaMalloc(&d_B, N * sizeof(int));

    cudaMemcpy(d_A, A, N * sizeof(int), cudaMemcpyHostToDevice);

    square_elements<<<1, 4>>>(d_A, d_B, N);
    cudaDeviceSynchronize();

    cudaMemcpy(B, d_B, N * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < N; i++){
        printf("El cuadrado de %d es: %d\n", A[i], B[i]);
    }

    cudaFree(d_A);
    cudaFree(d_B);

}