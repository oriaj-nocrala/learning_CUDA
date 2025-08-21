#include <stdio.h>

__global__ void thread_in_array() {
    
}

int main(){
    int* array = (int*)malloc(sizeof(int) * 4 );
    
    thread_in_array<<<2, 2>>>();
}