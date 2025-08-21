nvcc -std=c++17 -O3 main.cu kernels.cu glad.c -I. -lglfw -lGL -o template && ./template
