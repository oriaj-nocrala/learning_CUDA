nvcc -std=c++17 -O3 main.cu glad.c -I. -lglfw -lGL -o fluids
./fluids