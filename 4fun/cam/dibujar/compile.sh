nvcc main.cu -o dibujar_cuda `pkg-config --cflags --libs opencv4`
./dibujar_cuda