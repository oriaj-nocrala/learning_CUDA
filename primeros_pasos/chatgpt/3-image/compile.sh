#/bin/bash

nvcc main.cu $(pkg-config --cflags --libs opencv4)