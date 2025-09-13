#!/bin/bash
# CUDA Optimized Compilation Script
# Compiles the optimized CUDA version with maximum performance flags

echo "=== Compiling CUDA Optimized Version ==="
echo "Target: ~350 FPS (vs ~30 FPS CPU)"

nvcc -std=c++17 -O3 -DRELEASE \
     --use_fast_math \
     -Xcompiler -fopenmp \
     -Xcompiler -march=native \
     main.cu kernels.cu glad.c \
     -I. -lglfw -lGL -o bolas_cuda_optimized

if [ $? -eq 0 ]; then
    echo "✓ Compilation successful!"
    echo "Run with: ./bolas_cuda_optimized"
    echo "Features: CUDA-OpenGL interop, optimized kernels, VSync disabled"
else
    echo "✗ Compilation failed!"
    exit 1
fi