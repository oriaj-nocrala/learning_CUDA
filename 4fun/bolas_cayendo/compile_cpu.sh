#!/bin/bash
# CPU Optimized Compilation Script
# Compiles the CPU version with maximum optimization flags

echo "=== Compiling CPU Optimized Version ==="
echo "Target: ~30 FPS (baseline for comparison)"

g++ -std=c++17 -O3 \
    -fopenmp \
    -march=native \
    -ffast-math \
    -funroll-loops \
    main_cpu.cpp glad.c \
    -I. -lglfw -lGL -lpthread -o bolas_cpu_optimized

if [ $? -eq 0 ]; then
    echo "✓ Compilation successful!"
    echo "Run with: ./bolas_cpu_optimized"
    echo "Features: OpenMP parallelization, CPU sub-stepping physics"
else
    echo "✗ Compilation failed!"
    exit 1
fi