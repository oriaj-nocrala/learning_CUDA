#!/bin/bash

# Script para compilar la simulación de pelotas con CUDA y OpenGL (usando Spatial Grid)

# Limpiar compilaciones anteriores
rm -f bolas_cuda_grid

# Compilar el archivo renombrado con las rutas de inclusión correctas
nvcc main_grid.cu kernels.cu glad.c -o bolas_cuda_grid -I. -I./KHR -lglfw -lGL -Xcompiler "-g"

# Comprobar si la compilación fue exitosa
if [ $? -eq 0 ]; then
    echo "Compilación exitosa. Ejecutable: ./bolas_cuda_grid"
else
    echo "Error en la compilación."
fi
