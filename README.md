# 🚀 Learning CUDA

Un repositorio de aprendizaje y experimentación con CUDA (Compute Unified Device Architecture) para programación de GPU.

## 📁 Estructura del Proyecto

### 🎯 Primeros Pasos (`primeros_pasos/`)
Ejercicios básicos y tutoriales iniciales para familiarizarse con CUDA:
- **chatgpt/**: Ejemplos generados con IA para aprender conceptos básicos
  - Manejo de memoria CUDA
  - Arrays de entrada y procesamiento
  - Procesamiento de imágenes
- **udemy/**: Ejercicios de cursos en línea

### 🎮 Proyectos Divertidos (`4fun/`)
Proyectos más avanzados y experimentales:

#### 💧 **SPH-CUDA**
Simulación de fluidos usando Smoothed Particle Hydrodynamics con visualización en tiempo real.

#### ⚽ **Bolas Cayendo** 
Simulación física de esferas con colisiones y gravedad usando CUDA.

#### 🌊 **Water Simulation**
Simulación avanzada de agua con renderizado OpenGL.

#### 💨 **Fluid Simulation**
Experimentos con dinámicas de fluidos en GPU.

#### 🦌 **Fur Simulation**
Simulación de pelo/cabello usando técnicas de GPU computing.

#### 📸 **Cam (Interpolation)**
Súper resolución de imágenes usando modelos ONNX y CUDA.

### 📚 Documentación (`docs/`)
- **KERNEL_EXPLICACION.md**: Explicación detallada de kernels CUDA
- **KERNEL_OPTIMIZACION_MEMORIA.md**: Técnicas de optimización de memoria
- **OPTIMIZACIONES.md**: Estrategias de optimización general
- **CONCLUSION.md**: Conclusiones y aprendizajes

## 🛠️ Requisitos

- **NVIDIA GPU** con soporte CUDA
- **CUDA Toolkit** (versión 11.0 o superior recomendada)
- **CMake** para algunos proyectos
- **OpenGL/GLFW** para proyectos con visualización
- **Compilador compatible** (gcc, nvcc)

## 🚀 Compilación

La mayoría de proyectos incluyen scripts de compilación:

```bash
# Para proyectos individuales
cd 4fun/nombre_proyecto/
./compile.sh

# Para proyectos con CMake
mkdir build && cd build
cmake ..
make
```

## 📖 Cómo Usar Este Repositorio

1. **Principiantes**: Comienza con `primeros_pasos/` para entender conceptos básicos
2. **Intermedios**: Explora `4fun/` para ver aplicaciones prácticas
3. **Avanzados**: Revisa la documentación en `docs/` para optimizaciones

## 🎯 Conceptos Cubiertos

- ✅ Fundamentos de CUDA
- ✅ Gestión de memoria (Host ↔ Device)
- ✅ Kernels y threading
- ✅ Programación paralela
- ✅ Optimización de rendimiento
- ✅ Integración con OpenGL
- ✅ Simulaciones físicas
- ✅ Procesamiento de imágenes
- ✅ Interoperabilidad CUDA-OpenGL

## 🏗️ Estado del Proyecto

🔄 **En desarrollo activo** - Agregando nuevos experimentos y mejorando los existentes.

## 📝 Notas

- Cada proyecto tiene su propio README con instrucciones específicas
- Los binarios compilados están excluidos del repositorio (.gitignore)
- Se incluyen tanto ejemplos básicos como proyectos avanzados

## 🤝 Contribuciones

¡Las sugerencias y mejoras son bienvenidas! Este es un proyecto de aprendizaje en constante evolución.

---

*Desarrollado con ❤️ y muchas horas de debugging en GPU*