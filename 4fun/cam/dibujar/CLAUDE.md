# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Run Commands

- **Compile and run**: `./compile.sh` - Compiles the CUDA application using nvcc with OpenCV libraries and runs it
- **Manual compile**: `nvcc main.cu -o dibujar_cuda \`pkg-config --cflags --libs opencv4\``
- **Run executable**: `./dibujar_cuda`

## Project Architecture

This is a CUDA-accelerated computer vision application that performs real-time hand detection and finger tracking for digital drawing. The application creates a digital drawing canvas by detecting skin tone and tracking the index finger position.

### Key Components

- **main.cu**: Single-file application containing both CUDA kernels and host code
- **CUDA Kernel (`skin_detection_kernel`)**: GPU-accelerated skin detection using HSV color space with multiple skin tone ranges
- **Hand Analysis**: Sophisticated hand detection using convex hull, convexity defects, and geometric analysis
- **Finger Detection**: Identifies individual fingers and specifically tracks the index finger tip

### Architecture Flow

1. **Camera Capture**: OpenCV captures frames from webcam (device 0)
2. **GPU Processing**: Frame data is copied to GPU memory where CUDA kernel performs skin detection
3. **CPU Post-processing**: Advanced morphological operations, Gaussian blur, and contour analysis
4. **Hand Recognition**: Convex hull calculation, convexity defects analysis, and finger tip detection
5. **Index Finger Tracking**: Geometric analysis to identify and track the index finger specifically
6. **Drawing Logic**: Index finger tip position creates smooth drawing lines with variable thickness
7. **Display**: Shows camera feed with debug overlays, drawing canvas, and skin detection mask

### Dependencies

- CUDA Runtime
- OpenCV 4 (cv::VideoCapture, cv::imgproc, cv::highgui)
- pkg-config for OpenCV linking

### Hand Detection Parameters

- **Skin detection**: Multiple HSV ranges covering different skin tones
- **Minimum hand area**: 3000 pixels to filter noise
- **Finger detection**: Distance-based filtering and geometric analysis
- **GPU thread configuration**: 16x16 thread blocks
- **Morphological operations**: Elliptical kernel with opening/closing operations