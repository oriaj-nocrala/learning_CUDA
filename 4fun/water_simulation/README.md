# SPH Water Simulation with CUDA

A modular 2D water simulation implemented in CUDA using Smoothed Particle Hydrodynamics (SPH) for realistic water physics.

## Features

- **SPH Physics**: Smoothed Particle Hydrodynamics for realistic water behavior
- **CUDA Acceleration**: High-performance GPU-based particle simulation
- **Spatial Hashing**: Efficient neighbor finding for thousands of particles
- **Interactive**: Click and drag to add particles and create currents
- **Particle Rendering**: Beautiful water particles with transparency
- **Real-time**: Optimized for smooth performance with up to 8K particles

## Project Structure

```
water_simulation/
├── src/                    # Source files
│   ├── main.cpp           # Application entry point
│   ├── water_app.cpp      # Main application class
│   ├── sph_physics.cu     # SPH physics simulation (CUDA)
│   ├── sph_kernels.cu     # SPH CUDA kernels
│   ├── sph_renderer.cpp   # Particle rendering
│   └── input_handler.cpp  # Mouse/keyboard input
├── include/               # Header files
│   ├── common.h          # Common definitions
│   ├── sph_common.h      # SPH-specific definitions
│   ├── water_app.h       # Application class
│   ├── sph_physics.h     # SPH physics interface
│   ├── sph_renderer.h    # Particle renderer interface
│   └── input_handler.h   # Input handling
├── shaders/              # GLSL shaders
│   ├── water.vert        # Vertex shader
│   └── water.frag        # Fragment shader
├── glad/                 # OpenGL loader
├── build/                # Build output
├── Makefile             # Build system
└── compile.sh           # Build script
```

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit
- OpenGL 3.3+
- GLFW3
- Linux (Ubuntu/Debian tested)

## Building

### Install Dependencies (Ubuntu/Debian)
```bash
make install-deps
```

### Build and Run
```bash
# Build the project
make

# Run the simulation
make run

# Clean build files
make clean
```

## Usage

- **Left Click + Drag**: Add water particles and apply forces
- **Hold and Drag**: Create continuous water stream
- **ESC**: Exit the simulation

## Physically-Inspired Stable Physics

The simulation uses water-based physics scaled for numerical stability:
- **Particle Count**: Up to 8,192 particles  
- **Particle Radius**: 5mm (realistic SPH particle size)
- **Smoothing Length**: 1cm (2 × particle radius)
- **Scaled Density**: Based on real water density (1000 kg/m³)
- **Tuned Viscosity**: Physically motivated but stable
- **Scaled Gravity**: 10% of Earth's gravity for gentle dynamics
- **Time Step**: 5ms (stable integration)
- **Simulation Domain**: 1m × 1m (real-world scale)

## SPH Physics Implementation

The simulation implements stable SPH with physically-motivated equations:
- **Density**: Poly6 kernel summation: ρᵢ = Σⱼ mⱼ W(rᵢⱼ,h)
- **Pressure**: Linear equation of state: P = k(ρ - ρ₀) with clamping
- **Pressure Forces**: Monaghan SPH: fᵢᴾ = -Σⱼ mⱼ(Pᵢ+Pⱼ)/(2ρⱼ) ∇W(rᵢⱼ,h)
- **Viscosity Forces**: Physical viscosity: fᵢᵛ = μ Σⱼ mⱼ(vⱼ-vᵢ)/ρⱼ ∇²W(rᵢⱼ,h)
- **Integration**: Leapfrog with velocity damping and clamping
- **Stability**: Pressure clamping, velocity limits, gentle forces
- **Spatial Hashing**: O(n) neighbor finding for performance

## Rendering

Particles are rendered as:
- **Circular Points**: GPU-accelerated point sprites
- **Water Colors**: Blue gradient from center to edge
- **Alpha Blending**: Transparent particles for depth effect
- **Real-time Updates**: Direct particle position rendering

## Based On

This implementation uses Smoothed Particle Hydrodynamics (SPH), a computational method for simulating fluid flows. SPH is particularly well-suited for water simulation as it naturally handles free surfaces, splashing, and complex boundaries.