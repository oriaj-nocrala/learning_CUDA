#pragma once

// --- Parámetros Físicos y de Simulación ---
const int WIN_SIZE = 768;
const int NUM_BALLS = 4000;
const float BALL_RADIUS = 0.1f;
const float BOX_SIZE = 5.0f;
const float FRICTION = 0.1f;
const float RESTITUTION = 0.5f; // Coeficiente de restitución para colisiones con paredes

// --- Parámetros de la Cuadrícula Espacial ---
const float CELL_SIZE = BALL_RADIUS * 2.0f;
const int GRID_DIM_X = (int)(BOX_SIZE * 2 / CELL_SIZE) + 1;
const int GRID_DIM_Y = (int)(BOX_SIZE * 2 / CELL_SIZE) + 1;
const int GRID_DIM_Z = (int)(BOX_SIZE * 2 / CELL_SIZE) + 1;
const int NUM_CELLS = GRID_DIM_X * GRID_DIM_Y * GRID_DIM_Z;

// --- Parámetros de Fuerza de Colisión ---
const float STIFFNESS = 5000.0f; // Rigidez del "resorte" de colisión
const float DAMPING = 10.0f;    // Amortiguación para estabilizar
const float LINEAR_DAMPING = 0.02f; // Amortiguación lineal para simular arrastre
const float VELOCITY_STOP_THRESHOLD = 0.05f; // Umbral para detener completamente las pelotas lentas
