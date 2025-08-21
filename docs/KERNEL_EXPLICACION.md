# Guía del Kernel de CUDA: La Física del Pelo (`update_hair_kernel`)

Este documento explica en detalle el funcionamiento del kernel de CUDA que simula la física de cada hebra de pelo. El kernel se ejecuta en paralelo para miles de pelos simultáneamente en la GPU.

La filosofía detrás del kernel se basa en una técnica llamada **Dinámica Basada en Posiciones (Position-Based Dynamics o PBD)**. En lugar de calcular fuerzas complejas, la idea es mucho más simple e intuitiva:

1.  **Mover:** Mueve los puntos (los segmentos del pelo) a su siguiente posición basándose en la física simple (gravedad, inercia).
2.  **Corregir:** Después de moverlos, recorre una serie de "reglas" (restricciones) que corrigen las posiciones para que el pelo no se estire, no atraviese objetos y se mantenga anclado.

Este enfoque es muy estable y perfecto para ejecutar en la GPU.

---

### La Firma del Kernel

```c++
__global__ void update_hair_kernel(
    Hair* hairs, 
    vec3* vbo_ptr, 
    float dt, 
    vec3 sphere_center, 
    vec3 sphere_velocity, 
    vec3 interaction_sphere_center, 
    bool interaction_active
);
```

-   `Hair* hairs`: Un puntero a la memoria de la GPU que contiene los datos de estado de todos los pelos (posiciones, anclajes, etc.).
-   `vec3* vbo_ptr`: El puntero al búfer de vértices (VBO) de OpenGL. Aquí es donde el kernel escribirá las posiciones finales para que se dibujen.
-   `dt`: El "delta time" o el tiempo transcurrido desde el último fotograma. Esencial para que la física sea consistente.
-   `sphere_center`, `sphere_velocity`: La posición y velocidad de la esfera principal.
-   `interaction_sphere_center`, `interaction_active`: La posición de la "mano" y si está activa o no.

---

### Paso 1: Integración de Verlet y Gravedad (El Movimiento)

```c++
// 1. Integración Verlet y gravedad
for (int j = 0; j < HAIR_SEGMENTS; ++j) {
    vec3 temp = hairs[i].pos[j];
    hairs[i].pos[j] += (hairs[i].pos[j] - hairs[i].old_pos[j]) * 0.98f + vec3(0.0f, -1.5f, 0.0f) * dt * dt;
    hairs[i].old_pos[j] = temp;
}
```

-   **Integración de Verlet:** Esta es la forma en que calculamos el movimiento. En lugar de una física compleja, simplemente deducimos la nueva posición basándonos en la actual (`pos`) y la anterior (`old_pos`). Esto nos da inercia de forma implícita.
-   **Amortiguación (`* 0.98f`):** Multiplicamos el movimiento por un factor ligeramente menor que 1 para que el pelo pierda energía lentamente y no se mueva para siempre.
-   **Gravedad:** Añadimos una pequeña fuerza hacia abajo (`vec3(0.0f, -1.5f, 0.0f)`) para que el pelo tenga una caída sutil y natural.

Al final de este paso, cada segmento del pelo se ha movido, pero ignorando todas las reglas. El resultado es un caos de pelos estirados y que atraviesan la esfera.

---

### Paso 2: El Solucionador de Restricciones (La Corrección)

Esta es la parte más importante. Se ejecuta en un bucle (`for k < 15`) para aplicar las correcciones varias veces, lo que hace que el resultado sea más estable y rígido.

```c++
// 2. Solucionar restricciones
for (int k = 0; k < 15; ++k) {
    // ... aquí van las 4 reglas de corrección ...
}
```

#### A. Anclaje de la Raíz (El Folículo)

```c++
// a) Anclar la raíz a la superficie
 hairs[i].pos[0] = sphere_center + hairs[i].anchor_direction * SPHERE_RADIUS;
```
-   **Función:** Esta es la regla más importante. Asegura que la base de cada pelo (`pos[0]`) esté siempre "clavada" en su punto de origen en la superficie de la esfera, usando la dirección de anclaje única que le dimos al crearlo.

#### B. Propagación de la Rigidez (La Curva Natural)

```c++
// b) Propagar la dirección del folículo a lo largo del pelo
for (int j = 0; j < HAIR_SEGMENTS - 1; ++j) {
    // ...
}
```
-   **Función:** Esta es la regla que le da al pelo su aspecto erizado y su curva natural. Funciona en dos partes para cada par de segmentos:
    1.  **Restricción de Distancia:** Primero, asegura que los dos segmentos no estén ni más lejos ni más cerca de lo que dicta el `segment_length`. Si se han separado, los junta. Si se han solapado, los separa.
    2.  **Propagación de Dirección:** Luego, calcula la dirección "ideal" del segmento (que es la dirección de su folículo) y la mezcla con la dirección actual. El factor `stiffness` controla cuánto se impone la dirección ideal. Esto crea una curva suave en lugar de una esquina rígida.

#### C y D. Colisiones (Con la Esfera y con la "Mano")

```c++
// c) Colisión con la esfera principal
// d) Colisión con la esfera de interacción (la "mano")
if (dist_to_center < SPHERE_RADIUS) {
    vec3 normal = seg_to_center / dist_to_center;
    hairs[i].pos[j] = sphere_center + normal * SPHERE_RADIUS;
}
```
-   **Función:** Estas dos reglas son idénticas en su lógica. Para cada segmento del pelo, comprueban si está "dentro" de alguna de las esferas. Si lo está, calculan el vector normal (la dirección desde el centro de la esfera hacia el segmento) y lo usan para "empujar" el segmento justo a la superficie. Esto evita que el pelo atraviese los objetos.

---

### Paso 3: Actualización Final (Velocidad y Salida a Gráficos)

```c++
// 3. Actualizar velocidad y escribir en el VBO
for (int j = 0; j < HAIR_SEGMENTS; ++j) {
    hairs[i].vel[j] = (hairs[i].pos[j] - hairs[i].old_pos[j]) / dt;
    vbo_ptr[i * HAIR_SEGMENTS + j] = hairs[i].pos[j];
}
```
-   **Actualizar Velocidad:** Después de todas las correcciones, calculamos la velocidad final. Esto es necesario para que la inercia funcione en el siguiente fotograma.
-   **Escribir en el VBO:** Este es el paso final y crucial. La posición corregida y definitiva de cada segmento se escribe directamente en el puntero del VBO (`vbo_ptr`), que es la memoria que OpenGL usará para dibujar. Esto conecta la simulación directamente con el renderizado.
