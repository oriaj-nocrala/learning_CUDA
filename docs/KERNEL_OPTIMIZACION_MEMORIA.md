# Entendiendo la Memoria en CUDA: Global vs. Compartida (`__shared__`)

La lección más importante en la programación de CUDA es que **no toda la memoria es igual**. Entender la jerarquía de memoria de la GPU es la clave para pasar de un kernel que funciona a un kernel que vuela. En nuestro caso, nos centraremos en las dos memorias más importantes para la optimización: la global y la compartida.

### Una Analogía: La Fábrica de la GPU

Imagina que la GPU es una enorme fábrica y que cada **bloque de hilos** (por ejemplo, nuestro grupo de 256 hilos) es un equipo de trabajo al que se le asigna una tarea (simular 256 pelos).

-   La **Memoria Global** es el **almacén central** de la fábrica. Es gigantesco (varios Gigabytes), contiene todas las piezas que se necesitan y es accesible para todos los equipos de la fábrica. Sin embargo, está lejos de los puestos de trabajo, y cada vez que un trabajador necesita una pieza, tiene que hacer un viaje largo y lento para ir a buscarla. Si todos los trabajadores van constantemente al almacén por piezas pequeñas, pasarán más tiempo caminando que trabajando.

-   La **Memoria Compartida (`__shared__`)** es una **mesa de trabajo pequeña y comunal** que pertenece exclusivamente a un equipo. Es diminuta (solo unos pocos Kilobytes), pero está justo al lado de los trabajadores. Pueden poner las piezas que necesitan para su tarea actual en esa mesa y compartirlas entre ellos casi instantáneamente, sin tener que volver al gran almacén. Es extremadamente rápida.

El objetivo de la optimización es minimizar los viajes al "almacén" (memoria global) y hacer la mayor parte del trabajo usando la "mesa de trabajo" (memoria compartida).

---

### Memoria Global

-   **Características:**
    -   **Tamaño:** Muy grande (GBs).
    -   **Velocidad:** Muy lenta (alta latencia, cientos de ciclos de reloj).
    -   **Alcance:** Accesible por todos los hilos de todos los bloques, y también por la CPU (es la memoria que usamos para las transferencias con `cudaMemcpy`).
    -   **Uso Típico:** Almacenamiento principal para los datos de entrada y los resultados finales.

-   **El Problema en Nuestro Kernel:** En el bucle de restricciones (`for k < 15`), cada hilo está constantemente leyendo y escribiendo las posiciones de los segmentos de su pelo y de los pelos vecinos. Cada una de estas operaciones es un viaje al lento "almacén central", lo que acumula una enorme latencia.

### Memoria Compartida (`__shared__`)

-   **Características:**
    -   **Tamaño:** Muy pequeño (típicamente 48 KB o 96 KB por bloque de hilos).
    -   **Velocidad:** Extremadamente rápida (baja latencia, comparable a la caché L1).
    -   **Alcance:** Solo es visible para los hilos que pertenecen al mismo bloque. Sus datos desaparecen cuando el bloque termina su ejecución.
    -   **Uso Típico:** Un "caché" manual y programable para que los hilos de un bloque colaboren y compartan datos de forma eficiente.

---

### El Patrón de Optimización: De Global a Compartida

Para optimizar nuestro kernel, seguiríamos estos pasos:

#### Paso 1: Declarar la Memoria Compartida

Dentro del kernel, declaramos un array con el calificador `__shared__`. Su tamaño debe coincidir con el número de hilos de nuestro bloque.

```c++
// Cada bloque de 256 hilos tendrá su propio array local de 256 pelos.
__shared__ Hair local_hairs[256];
```

#### Paso 2: Cargar los Datos desde Global a Compartida (Una Sola Vez)

Al principio del kernel, cada hilo del bloque se encarga de copiar un pelo desde la memoria global (el "almacén") a la memoria compartida (la "mesa de trabajo").

```c++
int global_id = blockIdx.x * blockDim.x + threadIdx.x; // ID del pelo en el array global
int local_id = threadIdx.x; // ID del hilo dentro del bloque (0-255)

// Cada hilo copia su pelo correspondiente a la memoria compartida
if (global_id < NUM_HAIRS) {
    local_hairs[local_id] = hairs[global_id];
}

// ¡CRÍTICO! Sincronizar todos los hilos del bloque.
__syncthreads();
```

La llamada a `__syncthreads()` es fundamental. Actúa como una barrera y le dice a cada hilo: "Espera aquí hasta que todos los demás hilos de tu equipo hayan terminado de traer su pieza del almacén". Esto garantiza que `local_hairs` esté completamente lleno antes de que nadie empiece a trabajar con él.

#### Paso 3: Realizar Todos los Cálculos en Memoria Compartida

Ahora, todo el bucle de restricciones se modifica para que lea y escriba únicamente en el array `local_hairs`, que es súper rápido.

```c++
// El bucle de física ahora opera sobre la rapidísima memoria compartida
for (int k = 0; k < 15; ++k) {
    // Leemos y escribimos en local_hairs[local_id]
    vec3& p1 = local_hairs[local_id].pos[j];
    // ... etc.

    // Podríamos necesitar más __syncthreads() aquí si los hilos
    // necesitan leer los resultados de otros hilos en la misma iteración.
}
```

#### Paso 4: Escribir el Resultado Final de Vuelta a la Memoria Global

Después de que el bucle de restricciones termina, los datos en `local_hairs` son correctos y finales. El último paso es que cada hilo copie el resultado de su trabajo (la posición final de los vértices) de vuelta a la memoria global, que en nuestro caso es el puntero al VBO de OpenGL.

```c++
// Cada hilo escribe el resultado final en el VBO para que OpenGL lo dibuje.
for (int j = 0; j < HAIR_SEGMENTS; ++j) {
    vbo_ptr[global_id * HAIR_SEGMENTS + j] = local_hairs[local_id].pos[j];
}
```

Al seguir este patrón, reducimos los costosos accesos a la memoria global de `15 * (muchas lecturas y escrituras)` a solo **una lectura inicial y una escritura final**, lo que puede resultar en un aumento significativo del rendimiento del kernel.
