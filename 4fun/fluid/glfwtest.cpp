// Se incluyen las cabeceras necesarias.
// <iostream> para imprimir errores en la consola.
#include <iostream>
// GLAD es una alternativa moderna para cargar funciones de OpenGL, 
// pero para este ejemplo simple, el propio GLFW es suficiente.
#define GL_GLEXT_PROTOTYPES
#include <GLFW/glfw3.h>

int main() {
    // 1. Inicializar GLFW
    if (!glfwInit()) {
        std::cerr << "Error al inicializar GLFW" << std::endl;
        return -1;
    }

    // 2. Crear una ventana
    GLFWwindow* window = glfwCreateWindow(800, 600, "Prueba de Ventana GLFW", NULL, NULL);
    if (!window) {
        std::cerr << "Error al crear la ventana con GLFW" << std::endl;
        glfwTerminate();
        return -1;
    }

    // 3. Hacer que el contexto de la ventana sea el contexto actual de OpenGL
    glfwMakeContextCurrent(window);
    
    // Opcional: Habilita V-Sync (sincronización vertical)
    glfwSwapInterval(1);

    // 4. Bucle principal (se ejecutará hasta que se cierre la ventana)
    while (!glfwWindowShouldClose(window)) {
        // Establecer el color de fondo (R, G, B, Alfa). 
        // 0.0, 0.6, 0.6 es un tono de cian.
        glClearColor(0.0f, 0.6f, 0.6f, 1.0f);
        
        // Limpiar el buffer de color con el color definido arriba
        glClear(GL_COLOR_BUFFER_BIT);

        // 5. Intercambiar los buffers (frontal y trasero)
        // El buffer trasero (donde dibujamos) se muestra en la pantalla.
        glfwSwapBuffers(window);

        // 6. Procesar eventos (como clics, teclado, cierre de ventana)
        glfwPollEvents();
    }

    // 7. Liberar recursos y terminar GLFW
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}