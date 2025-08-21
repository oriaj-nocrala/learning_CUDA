#include "../include/water_app.h"

WaterApp::WaterApp() : window(nullptr) {
}

WaterApp::~WaterApp() {
    cleanup();
}

bool WaterApp::initialize() {
    if (!init_glfw()) {
        return false;
    }
    
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to initialize GLAD" << std::endl;
        return false;
    }
    
    input.setup_callbacks(window);
    glfwSwapInterval(1); // Enable vsync
    
    if (!renderer.initialize()) {
        std::cerr << "Failed to initialize renderer" << std::endl;
        return false;
    }
    
    physics.initialize();
    
    glClearColor(0.1f, 0.1f, 0.2f, 1.0f); // Dark blue background
    
    return true;
}

void WaterApp::cleanup() {
    physics.cleanup();
    renderer.cleanup();
    
    if (window) {
        glfwDestroyWindow(window);
        window = nullptr;
    }
    glfwTerminate();
}

bool WaterApp::init_glfw() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return false;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
    window = glfwCreateWindow(WIN_SIZE, WIN_SIZE, "Water Simulation - CUDA", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return false;
    }
    
    glfwMakeContextCurrent(window);
    
    return true;
}

void WaterApp::run() {
    while (!glfwWindowShouldClose(window)) {
        handle_input();
        update();
        render();
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}

void WaterApp::handle_input() {
    input.update();
    
    if (input.is_mouse_down()) {
        double mx, my, last_mx, last_my;
        input.get_mouse_position(mx, my);
        input.get_last_mouse_position(last_mx, last_my);
        
        // Convert mouse coordinates to world space [0,1]
        float world_x = mx / WIN_SIZE;
        float world_y = 1.0f - (my / WIN_SIZE); // Invert Y coordinate
        
        float2 mouse_pos = make_float2(world_x, world_y);
        float2 mouse_vel = make_float2((mx - last_mx) / WIN_SIZE * 10.0f, 
                                     -(my - last_my) / WIN_SIZE * 10.0f); // Invert Y velocity
        
        // Interact with water at mouse position
        if (world_x > 0.0f && world_x < 1.0f && world_y > 0.0f && world_y < 1.0f) {
            // Add fewer particles, only occasionally
            static int frame_counter = 0;
            if (frame_counter % 5 == 0) { // Add particles every 5 frames
                physics.add_particles_at(mouse_pos, 2); // Add only 2 particles
            }
            frame_counter++;
            
            // Always apply force interaction
            physics.add_particle_interaction(mouse_pos, mouse_vel);
        }
    }
}

void WaterApp::update() {
    physics.simulate_step();
}

void WaterApp::render() {
    renderer.render(physics.get_particles(), physics.get_particle_count());
}