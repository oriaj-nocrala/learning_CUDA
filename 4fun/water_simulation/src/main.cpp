#include "../include/water_app.h"

int main() {
    WaterApp app;
    
    if (!app.initialize()) {
        std::cerr << "Failed to initialize Water Simulation application" << std::endl;
        return -1;
    }
    
    std::cout << "Water Simulation initialized successfully!" << std::endl;
    std::cout << "Click and drag to interact with the water." << std::endl;
    
    app.run();
    
    return 0;
}