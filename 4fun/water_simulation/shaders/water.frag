#version 330 core
out vec4 FragColor;
in vec2 TexCoord;

uniform sampler2D waterTexture;

void main() {
    float density = texture(waterTexture, TexCoord).r;
    
    // Water-like color mapping
    vec3 waterColor = vec3(0.2, 0.6, 1.0); // Blue water
    vec3 foamColor = vec3(0.9, 0.95, 1.0); // White foam
    
    // Mix colors based on density
    float alpha = clamp(density * 2.0, 0.0, 1.0);
    vec3 color = mix(vec3(0.0), mix(waterColor, foamColor, density * 0.5), alpha);
    
    FragColor = vec4(color, alpha);
}