#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <cuda_runtime.h>

// --- MACRO PARA VERIFICAR ERRORES DE CUDA ---
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ==================================================================
// KERNEL DE CUDA PARA DETECCIÓN DE COLOR
// ==================================================================
// Convierte un píxel de BGR a HSV y comprueba si está en el rango de piel.
__device__ void bgrToHsv(const uchar3& bgr, float& h, float& s, float& v) {
    float r_f = bgr.z / 255.0f;
    float g_f = bgr.y / 255.0f;
    float b_f = bgr.x / 255.0f;

    float cmax = fmaxf(r_f, fmaxf(g_f, b_f));
    float cmin = fminf(r_f, fminf(g_f, b_f));
    float delta = cmax - cmin;

    // Cálculo de Hue (Tonalidad)
    if (delta == 0) {
        h = 0;
    } else if (cmax == r_f) {
        h = 60 * fmodf(((g_f - b_f) / delta), 6);
    } else if (cmax == g_f) {
        h = 60 * (((b_f - r_f) / delta) + 2);
    } else {
        h = 60 * (((r_f - g_f) / delta) + 4);
    }
    if (h < 0) h += 360;

    // Cálculo de Saturation (Saturación)
    s = (cmax == 0) ? 0 : (delta / cmax);

    // Cálculo de Value (Brillo)
    v = cmax;
}

__global__ void skin_detection_kernel(uchar3* frame, unsigned char* mask, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        float h, s, v;
        
        bgrToHsv(frame[idx], h, s, v);
        
        // Rangos HSV para detección de piel (valores originales optimizados)
        bool is_skin = false;
        
        // Rango principal de piel
        if ((h >= 0 && h <= 20) || (h >= 340 && h <= 360)) {
            if (s >= 0.23f && s <= 0.68f && v >= 0.35f && v <= 0.95f) {
                is_skin = true;
            }
        }
        // Rango secundario para tonos más claros
        else if (h >= 20 && h <= 40) {
            if (s >= 0.15f && s <= 0.85f && v >= 0.40f && v <= 0.95f) {
                is_skin = true;
            }
        }
        
        mask[idx] = is_skin ? 255 : 0;
    }
}

// ==================================================================
// FUNCIÓN PRINCIPAL
// ==================================================================
int main() {
    // --- Estructuras para análisis de mano ---
    struct FingerTip {
        cv::Point position;
        float angle;
        int finger_id; // 0=pulgar, 1=índice, 2=medio, 3=anular, 4=meñique
        bool is_extended;
    };
    
    // --- Parámetros ajustables de detección de piel (valores optimizados) ---
    float h_min1 = 0.0f, h_max1 = 20.0f;      // Rango principal (rojos)
    float h_min2 = 20.0f, h_max2 = 40.0f;     // Rango secundario (naranjas/amarillos)
    float s_min = 0.23f, s_max = 0.68f;       // Saturación (valores originales mejorados)
    float v_min = 0.35f, v_max = 0.95f;       // Brillo (valores originales mejorados)
    
    std::cout << "Controles de calibración:" << std::endl;
    std::cout << "  q/a: Ajustar Hue min1 (" << h_min1 << ")" << std::endl;
    std::cout << "  w/s: Ajustar Hue max1 (" << h_max1 << ")" << std::endl;
    std::cout << "  e/d: Ajustar Hue min2 (" << h_min2 << ")" << std::endl;
    std::cout << "  r/f: Ajustar Hue max2 (" << h_max2 << ")" << std::endl;
    std::cout << "  t/g: Ajustar Sat min (" << s_min << ")" << std::endl;
    std::cout << "  y/h: Ajustar Sat max (" << s_max << ")" << std::endl;
    std::cout << "  u/j: Ajustar Val min (" << v_min << ")" << std::endl;
    std::cout << "  i/k: Ajustar Val max (" << v_max << ")" << std::endl;
    std::cout << "  ESC: Salir" << std::endl;

    // 1. Abrir la cámara web con configuraciones para PS3 Eye
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: No se pudo abrir la cámara web." << std::endl;
        return -1;
    }
    
    // Configuraciones específicas para PS3 Eye y cámaras de baja calidad
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);
    cap.set(cv::CAP_PROP_SATURATION, 0.6);  // Reducir saturación
    cap.set(cv::CAP_PROP_CONTRAST, 0.7);   // Ajustar contraste
    cap.set(cv::CAP_PROP_BRIGHTNESS, 0.5); // Ajustar brillo

    // Obtener dimensiones del fotograma
    int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    size_t frame_bytes = frame_width * frame_height * sizeof(uchar3);
    size_t mask_bytes = frame_width * frame_height * sizeof(unsigned char);

    // 2. Crear ventanas redimensionables
    cv::namedWindow("Webcam", cv::WINDOW_NORMAL);
    cv::namedWindow("Lienzo", cv::WINDOW_NORMAL);
    cv::namedWindow("Hand Mask", cv::WINDOW_NORMAL);
    
    // Redimensionar ventanas para que sean más grandes
    cv::resizeWindow("Webcam", 800, 600);
    cv::resizeWindow("Lienzo", 800, 600);
    cv::resizeWindow("Hand Mask", 400, 300);
    
    // Posicionar ventanas para que no se superpongan
    cv::moveWindow("Webcam", 50, 50);
    cv::moveWindow("Lienzo", 900, 50);
    cv::moveWindow("Hand Mask", 50, 700);

    // 3. Alojar memoria
    // Host (CPU)
    cv::Mat frame, mask(frame_height, frame_width, CV_8UC1);
    cv::Mat canvas = cv::Mat::zeros(frame_height, frame_width, CV_8UC3);
    std::vector<cv::Point> points; // Para guardar la trayectoria del dedo

    // Device (GPU)
    uchar3* d_frame;
    unsigned char* d_mask;
    CHECK_CUDA(cudaMalloc(&d_frame, frame_bytes));
    CHECK_CUDA(cudaMalloc(&d_mask, mask_bytes));

    std::cout << "Iniciando captura. Muestra tu MANO a la cámara para dibujar con el dedo índice." << std::endl;
    std::cout << "Presiona 'q' o ESC para salir." << std::endl;

    // 4. Bucle principal
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Voltear la imagen para efecto espejo
        cv::flip(frame, frame, 1);

        // --- Fase de CUDA: Detección de piel en la GPU ---
        CHECK_CUDA(cudaMemcpy(d_frame, frame.data, frame_bytes, cudaMemcpyHostToDevice));

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((frame_width + 15) / 16, (frame_height + 15) / 16);
        skin_detection_kernel<<<numBlocks, threadsPerBlock>>>(d_frame, d_mask, frame_width, frame_height);
        
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(mask.data, d_mask, mask_bytes, cudaMemcpyDeviceToHost));

        // --- Fase de CPU: Análisis sofisticado de mano ---
        // Operaciones morfológicas para limpiar la máscara
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
        
        // Aplicar filtro gaussiano para suavizar
        cv::GaussianBlur(mask, mask, cv::Size(5, 5), 0);
        
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        cv::Point index_finger_tip(-1, -1);
        
        // Encontrar el contorno más grande (la mano)
        double max_area = 1000; // Área mínima para considerar como mano
        int hand_contour_idx = -1;
        for (int i = 0; i < contours.size(); i++) {
            double area = cv::contourArea(contours[i]);
            if (area > max_area) {
                max_area = area;
                hand_contour_idx = i;
            }
        }
        
        if (hand_contour_idx != -1 && max_area > 3000) {
            std::vector<cv::Point>& hand_contour = contours[hand_contour_idx];
            
            // Calcular convex hull
            std::vector<cv::Point> hull_points;
            std::vector<int> hull_indices;
            cv::convexHull(hand_contour, hull_points, false);
            cv::convexHull(hand_contour, hull_indices, false);
            
            // Calcular convexity defects
            std::vector<cv::Vec4i> defects;
            if (hull_indices.size() > 3) {
                cv::convexityDefects(hand_contour, hull_indices, defects);
            }
            
            // Encontrar fingertips usando defects
            std::vector<FingerTip> finger_tips;
            
            // Buscar puntos altos del convex hull (candidatos a dedos)
            cv::Moments hand_moments = cv::moments(hand_contour);
            cv::Point hand_center(hand_moments.m10 / hand_moments.m00, hand_moments.m01 / hand_moments.m00);
            
            for (const cv::Point& hull_point : hull_points) {
                double dist_to_center = cv::norm(hull_point - hand_center);
                
                // Verificar si este punto está suficientemente lejos del centro
                if (dist_to_center > 80) {
                    FingerTip tip;
                    tip.position = hull_point;
                    tip.angle = atan2(hull_point.y - hand_center.y, hull_point.x - hand_center.x) * 180.0 / CV_PI;
                    tip.is_extended = true;
                    
                    // Verificar que no está muy cerca de otros dedos ya detectados
                    bool too_close = false;
                    for (const FingerTip& existing : finger_tips) {
                        if (cv::norm(tip.position - existing.position) < 50) {
                            too_close = true;
                            break;
                        }
                    }
                    
                    if (!too_close && finger_tips.size() < 5) {
                        finger_tips.push_back(tip);
                    }
                }
            }
            
            // Identificar el dedo índice (normalmente el más alto y a la derecha del pulgar)
            if (!finger_tips.empty()) {
                // Ordenar dedos por posición Y (más arriba = menor Y)
                std::sort(finger_tips.begin(), finger_tips.end(), 
                         [](const FingerTip& a, const FingerTip& b) {
                             return a.position.y < b.position.y;
                         });
                
                // El dedo más alto suele ser el índice o medio
                if (finger_tips.size() >= 2) {
                    // Entre los dos dedos más altos, elegir el que esté más a la derecha
                    if (finger_tips[0].position.x > finger_tips[1].position.x) {
                        index_finger_tip = finger_tips[0].position;
                    } else {
                        index_finger_tip = finger_tips[1].position;
                    }
                } else {
                    index_finger_tip = finger_tips[0].position;
                }
                
                // Dibujar todos los dedos detectados para debug
                for (size_t i = 0; i < finger_tips.size(); i++) {
                    cv::circle(frame, finger_tips[i].position, 8, 
                              i == 0 ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), -1);
                }
            }
            
            // Dibujar contorno y convex hull para debug
            cv::drawContours(frame, contours, hand_contour_idx, cv::Scalar(255, 0, 0), 2);
            cv::polylines(frame, hull_points, true, cv::Scalar(0, 255, 255), 2);
            cv::circle(frame, hand_center, 5, cv::Scalar(255, 255, 0), -1);
        }
        
        // Actualizar puntos de dibujo basado en el dedo índice
        if (index_finger_tip.x > 0 && index_finger_tip.y > 0) {
            points.push_back(index_finger_tip);
            
            // Limitar el número de puntos para evitar lag
            if (points.size() > 1000) {
                points.erase(points.begin(), points.begin() + 100);
            }
        } else {
            // Si no se detecta el dedo, mantener los puntos actuales por un momento
            // pero eventualmente limpiar para empezar un nuevo trazo
            static int frames_without_detection = 0;
            frames_without_detection++;
            if (frames_without_detection > 30) { // ~1 segundo a 30 FPS
                points.clear();
                frames_without_detection = 0;
            }
        }

        // Dibujar la trayectoria en el lienzo con líneas suavizadas
        if (points.size() > 1) {
            for (size_t i = 1; i < points.size(); ++i) {
                // Calcular grosor basado en la velocidad del movimiento
                double distance = cv::norm(points[i] - points[i-1]);
                int thickness = std::max(2, std::min(8, static_cast<int>(20 - distance/3)));
                
                cv::line(canvas, points[i-1], points[i], cv::Scalar(0, 255, 255), thickness); // Línea amarilla
            }
        }
        
        // Combinar la cámara con el lienzo para una vista final
        cv::Mat final_view;
        cv::add(frame, canvas, final_view);

        // 5. Mostrar las imágenes
        cv::imshow("Webcam", final_view); // Muestra la cámara con el dibujo superpuesto
        cv::imshow("Lienzo", canvas);   // Muestra solo el dibujo
        cv::imshow("Hand Mask", mask);  // Muestra la máscara de detección de piel

        // Controles de calibración y salida
        char key = (char)cv::waitKey(1);
        bool params_changed = false;
        
        switch(key) {
            // Hue range 1
            case 'q': h_min1 = std::max(0.0f, h_min1 - 2.0f); params_changed = true; break;
            case 'a': h_min1 = std::min(360.0f, h_min1 + 2.0f); params_changed = true; break;
            case 'w': h_max1 = std::max(0.0f, h_max1 - 2.0f); params_changed = true; break;
            case 's': h_max1 = std::min(360.0f, h_max1 + 2.0f); params_changed = true; break;
            
            // Hue range 2
            case 'e': h_min2 = std::max(0.0f, h_min2 - 2.0f); params_changed = true; break;
            case 'd': h_min2 = std::min(360.0f, h_min2 + 2.0f); params_changed = true; break;
            case 'r': h_max2 = std::max(0.0f, h_max2 - 2.0f); params_changed = true; break;
            case 'f': h_max2 = std::min(360.0f, h_max2 + 2.0f); params_changed = true; break;
            
            // Saturation
            case 't': s_min = std::max(0.0f, s_min - 0.05f); params_changed = true; break;
            case 'g': s_min = std::min(1.0f, s_min + 0.05f); params_changed = true; break;
            case 'y': s_max = std::max(0.0f, s_max - 0.05f); params_changed = true; break;
            case 'h': s_max = std::min(1.0f, s_max + 0.05f); params_changed = true; break;
            
            // Value/Brightness
            case 'u': v_min = std::max(0.0f, v_min - 0.05f); params_changed = true; break;
            case 'j': v_min = std::min(1.0f, v_min + 0.05f); params_changed = true; break;
            case 'i': v_max = std::max(0.0f, v_max - 0.05f); params_changed = true; break;
            case 'k': v_max = std::min(1.0f, v_max + 0.05f); params_changed = true; break;
            
            case 27: // ESC
                break;
        }
        
        if (params_changed) {
            std::cout << "H1:[" << h_min1 << "-" << h_max1 << "] H2:[" << h_min2 << "-" << h_max2 << "] S:[" << s_min << "-" << s_max << "] V:[" << v_min << "-" << v_max << "]" << std::endl;
        }
        
        if (key == 27) break; // ESC para salir
    }

    // 6. Limpieza final
    cap.release();
    cv::destroyAllWindows();
    CHECK_CUDA(cudaFree(d_frame));
    CHECK_CUDA(cudaFree(d_mask));

    return 0;
}