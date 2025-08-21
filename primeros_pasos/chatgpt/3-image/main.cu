#include <iostream>
#include <opencv4/opencv2/opencv.hpp>

// Función para manejar errores de CUDA
#define CUDA_CHECK(err) { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        std::cerr << "Error CUDA: " << cudaGetErrorString(err_) << " en " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

// ✨ Bandera para controlar la actualización. La hacemos 'volatile' para asegurar
// que el compilador no la optimice de forma inesperada.
volatile bool g_needs_update = true;

// ✨ Callback que se llamará CADA VEZ que un slider se mueva.
// Su única función es activar la bandera.
void on_trackbar_change(int, void*){
    g_needs_update = true;
}

__global__ void invertColors(unsigned char* d_src, unsigned char* d_dst, int width, int height, int brightness, bool invert_value, int r_value, int g_value, int b_value, bool sepia, bool grayscale){
    
    int t_x = blockIdx.x * blockDim.x + threadIdx.x;
    int t_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (t_x >= width || t_y >= height) return;

    int t_id = t_y * width + t_x;
    int byte = t_id * 3;

    int b = d_src[byte];
    int g = d_src[byte + 1];
    int r = d_src[byte + 2];

    r = r + brightness;
    g = g + brightness;
    b = b + brightness;

    if(invert_value){
        r = 255 - r;
        g = 255 - g;
        b = 255 - b;
    }

    if(grayscale){
        int gray = 0.299f * r + 0.587f * g + 0.114f * b;
        r = gray;
        g = gray;
        b = gray;
    }
    if(sepia){
        int tr = 0.393f * r + 0.769f * g + 0.189f * b;
        int tg = 0.349f * r + 0.686f * g + 0.168f * b;
        int tb = 0.272f * r + 0.534f * g + 0.131f * b;

        r = tr;
        g = tg;
        b = tb;
    }

    d_dst[byte + 2] = (unsigned char)max(0, min(255, r * r_value));
    d_dst[byte + 1] = (unsigned char)max(0, min(255, g * g_value));
    d_dst[byte]     = (unsigned char)max(0, min(255, b * b_value));
}

int main(int argc, char** argv){
    if(argc < 2){
        std::cerr << "El nombre del archivo debe ser el segundo parametro." << std::endl;
        return 1;
    }

    const char* filename = argv[1];
    cv::Mat src = cv::imread(filename);
    int w = src.cols;
    int h = src.rows;



    cv::Mat image_cpu = src(cv::Rect(0, 0, w, h));

    if (image_cpu.empty()) {
        std::cerr << "Error al cargar la imagen." << std::endl;
        return 1;
    }
    if (image_cpu.channels() != 3) {
        std::cerr << "La imagen debe ser a color (3 canales)." << std::endl;
        return 1;
    }

    int width = image_cpu.cols;
    int height = image_cpu.rows;

    //asignamos memoria para imagen de origen
    unsigned char* d_src = nullptr;
    size_t img_size = width * height * image_cpu.channels() * sizeof(unsigned char);
    CUDA_CHECK(cudaMalloc(&d_src, img_size));

    //memoria para imagen de destino
    unsigned char* d_dst = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dst, img_size));

    //copiamos imagen desde CPU a GPU
    CUDA_CHECK(cudaMemcpy(d_src, image_cpu.data, img_size, cudaMemcpyHostToDevice));

    //configuramos kernel
    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // invertColors<<<numBlocks, threadsPerBlock>>>(d_src, d_dst, width, height);
    // CUDA_CHECK(cudaGetLastError());

    //copiar de GPU a CPU
    cv::Mat result_cpu(height, width, CV_8UC3);
    // CUDA_CHECK(cudaMemcpy(result_cpu.data, d_dst, img_size, cudaMemcpyDeviceToHost));



    // mostrar
    // cv::namedWindow("Imagen Original", cv::WINDOW_NORMAL);
    cv::namedWindow("Resultado", cv::WINDOW_NORMAL);

    // cv::imshow("Imagen Original", image_cpu);
    // cv::imshow("Resultado", result_cpu);
        
    int brightness_slider = 255; // Valor inicial 255, para que el brillo comience en 0 (-255)
    int invert_slider = 0;       // 0 para no invertir, 1 para invertir
    int r_slider = 1;
    int g_slider = 1;
    int b_slider = 1;
    int sepia_slider = 0;
    int grayscale_slider = 0;

    cv::createTrackbar("Brillo (-255 a +255)", "Resultado", &brightness_slider, 510, on_trackbar_change);
    cv::createTrackbar("Invertir Colores (0/1)", "Resultado", &invert_slider, 1, on_trackbar_change);
    cv::createTrackbar("R", "Resultado", &r_slider, 10, on_trackbar_change);
    cv::createTrackbar("G", "Resultado", &g_slider, 10, on_trackbar_change);
    cv::createTrackbar("B", "Resultado", &b_slider, 10, on_trackbar_change);
    cv::createTrackbar("Sepia (0/1)", "Resultado", &sepia_slider, 1, on_trackbar_change);
    cv::createTrackbar("Grayscale (0/1)", "Resultado", &grayscale_slider, 1, on_trackbar_change);


    while(true){
        if(g_needs_update){
            int brightness_value = brightness_slider - 255;
            bool invert_value = (invert_slider == 1);
            int r_value = r_slider;
            int g_value = g_slider;
            int b_value = b_slider;
            bool sepia_value = (sepia_slider == 1);
            bool grayscale_value = (grayscale_slider == 1);
    
            invertColors<<<numBlocks, threadsPerBlock>>>(d_src, d_dst, width, height, brightness_value, invert_value, r_value, g_value, b_value, sepia_value, grayscale_value);
            CUDA_CHECK(cudaGetLastError());
    
            CUDA_CHECK(cudaMemcpy(result_cpu.data, d_dst, img_size, cudaMemcpyDeviceToHost));

            cv::imshow("Resultado", result_cpu);

            g_needs_update = false;
        }

        if(cv::waitKey(1) == 27) {
            break;
        }
    }


    // liberar memoria de la GPU
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));

    cv::destroyAllWindows();

    return 0;
}