
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <fstream>
#include <algorithm>
#include <conio.h>
#include <opencv2/opencv.hpp>

#include "control_menu.h"
#include "my_little_json_parser.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

cudaError_t applyFilter(int width, int height, std::vector<float> kernel, int kernel_size, cv::Mat image, cv::Mat* output_image);

__global__ void convolveKernel(const uchar* image, uchar* result, const float* kernel, const int imageWidth, const int imageHeight, const int kernelSize);

std::vector<float> generateGaussianKernel(int kernelSize, float sigma);
std::vector<float> generateSobelOperator(bool typeX=true);

void operateOnJson(const std::string& jsonPath, bool& gb, bool& so, float& sigma, std::vector<float>& kernel, int& kernel_size, std::string& img_path, bool& saveOutputImage, std::string& outputImagePath);
void saveImageToFile(cv::Mat image, const std::string imagePath);

void menu(bool& gb, bool& so, float& sigma, std::vector<float>& kernel, int& kernel_size, std::string& img_path, bool& saveOutputImage, std::string& outputImagePath);


int main(int argc, char* argv[])
{
    bool gb = false;
    bool so = false;
    float sigma;
    std::vector<float> kernel;
    int kernel_size = 3;
    std::string img_path;

    bool saveOutputImage = false;
    std::string outputImagePath = "./output.png";

    if (argc > 1)
    {
       std::string jsonPath = argv[1];
        operateOnJson(jsonPath, gb, so, sigma, kernel, kernel_size, img_path, saveOutputImage, outputImagePath);
    }
    else
    {
        menu(gb, so, sigma, kernel, kernel_size, img_path, saveOutputImage, outputImagePath);
    }
    
    if (gb)
    {
       kernel = generateGaussianKernel(kernel_size, sigma);
    }
    if (so)
    {
        // detection        
    }
    cv::Mat iamge = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
    // TODO 3 channel images
    if (iamge.empty())
    {
        std::cout << "Could not read the image: " << img_path << std::endl;
        return 1;
    }
    cv::Size s = iamge.size();
    const int width = s.width;
    const int height = s.height;
    cv::Mat out_image = cv::Mat::zeros(height, width, CV_8UC1);

    std::cout << "before rodeo\n";

    // apply filter
    cudaError_t cudaStatus = applyFilter(width, height,
        kernel, kernel_size,
        iamge, &out_image);
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) 
    {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    std::cout << "displaying image" << std::endl;
    cv::imshow("Display window - out", out_image);
    int k = cv::waitKey(0);

    if (saveOutputImage)
    {
        saveImageToFile(out_image, outputImagePath);
    }
    

    return 0;
}

cudaError_t applyFilter(int width, int height,
    std::vector<float> kernel, int kernel_size,
    cv::Mat image, cv::Mat* output_image)
{
    cudaError_t cudaStatus;
    uchar* dev_in_m = 0;
    uchar* dev_out_m = 0;
    float* dev_kernel_mat = 0;

    const size_t size = width * height * sizeof(uchar);

    std::cout << "rodeo \n";

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_in_m, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_out_m, size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_kernel_mat, kernel_size * kernel_size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    std::cout << "rodeo allocated \n";

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_in_m, image.data, size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_kernel_mat, kernel.data(), kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    std::cout << "rodeo copied to device \n";

    // Launch a kernel on the GPU with one thread for each element.
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // start kernel CUDA
    convolveKernel << <blocksPerGrid, threadsPerBlock >> > (dev_in_m, dev_out_m, dev_kernel_mat, width, height, kernel_size);

    std::cout << "rodeo kernel finished \n";

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(output_image->data, dev_out_m, size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_in_m);
    cudaFree(dev_out_m);
    cudaFree(dev_kernel_mat);

    return cudaStatus;
}

__global__ void convolveKernel(const uchar* image, uchar* result, const float* kernel, const int imageWidth, const int imageHeight, const int kernelSize)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int radius = kernelSize / 2;
    float value = 0.0;

    // Checking whether the coordinates are within the image boundaries
    if (x < imageWidth && y < imageHeight) 
    {
        for (int ky = -radius; ky <= radius; ++ky) 
        {
            for (int kx = -radius; kx <= radius; ++kx) 
            {
                int px = x + kx;
                int py = y + ky;

                if (px >= 0 && px < imageWidth && py >= 0 && py < imageHeight) {
                    value += image[py * imageWidth + px] * kernel[(ky + radius) * kernelSize + (kx + radius)];
                }
            }
        }
        result[y * imageWidth + x] = value;
    }
}

std::vector<float> generateGaussianKernel(int kernelSize, float sigma) 
{
    std::vector<float> kernel(kernelSize * kernelSize);
    int radius = kernelSize / 2;
    float sum = 0.0f;

    for (int y = -radius; y <= radius; ++y) 
    {
        for (int x = -radius; x <= radius; ++x) 
        {
            float value = expf(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * M_PI * sigma * sigma);
            kernel[(y + radius) * kernelSize + (x + radius)] = value;
            sum += value;
        }
    }

    for (auto& k : kernel) 
    {
        k /= sum;
    }

    return kernel;
}

std::vector<float> generateSobelOperator(bool typeX)
{
    std::vector<float> sobelX = 
    {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };

    std::vector<float> sobelY = 
    {
        -1, -2, -1,
         0,  0,  0,
         1,  2,  1
    };
    if (typeX)
    {
        return sobelX;
    }
    else
    {
        return sobelY;
    }
}

void operateOnJson(const std::string& jsonPath, bool& gb, bool& so, float& sigma, std::vector<float>& kernel, int& kernel_size, std::string& img_path, bool& saveOutputImage, std::string& outputImagePath)
{
    auto parsedJson = getJsonContent(jsonPath);
    if (operationTypeNCorrectFormat(parsedJson))
    {
        gb = true;
        getGaussianDataFromJSON(parsedJson, sigma, kernel_size);
        kernel = generateGaussianKernel(kernel_size, sigma);
    }
    else
    {
        so = true;
        if (getSobelDataFromJSON(parsedJson))
        {
            kernel = generateSobelOperator(true);
        }
        else
        {
            kernel = generateSobelOperator(false);
        }

    }

    img_path = parsedJson.at("image path");
    saveOutputImage = getSaveDataFromJSON(parsedJson, outputImagePath);
}

void menu(bool& gb, bool& so, float& sigma, std::vector<float>& kernel, int& kernel_size, std::string& img_path, bool& saveOutputImage, std::string& outputImagePath)
{
    // Main menu options:
    std::vector<std::string> options = 
    {
        "Option 1: Gaussian Blur",
        "Option 2: Sobel Operators",
        "Option 3: Get JSON",
        "Option 4: Force exit"
    };

    int current = 0;
    while (true) 
    {
        clearScreen();
        std::cout << "Select option (W/S - move cursor, Enter - select):\n\n";
        for (size_t i = 0; i < options.size(); ++i) 
        {
            if (i == current) 
            {
                std::cout << " > " << options[i] << "\n";
            }
            else 
            {
                std::cout << "   " << options[i] << "\n";
            }
        }

        char input = _getch();
        if (input == 'w' || input == 'W') 
        {
            current = (current > 0) ? current - 1 : options.size() - 1;
        }
        else if (input == 's' || input == 'S') 
        {
            current = (current < options.size() - 1) ? current + 1 : 0;
        }
        else if (input == '\r') // Enter
        {
            if (current == 0) 
            {
                float selectedSigmaValue = handleFloatSelection();
                int selectedKernelSize = kernelSizeMenu();
                saveOutputImage = saveImageMenu();
                kernel_size = selectedKernelSize;
                sigma = selectedSigmaValue;
                gb = true;

                std::cout << "Path to IMG...\n";
                std::cin >> img_path;
                clearScreen();
                break;
            }
            else if (current == 1) 
            {
                int selected_option = xyOperatorMenu();
                saveOutputImage = saveImageMenu();
                if (selected_option == 0)
                {
                    so = true;
                    kernel = generateSobelOperator();
                }
                else if (selected_option == 1)
                {
                    so = true;
                    kernel = generateSobelOperator(false);
                }
                std::cout << "Path to IMG...\n";
                std::cin >> img_path;
                break;
            }
            else if (current == 2)
            {
                clearScreen();

                std::cout << "Path to data...\n";

                std::string jsonPath;
                std::cin >> jsonPath;

                operateOnJson(jsonPath, gb, so, sigma, kernel, kernel_size, img_path, saveOutputImage, outputImagePath);
                break;
            }
            else if (current == 3) 
            {
                clearScreen();
                std::cout << "Fast exit!\n";
                
                exit(0);
            }
        }
    }
}

void saveImageToFile(cv::Mat image, const std::string imagePath)
{
    bool isSaved = cv::imwrite(imagePath, image);
    if (isSaved) 
    {
        std::cout << "Picture saved in path : " << imagePath << std::endl;
    }
    else 
    {
        std::cerr << "Failed to save picture" << std::endl;
        exit(-2);
    }
}

// TODO: docs