
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <bitset>

using namespace std;

#define width 384
#define height 288
#define gridSize 1
#define blockSize 96

uint8_t* get_image_array() {
    FILE* imageFile = fopen("./1x2a-night-000.png", "rb");
    if (imageFile == NULL) {
        perror("ERROR: Cannot open output file");
        exit(EXIT_FAILURE);
    }
    uint8_t* image_array = (uint8_t*)malloc(width * height * sizeof(uint8_t));
    fread(image_array, sizeof(uint8_t), width * height, imageFile);
    fclose(imageFile);
    return image_array;
}

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__global__ void testKernel(uint8_t* image_array, int* output_array) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < width * height; i += stride) {
        output_array[i * 35] = 5; // Set a known value
    }
}

int main(void) {
    uint8_t* image_array = get_image_array();
    int* output_array = (int*)malloc(width * height * 35 * sizeof(int));

    uint8_t* d_image_array;
    cudaError_t err;
    err = cudaMalloc(&d_image_array, width * height * sizeof(uint8_t));
    checkCudaError(err, "Failed to allocate device memory for image array");

    err = cudaMemcpy(d_image_array, image_array, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);
    checkCudaError(err, "Failed to copy image array to device");

    int* d_output_array;
    err = cudaMalloc(&d_output_array, width * height * 35 * sizeof(int));
    checkCudaError(err, "Failed to allocate device memory for output array");

    const auto start = std::chrono::high_resolution_clock::now();
    testKernel<<<gridSize, blockSize>>>(d_image_array, d_output_array);
    err = cudaGetLastError();
    checkCudaError(err, "Kernel launch failed");

    err = cudaMemcpy(output_array, d_output_array, width * height * 35 * sizeof(int), cudaMemcpyDeviceToHost);
    checkCudaError(err, "Failed to copy output array from device");

    cudaDeviceSynchronize();
    const auto end = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < width * height; i++) {
        std::cout << output_array[i * 35] << " " << i << std::endl;
    }

    cudaFree(d_image_array);
    cudaFree(d_output_array);
    free(image_array);
    free(output_array);

    return 0;
}
