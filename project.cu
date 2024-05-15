#include <cuda_runtime.h>
#include <iostream>

__global__ void initColors(int *array, int startPurple, int startRed, int startGreen, int totalColors) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < startPurple) {
        // RGB voor paars: (128, 0, 128)
        array[index * 3] = 128;     // R
        array[index * 3 + 1] = 0;   // G
        array[index * 3 + 2] = 128; // B
    } else if (index >= startPurple && index < startRed) {
        // RGB voor rood: (255, 0, 0)
        array[index * 3] = 255;     // R
        array[index * 3 + 1] = 0;   // G
        array[index * 3 + 2] = 0;   // B
    } else if (index >= startRed && index < totalColors) {
        // RGB voor groen: (0, 255, 0)
        array[index * 3] = 0;       // R
        array[index * 3 + 1] = 255; // G
        array[index * 3 + 2] = 0;   // B
    }
}

int main() {
    // Initialize the first and second arrays
    int numColors1 = 25; // 10 paars + 15 rood
    int numColors2 = 30; // 2 paars + 20 rood + 8 groen

    int *dev_array1, *dev_array2, *dev_combined;
    int *host_array1 = new int[numColors1 * 3];
    int *host_array2 = new int[numColors2 * 3];
    int numCombinedColors = 38; // 10 paars + 20 rood + 8 groen
    int *host_combined = new int[numCombinedColors * 3];

    cudaMalloc((void**)&dev_array1, numColors1 * 3 * sizeof(int));
    cudaMalloc((void**)&dev_array2, numColors2 * 3 * sizeof(int));
    cudaMalloc((void**)&dev_combined, numCombinedColors * 3 * sizeof(int));

    int threadsPerBlock = 256;
    int blocksPerGrid1 = (numColors1 + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGrid2 = (numColors2 + threadsPerBlock - 1) / threadsPerBlock;
    int blocksPerGridCombined = (numCombinedColors + threadsPerBlock - 1) / threadsPerBlock;

    // Fill the first and second arrays
    initColors<<<blocksPerGrid1, threadsPerBlock>>>(dev_array1, 10, 25, 25, numColors1);
    initColors<<<blocksPerGrid2, threadsPerBlock>>>(dev_array2, 2, 22, 30, numColors2);

    // Copy data back to host
    cudaMemcpy(host_array1, dev_array1, numColors1 * 3 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_array2, dev_array2, numColors2 * 3 * sizeof(int), cudaMemcpyDeviceToHost);

    // Now merge the data into the combined array
    initColors<<<blocksPerGridCombined, threadsPerBlock>>>(dev_combined, 10, 30, 38, numCombinedColors);
    cudaMemcpy(host_combined, dev_combined, numCombinedColors * 3 * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the combined array
    std::cout << "Combined array contents: ";
    for (int i = 0; i < numCombinedColors; ++i) {
        int r = host_combined[i * 3];
        int g = host_combined[i * 3 + 1];
        int b = host_combined[i * 3 + 2];
        std::cout << "(" << r << "," << g << "," << b << ")";
        if (i < numCombinedColors - 1) std::cout << ", ";
    }
    std::cout << std::endl;

    // Free memory
    cudaFree(dev_array1);
    cudaFree(dev_array2);
    cudaFree(dev_combined);
    delete[] host_array1;
    delete[] host_array2;
    delete[] host_combined;

    return 0;
}