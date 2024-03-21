#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 96 // Size of the array


__global__ void GPUatomicmax(int *input, int *max2, int size){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // total index = number of the block we're in now x number of threads + number of the thread in the current block
    int stride = blockDim.x*gridDim.x;

    int max = 0;

    for(int i = tid; i<size; i+=stride){
        atomicMax(&max,  input[i]);
    }

    atomicMax(max2, max);
}

__global__ void GPUreductionmax(int *input, int *max3, int size){
    unsigned int tid = threadIdx.x;
    __syncthreads();

    for (int s = 1; s < size/2 ; s++) {
        if (tid < s) {
            input[s*tid] = max(input[s*tid], input[s*(tid + 1)]);
        }
        __syncthreads();
    }
     *max3 = input[0];
     //relatie tussen de verschillende -> bij de eerste is *2 en +1, volgende is *4 en +2 , volgende ...
     
}

// CPU code
int main() {
    // Creation of the input array of size N (defined on line 5) filled with random integers
    int inputArray[N];
    for (int i=0; i<N; i++){
        inputArray[i] = rand();
    }

    // Allocating a memoryplace for the input array, putting it there and moving it to the GPU memory
    int *d_inputArray;
    cudaMalloc(&d_inputArray, N * sizeof(int)); 
    cudaMemcpy(d_inputArray, inputArray, N * sizeof(int), cudaMemcpyHostToDevice); // cpu memory to GPU memory

    // Define grid and block size
    int blockSize = 256;  // aantal threads in block
    int gridSize = (N + blockSize - 1) / blockSize;

    // Nested for-loop on CPU
    int maxcpu = 0;
    const auto start = std::chrono::steady_clock::now();

    for ( int i = 0; i<N; i++) {
        for(int j=i+1; j<N; j++) {
            if(inputArray[i]>inputArray[j] & inputArray[i]>maxcpu){
                maxcpu = inputArray[i];
            }
        }
    }

    const auto end = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed_seconds_cpu{end - start};

    //GPU atomic
    int maxgpu1 = 0;
    int *d_maxgpu1;
    cudaMalloc(&d_maxgpu1, sizeof(int)); 
    const auto startgpu1 = std::chrono::high_resolution_clock::now();

    GPUatomicmax<<<gridSize, blockSize>>>(d_inputArray, d_maxgpu1, N);  // geef pointers naar arrays mee
    cudaMemcpy(&maxgpu1, d_maxgpu1, sizeof(int), cudaMemcpyDeviceToHost); // cpu memory to GPU memory

    cudaFree(d_maxgpu1);
    
    const auto endgpu1 = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed_seconds_gpu{endgpu1 - startgpu1};

    //GPU reduction
    int maxgpu2 = 0;
    int *d_maxgpu2;
    cudaMalloc(&d_maxgpu2, sizeof(int)); 
    const auto startgpu2 = std::chrono::high_resolution_clock::now();

    GPUreductionmax<<<gridSize, blockSize>>>(d_inputArray, d_maxgpu2, N);

    cudaDeviceSynchronize();

    cudaMemcpy(&maxgpu2, d_maxgpu2, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(d_inputArray);
    cudaFree(d_maxgpu2);

    const auto endgpu2 = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed_seconds_gpu2{endgpu2 - startgpu2};

    // Print the result (standaard cpp stuff)
    std::cout << "Input Array: ";
    for (int i = 0; i < N; i++) {
        std::cout << inputArray[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "CPU" << std::endl;
    std::cout << "Max: " << maxcpu << std::endl;
    std::cout << elapsed_seconds_cpu.count() << "\n";

    std::cout << "GPU atomic" << std::endl;
    std::cout << "Max: " << maxgpu1 << std::endl;
    std::cout << elapsed_seconds_gpu.count() << "\n";

    std::cout << "GPU reduction" << std::endl;
    std::cout << "Max: " << maxgpu2 << std::endl;

    std::cout << elapsed_seconds_gpu2.count() << "\n";

    return 0;
}