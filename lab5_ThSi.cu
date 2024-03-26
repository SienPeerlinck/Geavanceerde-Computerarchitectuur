#include <cstdint>      // Data types
#include <iostream>     // File operations
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>
// #define N 96 // Size of the arrays



__global__ void reductionMax(int* data, int data_size, int* result) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < data_size) {
        for (int stride = 1; stride < data_size; stride *= 2) {
            int index = 2 * stride * idx;
            if (index < data_size && index+stride < data_size) {
                int lhs = data[index];
                int rhs = data[index + stride];
                data[index] = (lhs < rhs) ? rhs : lhs;
            }
            __syncthreads();
        }
        if (idx == 0) {
            *result = data[0];
        }
    }
}

__global__ void minGPUReduction(int* data, int data_size, int* result) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < data_size) {
        for (int stride = 1; stride < data_size; stride *= 2) {
            int index = 2 * stride * idx;
            if (index < data_size && index+stride < data_size) {
                int lhs = data[index];
                int rhs = data[index + stride];
                data[index] = (lhs > rhs) ? rhs : lhs;
            }
            __syncthreads();
        }
        if (idx == 0) {
            *result = data[0];
        }
    }
}

// __global__ void reductionSum(int* in) {
//     int step = 1;
//     unsigned int i = (blockIdx.x * blockDim.x + threadIdx.x);
//     while(i % step == 0 && step < N) {
//         int k = 0;
//         while(i + k * gridDim.x * blockDim.x + step < N) {
//             int index = i + k * gridDim.x * blockDim.x;
//             in[2*index] += in[2*index + step];
//             k++;
//         }
//         step *= 2;
//         __syncthreads();
//     }
// }

__global__ void reductionSum(int* data, int data_size, int* result) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < data_size) {
        for (int stride = 1; stride < data_size; stride *= 2) {
            int index = 2 * stride * idx;
            if (index < data_size && index+stride < data_size) {
                int lhs = data[index];
                int rhs = data[index + stride];
                data[index] = lhs + rhs;
            }
            __syncthreads();
        }
        if (idx == 0) {
            *result = data[0];
        }
    }
}

__global__ void reductionMultiply(int* data, int data_size, int* result) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < data_size) {
        for (int stride = 1; stride < data_size; stride *= 2) {
            int index = 2 * stride * idx;
            if (index < data_size && index+stride < data_size) {
                int lhs = data[index];
                int rhs = data[index + stride];
                data[index] = lhs * rhs;
            }
            __syncthreads();
        }
        if (idx == 0) {
            *result = data[0];
        }
    }
}

int* array_seed(int N){
        unsigned int seed = time(NULL);
    int* inputArray = new int[N];
    for (int i=0; i<N; i++){
        srand(seed + i);
        inputArray[i] = (rand()%20)+1;
        // std::cout << inputArray[i] << ' ';
    }
    return inputArray;
}



// CPU code
int main() {
    // int inputArray[N];
    // for (int i=0; i<N; i++){
    //     inputArray[i] = (rand()%20)+1;
    //     std::cout << "Max: " << inputArray[i] << std::endl;
    // }


   


//GPU reduction
    int resultmax = 0;
    int resultmin = 0;
    int resultsum = 0;
    int resultmultiply = 0;
    int *d_resultmax;
    int *d_resultmin;
    int *d_resultsum;
    int *d_resultmultiply;
    cudaMalloc(&d_resultmax, sizeof(int));
    cudaMalloc(&d_resultmin, sizeof(int));
    cudaMalloc(&d_resultsum, sizeof(int));
    cudaMalloc(&d_resultmultiply, sizeof(int));

    cudaStream_t streams[4];

    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);
    cudaStreamCreate(&streams[2]);
    cudaStreamCreate(&streams[3]);
    std::ofstream outputFile("timing_asynchronous_cpu_vs_gpu.csv"); // for the measurements
        if (!outputFile.is_open()) {
            std::cerr << "Error opening file!" << std::endl;
            return 1;
        }
    for (int k = 0; k < 999; k++){

    outputFile << "blocksize,elapsed_time" << std::endl;

    int N = k;

    // Define grid and block size
    int blockSize = N;  // aantal threads in block
    int gridSize = 1;

    for (int j = 0; j < 20; j++){

    // const auto start = std::chrono::high_resolution_clock::now();

    // kerneleverything<<<gridSize, blockSize>>>(kernel, d_inputArray, d_result, N);
    // int max_arr = array_seed();
    
    const auto start_cpu1 = std::chrono::high_resolution_clock::now();
    int *maxarr = array_seed(N);
    int *d_inputArrayreduction;
    const auto end_cpu1 = std::chrono::high_resolution_clock::now();

    const auto start_gpu1 = std::chrono::high_resolution_clock::now();
    cudaMalloc(&d_inputArrayreduction, N * sizeof(int)); 
    cudaMemcpyAsync(d_inputArrayreduction, maxarr, N * sizeof(int), cudaMemcpyHostToDevice, streams[0]); // cpu memory to GPU memory

    reductionMax<<<gridSize,blockSize>>>(d_inputArrayreduction,N,d_resultmax);
    cudaMemcpyAsync(&resultmax,d_resultmax, sizeof(int),cudaMemcpyDeviceToHost, streams[0]);
    const auto end_gpu1 = std::chrono::high_resolution_clock::now();

    const auto start_cpu2 = std::chrono::high_resolution_clock::now();
    int *minarr = array_seed(N);
    int *d_inputArraymin;
    const auto end_cpu2 = std::chrono::high_resolution_clock::now();

    const auto start_gpu2 = std::chrono::high_resolution_clock::now();
    cudaMalloc(&d_inputArraymin, N * sizeof(int)); 
    cudaMemcpyAsync(d_inputArraymin, minarr, N * sizeof(int), cudaMemcpyHostToDevice, streams[1]); // cpu memory to GPU memory

    minGPUReduction<<<gridSize,blockSize>>>(d_inputArraymin,N,d_resultmin);
    cudaMemcpyAsync(&resultmin,d_resultmin, sizeof(int),cudaMemcpyDeviceToHost, streams[1]);
    const auto end_gpu2 = std::chrono::high_resolution_clock::now();

    const auto start_cpu3 = std::chrono::high_resolution_clock::now();
    int *sumarr = array_seed(N);
    int *d_inputArraysum;
    const auto end_cpu3 = std::chrono::high_resolution_clock::now();

    const auto start_gpu3 = std::chrono::high_resolution_clock::now();
    cudaMalloc(&d_inputArraysum, N * sizeof(int)); 
    cudaMemcpyAsync(d_inputArraysum, sumarr, N * sizeof(int), cudaMemcpyHostToDevice, streams[2]); // cpu memory to GPU memory

    reductionSum<<<gridSize,blockSize>>>(d_inputArraysum,N,d_resultsum);
    cudaMemcpyAsync(&resultsum,d_resultsum, sizeof(int),cudaMemcpyDeviceToHost, streams[2]);
    const auto end_gpu3 = std::chrono::high_resolution_clock::now();

    const auto start_cpu4 = std::chrono::high_resolution_clock::now();
    int *mularr = array_seed(N);
    int *d_inputArraymultiply;
    const auto end_cpu4 = std::chrono::high_resolution_clock::now();

    const auto start_gpu4 = std::chrono::high_resolution_clock::now();
    cudaMalloc(&d_inputArraymultiply, N * sizeof(int)); 
    cudaMemcpyAsync(d_inputArraymultiply, mularr, N * sizeof(int), cudaMemcpyHostToDevice, streams[3]); // cpu memory to GPU memory

    reductionMultiply<<<gridSize,blockSize>>>(d_inputArraymultiply,N,d_resultmultiply);
    cudaMemcpyAsync(&resultmultiply,d_resultmultiply, sizeof(int),cudaMemcpyDeviceToHost, streams[3]);
    const auto end_gpu4 = std::chrono::high_resolution_clock::now();
    
    
    cudaDeviceSynchronize();


    // const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed_seconds_cpu{end_cpu1 -start_cpu1 + end_cpu2 - start_cpu2  + end_cpu3 - start_cpu3 + end_cpu4 -start_cpu4};
    const std::chrono::duration<double> elapsed_seconds_gpu{end_gpu1 - start_gpu1 + end_gpu2 - start_gpu2 + end_gpu3 - start_gpu3 + end_gpu4 -start_gpu4};

    cudaFree(d_inputArrayreduction);
    cudaFree(d_inputArraymin);
    cudaFree(d_inputArraymultiply);
    cudaFree(d_inputArraysum);
    outputFile << k << "," << elapsed_seconds_cpu.count() <<"," << elapsed_seconds_gpu.count() << std::endl;
    // std::cout << "timing: " << elapsed_seconds_gpu2.count();

    }
}
    std::cout << "GPU reduction" << std::endl;
    std::cout << "Max: " << resultmax << std::endl;
    std::cout << "min: " << resultmin << std::endl;
    std::cout << "sum: " << resultsum << std::endl;
    std::cout << "multiply: " << resultmultiply << std::endl;
    // std::cout << "timing: " << elapsed_seconds_gpu2.count();
    


    cudaFree(d_resultmax);
    cudaFree(d_resultmin);
    cudaFree(d_resultmultiply);
    cudaFree(d_resultsum);

    return 0;
}