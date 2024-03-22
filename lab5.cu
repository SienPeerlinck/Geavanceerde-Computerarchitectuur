#include <iostream>
#include <cuda_runtime.h>
#include <chrono>


#define N 96 // Size of the arrays


__global__ void reductionMax(int* in) {
    int step = 1;
    unsigned int i = (blockIdx.x * blockDim.x + threadIdx.x);
    while(i % step == 0 && step < N) {
        unsigned int k = 0;
        while(i + k * gridDim.x * blockDim.x + step < N) {
            int index = i + k * gridDim.x * blockDim.x;
            if (in[2 * index] < in[2 * index + step]) in[2 * index] = in[2 * index + step];
            k++;
        }
        step *= 2;
        __syncthreads();
    }
}

__global__ void reductionMin(int* in) {
    int step = 1;
    unsigned int i = (blockIdx.x * blockDim.x + threadIdx.x);
    while(i % step == 0 && step < N) {
        int k = 0;
        while(i + k * gridDim.x * blockDim.x + step < N) {
            int index = i + k * gridDim.x * blockDim.x;
            if(in[2*index] > in[2*index + step]) in[2*index] = in[2*index + step];

            k++;
        }
        step *= 2;
        __syncthreads();
    }
}

__global__ void reductionSum(int* in) {
    int step = 1;
    unsigned int i = (blockIdx.x * blockDim.x + threadIdx.x);
    while(i % step == 0 && step < N) {
        int k = 0;
        while(i + k * gridDim.x * blockDim.x + step < N) {
            int index = i + k * gridDim.x * blockDim.x;
            in[2*index] += in[2*index + step];
            k++;
        }
        step *= 2;
        __syncthreads();
    }
}

__global__ void reductionMultiply(int* in) {
    int step = 1;
    unsigned int i = (blockIdx.x * blockDim.x + threadIdx.x);
    while(i % step == 0 && step < N) {
        int k = 0;
        while(i + k * gridDim.x * blockDim.x + step < N) {
            int index = i + k * gridDim.x * blockDim.x;
            in[2*index] *= in[2*index + step];
            k++;
        }
        step *= 2;
        __syncthreads();
    }
}

int* array_seed(){
        unsigned int seed = time(NULL);
    int* inputArray = new int[N];
    for (int i=0; i<N; i++){
        srand(seed + i);
        inputArray[i] = 1;//(rand()%100)+1;
        std::cout << inputArray[i];
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


   
    // Define grid and block size
    int blockSize = N;  // aantal threads in block
    int gridSize = 1;

//GPU reduction
    int resultmax = 0;
    int resultmin = 0;
    int resultsum = 0;
    int resultmultiply = 0;
    // int *d_resultmax;
    // int *d_resultmin;
    // int *d_resultsum;
    // int *d_resultmultiply;
    // cudaMalloc(&d_resultmax, sizeof(int));
    // cudaMalloc(&d_resultmin, sizeof(int));
    // cudaMalloc(&d_resultsum, sizeof(int));
    // cudaMalloc(&d_resultmultiply, sizeof(int));
    const auto start = std::chrono::high_resolution_clock::now();

    // kerneleverything<<<gridSize, blockSize>>>(kernel, d_inputArray, d_result, N);
    // int max_arr = array_seed();
    int *maxarr = array_seed();
    int *d_inputArrayreduction;
    cudaMalloc(&d_inputArrayreduction, N * sizeof(int)); 
    cudaMemcpy(d_inputArrayreduction, maxarr, N * sizeof(int), cudaMemcpyHostToDevice); // cpu memory to GPU memory

    reductionMax<<<gridSize,blockSize>>>(d_inputArrayreduction);
    cudaMemcpy(&resultmax,d_inputArrayreduction, sizeof(int),cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();

    int *minarr = array_seed();
    int *d_inputArraymin;
    cudaMalloc(&d_inputArraymin, N * sizeof(int)); 
    cudaMemcpy(d_inputArraymin, minarr, N * sizeof(int), cudaMemcpyHostToDevice); // cpu memory to GPU memory

    reductionMin<<<gridSize,blockSize>>>(d_inputArraymin);
    cudaMemcpy(&resultmin,d_inputArraymin, sizeof(int),cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();

    int *sumarr = array_seed();
    int *d_inputArraysum;
    cudaMalloc(&d_inputArraysum, N * sizeof(int)); 
    cudaMemcpy(d_inputArraysum, sumarr, N * sizeof(int), cudaMemcpyHostToDevice); // cpu memory to GPU memory

    reductionSum<<<gridSize,blockSize>>>(d_inputArraysum);
    cudaMemcpy(&resultsum,d_inputArraysum, sizeof(int),cudaMemcpyDeviceToHost);
    // cudaDeviceSynchronize();

    int *mularr = array_seed();
    int *d_inputArraymultiply;
    cudaMalloc(&d_inputArraymultiply, N * sizeof(int)); 
    cudaMemcpy(d_inputArraymultiply, mularr, N * sizeof(int), cudaMemcpyHostToDevice); // cpu memory to GPU memory

    reductionMultiply<<<gridSize,blockSize>>>(d_inputArraymultiply);
    cudaMemcpy(&resultmultiply,d_inputArraymultiply, sizeof(int),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();


    cudaFree(d_inputArrayreduction);
    cudaFree(d_inputArraymin);
    cudaFree(d_inputArraymultiply);
    cudaFree(d_inputArraysum);

    const auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed_seconds_gpu2{end - start};

    std::cout << "GPU reduction" << std::endl;
    std::cout << "Max: " << resultmax << std::endl;
    std::cout << "min: " << resultmin << std::endl;
    std::cout << "sum: " << resultsum << std::endl;
    std::cout << "multiply: " << resultmultiply << std::endl;
    std::cout << "timing: " << elapsed_seconds_gpu2.count();
    

    return 0;
}