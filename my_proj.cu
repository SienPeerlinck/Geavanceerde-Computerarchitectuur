

#include <iostream>
#include <cuda_runtime.h>

#define N 8 // Size of the arrays

// GPU kernel
__global__ void arrayFlip(int *input, int *output, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;  // totale index = nummer van blok waar we nu inzitten x aantal blokken + nummer van de thread

    if (tid < size) {  // stopt na de volledige array is doorlopen
        output[tid] = input[size - tid - 1];  // eigenlijke functie die arrays omdraait
    }
}


// CPU code
int main() {
    int inputArray[N];  // define arrays of size N
    int outputArray[N];

    int *d_inputArray, *d_outputArray;  // define pointer to address of above

    // Fill input array with numbers
    for (int i = 0; i < N; i++) {
        inputArray[i] = i;
    }

    // Allocate memory on GPU on the address pointed to by the defined pointer with the size of N times the size of an int (so the total size of the array)
    cudaMalloc((void **)&d_inputArray, N * sizeof(int));  
    cudaMalloc((void **)&d_outputArray, N * sizeof(int));

    // Put defined input array in the memory allocated above
    cudaMemcpy(d_inputArray, inputArray, N * sizeof(int), cudaMemcpyHostToDevice); // cpu memory to GPU memory

    // Define grid and block size
    int blockSize = 256;  // aantal threads in block
    int gridSize = (N + blockSize - 1) / blockSize;  // aantal blocks in grid, made large enough to cover all N without having too much threads (resource waste) (rond naar beneden af)
    // in case of N=2000 and block_size = 256 : gridsize=8.8 => 8 ; 8*256=2048>2000

    // Launch GPU kernel to flip the array
    arrayFlip<<<gridSize, blockSize>>>(d_inputArray, d_outputArray, N);  // geef pointers naar arrays mee

    // put calculated output array in allocated memory
    cudaMemcpy(outputArray, d_outputArray, N * sizeof(int), cudaMemcpyDeviceToHost);  // gpu memory to CPU memory

    // Free memory on GPU (want als ge memory allocate, moet ge het ook weer freëen)
    cudaFree(d_inputArray);
    cudaFree(d_outputArray);

    // Print the result (standaard cpp stuff)
    std::cout << "Input Array: ";
    for (int i = 0; i < N; i++) {
        std::cout << inputArray[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Output Array: ";
    for (int i = 0; i < N; i++) {
        std::cout << outputArray[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
