#include <cstdint>      // Data types
#include <iostream>     // File operations
#include <cuda_runtime.h>
#include <chrono>
#include <fstream>

#define BLOCK_SIZE 32
// #define width 2
#define breed 78


//constant memory
__constant__ int C_A[breed*breed];
__constant__ int C_B[breed*breed];

__global__ void matrixmultiply(int* A, int* B, int* C, int width){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width){
        // std::cout << A << std::endl;
        int tmpsum = 0;
        for (int i = 0; i < width; i++){
            tmpsum += A[row * width + i] *B[i*width + col];
        }
        __syncthreads();
        C[row*width + col] = tmpsum;

    }
    
}

__global__ void matrixmultiply_shared(int* A, int* B, int* C, int width){
    // const int blockSize = 16;
    __shared__ int As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int tmpsum = 0;

    for (int t = 0; t < width/BLOCK_SIZE; t++){
        As[threadIdx.y][threadIdx.x] = A[row*width+t*BLOCK_SIZE+threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t*BLOCK_SIZE+threadIdx.y)*width + col];
        __syncthreads();
        for (int i = 0; i < BLOCK_SIZE; i++){
            tmpsum += As[threadIdx.y][i]*Bs[i][threadIdx.x];
        }
        __syncthreads();
    }
    if(row < width && col < width){
        C[row*width + col] = tmpsum;
    }

}

__global__ void matrixmultiply_constant(int*C, int width){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < width && col < width){
        int tmpsum = 0;
        for (int i = 0; i < width; i++)
        {
            tmpsum += C_A[row *width + i] * C_B[i*width + col];
        }
        C[row * width + col] = tmpsum;
    }


}





int main (void) {

    std::ofstream outputFile("timing_constant78.csv"); // for the measurements
    if (!outputFile.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }

    for (int width = 1; width <= 78; width++){
    outputFile << "blocksize,elapsed_time" << std::endl;
    for(int i = 0; i <= 20; i += 1) {
    // int width = 512;

    int* A = new int[width*width];
    int* B = new int[width*width];
    int* C = new int[width*width];
    // int* D = new int[width*width];

    int num;

    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < width; j++)
        {
            A[i*width + j]= (rand()%20)+1;
        }
        
    }
        for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < width; j++)
        {
            B[i*width + j]= (rand()%20)+1;
        }
        
    }



    int* d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(int)*width*width);
    cudaMalloc(&d_B, sizeof(int)*width*width);
    cudaMalloc(&d_C, sizeof(int)*width*width);

    cudaMemcpy(d_A,A,sizeof(int)*width*width,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,B,sizeof(int)*width*width,cudaMemcpyHostToDevice);




    cudaMemcpyToSymbol(C_A, A, width*width*sizeof(int));
    cudaMemcpyToSymbol(C_B, B, width*width*sizeof(int));



    dim3 block(BLOCK_SIZE,BLOCK_SIZE);
    // dim3 grid(width/block.x, width/block.y);
    dim3 grid((width+ block.x - 1)/block.x, (width + block.y -1)/block.y);



    const auto startMatrix = std::chrono::high_resolution_clock::now();

    // matrixmultiply<<<grid,block>>>(d_A, d_B, d_C, width);
    matrixmultiply_shared<<<grid,block>>>(d_A, d_B,d_C, width);
    // matrixmultiply_constant<<<grid,block>>>(d_C, width);
    cudaDeviceSynchronize();

    const auto endMatrix = std::chrono::high_resolution_clock::now();

    cudaMemcpy(C, d_C, sizeof(int)*width*width, cudaMemcpyDeviceToHost);
    // cudaMemcpy(D, d_D, sizeof(int)*width*width, cudaMemcpyDeviceToHost);
    const std::chrono::duration<double> elapsed_seconds_matrix{endMatrix - startMatrix};
    
    outputFile << i << "," << elapsed_seconds_matrix.count() << std::endl;


    //  for (int i=0; i< width; i++){
    //     for(int j=0;j<width;j++){
    //         if (C[i*width + j] != D[i*width + j])
    //         {
    //             std::cout << "FOUT";
    //         }
            
    //     }
    //         std::cout << std::endl;
    // }
    //     std::cout << std::endl;


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    // cudaFree(d_D);

    delete[] A;
    delete[] B;
    delete[] C;
    // delete[] D;

    }
    }

    // std::cout << "timing: "  << elapsed_seconds_matrix.count() << std::endl;


        //     std::cout << "A" << std::endl;
    // for (int i=0; i< width; i++){
    //     for(int j=0;j<width;j++){
    //         std::cout << A[i*width + j] << "\t";
    //     }
    //         std::cout << std::endl;
    // }
    //     std::cout << std::endl;
    // std::cout << "B" << std::endl;
    // for (int i=0; i< width; i++){
    //     for(int j=0;j<width;j++){
    //         std::cout << B[i*width + j] << "\t";
    //     }
    //         std::cout << std::endl;
    // }
    //     std::cout << std::endl;
    // std::cout << "C" << std::endl;
    // for (int i=0; i< width; i++){
    //     for(int j=0;j<width;j++){
    //         std::cout << C[i*width + j] << "\t";
    //     }
    //         std::cout << std::endl;
    // }



    cudaFree(C_A);
    cudaFree(C_B);



    return 0;

}
