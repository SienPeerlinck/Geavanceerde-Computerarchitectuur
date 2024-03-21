
/*
 * To convert an image to a pixel map, run `convert <name>.<extension> <name>.ppm
 *
 */
#include <cstdint>      // Data types
#include <iostream>     // File operations
#include <cuda_runtime.h>
#include <chrono>

// #define M 512       // Lenna width
// #define N 512       // Lenna height
#define M 941       // VR width
#define N 704       // VR height
#define C 3         // Colors
#define OFFSET 15   // Header length


__global__ void invert_image(int *image_array, int *output_array){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;

    for(int i = tid; i<M*N*C; i+=stride){
        output_array[i] = 255 - image_array[i];
    }


}


uint8_t* get_image_array(void){
    /*
     * Get the data of an (RGB) image as a 1D array.
     *
     * Returns: Flattened image array.
     *
     * Notes:
     *  - Images data is flattened per color, column, row.
     *  - The first 3 data elements are the RGB components
     *  - The first 3*M data elements represent the firts row of the image
     *  - For example, r_{0,0}, g_{0,0}, b_{0,0}, ..., b_{0,M}, r_{1,0}, ..., b_{b,M}, ..., b_{N,M}
     *
     */        
    // Try opening the file
    FILE *imageFile;
    imageFile=fopen("./input_image.ppm","rb");
    if(imageFile==NULL){
        perror("ERROR: Cannot open output file");
        exit(EXIT_FAILURE);
    }
   
    // Initialize empty image array
    uint8_t* image_array = (uint8_t*)malloc(M*N*C*sizeof(uint8_t)+OFFSET);
   
    // Read the image
    fread(image_array, sizeof(uint8_t), M*N*C*sizeof(uint8_t)+OFFSET, imageFile);
   
    // Close the file
    fclose(imageFile);
       
    // Move the starting pointer and return the flattened image array
    return image_array + OFFSET;
}


void save_image_array(uint8_t* image_array){
    /*
     * Save the data of an (RGB) image as a pixel map.
     *
     * Parameters:
     *  - param1: The data of an (RGB) image as a 1D array
     *
     */            
    // Try opening the file
    FILE *imageFile;
    imageFile=fopen("./output_image.ppm","wb");
    if(imageFile==NULL){
        perror("ERROR: Cannot open output file");
        exit(EXIT_FAILURE);
    }
   
    // Configure the file
    fprintf(imageFile,"P6\n");               // P6 filetype
    fprintf(imageFile,"%d %d\n", M, N);      // dimensions
    fprintf(imageFile,"255\n");              // Max pixel
   
    // Write the image
    fwrite(image_array, 1, M*N*C, imageFile);
   
    // Close the file
    fclose(imageFile);
}


int main (void) {
   
    // Read the image
    uint8_t* image_array = get_image_array();
   
    // Allocate output
    uint8_t* output_array = (uint8_t*)malloc(M*N*C);
   
    // Convert to grayscale using only the red color component
    //for(int i=0; i<M*N*C; i++){
    //    new_image_array[i] = image_array[i/3*3];
    //}
   
    // Save the image
    //save_image_array(new_image_array);


    int *d_image_array;
    cudaMalloc(&d_image_array, M*N*C);
    cudaMemcpy(d_image_array, image_array, M*N*C, cudaMemcpyHostToDevice);

    int *d_output_array;
    cudaMalloc(&d_output_array, M*N*C);

    int blockSize = 1024;
    int gridSize = 10;

    const auto startinvert = std::chrono::high_resolution_clock::now();
   
    invert_image<<<gridSize, blockSize>>>(d_image_array, d_output_array);  // geef pointers naar arrays mee
    cudaMemcpy(output_array, d_output_array, M*N*C, cudaMemcpyDeviceToHost); // cpu memory to GPU memory

    cudaFree(d_output_array);

    const auto endinvert = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> elapsed_seconds_invert{endinvert - startinvert};

    std::cout << elapsed_seconds_invert.count() << "\n";
    save_image_array(output_array);
    return 0;
}