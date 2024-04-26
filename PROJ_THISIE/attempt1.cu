
#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define width 384
#define height 288



uint8_t* get_image_array(void){
    /*
     * Get the data of an (RGB) image as a 1D array.
     * 
     * Returns: Flattened image array.
     * 
     * Noets:
     *  - Images data is flattened per color, column, row.
     *  - The first 3 data elements are the RGB componentsinvert_image
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

__global__ void felics{uint8_t *image_array, uint8_t *ouput_array}{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;

    ouput_array[0] = image_array[0];
    ouput_array[1] = image_array[1];

    for{ int i = tid + 2*stride; i < height; i+= stride}{
        flipfloep<<<gridSize,blockSize>>>(image_array[i-2*stride], image_array[i-stride], output_array[i]);

    }
}

__global__ void flipfloep {int *N1, int *N2, int *P}{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;

    for{int i= tid + 2*stride; i < width; i += stride}{
        //bepalen welke hoogste is (N1 of N2) voor door te sturen naar ABC
        //-> hier delta al berekenen, maar 1 getal meegeven ipv 2
        if( N1 > N2){
            int H = N1;
            int L = N2;
        }
        else{
            int H = N2;
            int L = N1;
        }
        int delta = H - L;
        if( delta < 255){
            delta = 255;
        }
        
    }
}

__global__ void ABC{int *delta, int* P}{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;

    int range = delta + 1;
    int upper_bound = ceil(log2(delta));
    int threshold = 2**(upper_bound) - range;
    int shift = (range - threshold)/2;

    if(P < shift){
        int result = (P + threshold)*2 + 1;
    }
    else if(P > delta - shift){
        int result = (delta - P + threshold)*2;
    }
    else{
        int result = P - shift;
    }
    return result;
}

__global__ void SEC{int* P}{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;

    int k = 12;
    int bitlen = ceil(log2(P+1)) - 1;
    if(bitlen < k){
        int b = k;
        int u = 0;
    }
    else{
        int b = bitlen;
        int u = b - k + 1;
    }
    int result_arr[u+b+1];
    for(int i; i < u; i++){
        result_arr[i] = 1;
    }
    result_arr[u+1] = 0;
    int O[32];
    for(int i =0; i< 31; i++){
        O[i] = 0;
    }
    std::bitset<32> A=P;
    for(int i=32-b,j =b-1;i<32;i++,j--){
        O[i] = A[j];
    }




    int p_arr[ceil(log2(P+1))];
    for(int i; i< ceil(log2(P+1)); i++){
        p_arr[i] = 
    }
}




int main (void){



}