
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

 __device__ int* toBinary(int n, int* r) {
    int count = 0;
    for (int i = 0; i < 35; i++) r[i] = 0;  // Initialize the array
    while (n != 0) {
        r[count + 2] = (n % 2 == 0 ? 0 : 1);
        n /= 2;
        count++;
    }
    r[34] = count;
    return r;
}

 __device__ double log2v2(int n) {
    int count = 0;
    while (n >>= 1) count++;
    return count;
}

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

void save_image_array(uint8_t* image_array) {
    FILE* imageFile = fopen("./compressed_img.csv", "wb");
    if (imageFile == NULL) {
        perror("ERROR: Cannot open output file");
        exit(EXIT_FAILURE);
    }
    fprintf(imageFile, "P6\n");
    fprintf(imageFile, "%d %d\n", width, height);
    fprintf(imageFile, "255\n");
    fwrite(image_array, 1, width * height, imageFile);
    fclose(imageFile);
}

 __device__ int ABC(int delta, int P) {
    int range = delta + 1;
    int upper_bound = int(ceil(log2(double(delta))));
    int threshold = pow(2, upper_bound) - range;
    int shift = range - threshold / 2;
    int result;
    if (P < shift) {
        result = (P + threshold) * 2 + 1;
    }
    else if (P > delta - shift) {
        result = (delta - P + threshold) * 2;
    }
    else {
        result = P - shift;
    }
    return result;
}


 __device__ int SEC(int P) {
    int result = 0;
    int b = 0;
    int u = 0;
    int k = 12;
    int bitlen = ceil(log2v2(P + 1)) - 1;
    if (bitlen < k) {
        b = k;
        u = 0;
    }
    else {
        b = bitlen;
        u = b - k + 1;
    }
    int* result_arr = (int*)malloc((u + b + 1) * sizeof(int));
    for (int i = 0; i < u; i++) {
        result_arr[i] = 1;
    }
    result_arr[u] = 0;
    int *A;
    toBinary(P,A);
    // std::bitset<32> A = P;
    for (int i = u + 1, j = b - 1; i < u + b + 1; i++, j--) {

        result_arr[i] = A[j];
    }
    for (int i = 0, j = u + b; i < u + b + 1; i++, j--) {
        result += result_arr[j] * pow(2, i);
    }
    free(result_arr);
    return result;
}

 __device__ int* flipfloep(uint8_t N1, uint8_t N2, uint8_t P, int* res_str) {
    int H, L;
    if (N1 > N2) {
        H = N1;
        L = N2;
    }
    else {
        H = N2;
        L = N1;
    }
    int delta = H - L;
    if (delta < 255) {
        delta = 255;
    }
    int result;
    int temp_res[35];
    if (P < H && P > L) {
        int x = P - L;
        result = ABC(delta, x);
        toBinary(result, res_str);
        res_str[0] = 0;
        res_str[1] = 0;
    }
    else if (P < L) {
        int x = L - P;
        result = SEC(x);
        toBinary(result, res_str);
        res_str[0] = 1;
        res_str[1] = 0;
    }
    else {
        int x = P - H;
        result = SEC(x);
        toBinary(result, res_str);
        res_str[0] = 1;
        res_str[1] = 1;
    }
    // return res_str;
    //res_str = temp_res;
    return res_str;
}

__global__ void felics(uint8_t* image_array, int* output_array) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < width * height; i += stride) {
        if (i % width == 0 || i % width == 1) {
            output_array[i * 35] = image_array[i]; // Adjust the index for the flattened structure
        }
        else {
            // int* res_str[35];
            // *res_str[2] = 2;
            // res_str[1] = flipfloep(image_array[i - 2], image_array[i - 1], image_array[i], *res_str); // Use the correct indices
            // // for (int j = 0; j < 35; j++) {
            // output_array[i] = *res_str[1]; //res[j]; // Store the 35-bit result
            // // }
            output_array[i] = {5};
        }
    }

    //    for (int i = tid; i < width * height; i += stride) {
    //     output_array[i * 35] = 5; // Set a known value
    // }

}

int main(void) {
    uint8_t* image_array = get_image_array();
    int* output_array = (int*)malloc(width * height * 35 * sizeof(int));

    // for (int i=0; i<width*height;i++){
    //     std::cout << image_array[i] << std::endl;
    // }

    uint8_t* d_image_array;
    cudaMalloc(&d_image_array, width * height * sizeof(uint8_t));
    cudaMemcpy(d_image_array, image_array, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // std::cout << image_array << std::cout;

    int* d_output_array;
    cudaMalloc(&d_output_array, width * height * 35 * sizeof(int));

    const auto start = std::chrono::high_resolution_clock::now();
    felics<<<gridSize, blockSize>>>(d_image_array, d_output_array);
    cudaMemcpy(output_array, d_output_array, width * height * 35 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    const auto end = std::chrono::high_resolution_clock::now();

    // std::cout << output_array[100] << std::endl;

    string s;
    for (int i = 0; i < width * height; i++) {
        if (i % width == 0 || i % width == 1) {
            //std::cout << "onveranderde pixel " << output_array[i] << " " << i << std::endl;
        }
        else {
        //     s = "";
        //     for (int j = 0; j < 35; j++) {
        //         s += to_string(output_array[i * 35 + j]);
        //     }
        //     std::cout << s << " " << i << std::endl;
        std::cout << output_array[i] << " " << i << std::endl;
        }
    }

    cudaFree(d_image_array);
    cudaFree(d_output_array);
    free(image_array);
    free(output_array);

    return 0;
}
