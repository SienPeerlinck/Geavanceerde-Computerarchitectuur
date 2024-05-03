#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <bitset>
using namespace std;

#define width 384
#define height 288

#define gridSize 1
#define blockSize 96


__host__ __device__ int* toBinary(int n){
    int count;
    int r[35];

    while(n != 0){
        r[count+2] = (n%2==0 ?0:1)+r[count+2];
        n/=2;
        count ++;
    }
    r[34] = count;
    int* x =  r;
    return x;
}


__host__ __device__ double log2v2(int n){
    int count = 0;
    while (n >>= 1) count ++;
    return count;
}


uint8_t* get_image_array(void){
   FILE *imageFile;
   imageFile=fopen("./input_image.ppm","rb");
   if(imageFile==NULL){
       perror("ERROR: Cannot open output file");
       exit(EXIT_FAILURE);
   }
   // Initialize empty image array
   uint8_t* image_array = (uint8_t*)malloc(width*height*sizeof(uint8_t));
   
   // Read the image
   fread(image_array, sizeof(uint8_t), width*height*sizeof(uint8_t), imageFile);
       
   // Close the file
   fclose(imageFile);
       
   // Move the starting pointer and return the flattened image array
   return image_array;


}

void save_image_array(uint8_t* image_array){

   FILE *imageFile;
   imageFile=fopen("./output_image.ppm","wb");
   if(imageFile==NULL){
       perror("ERROR: Cannot open output file");
       exit(EXIT_FAILURE);
   }
   
   // Configure the file
   fprintf(imageFile,"P6\n");               // P6 filetype
   fprintf(imageFile,"%d %d\n", width, height);      // dimensions
   fprintf(imageFile,"255\n");              // Max pixel
   
   // Write the image
   fwrite(image_array, 1, width*height, imageFile);
   
   // Close the file
   fclose(imageFile);
}


__host__ __device__ int ABC(int *delta, int* P){

    int range = *delta + 1;
    int upper_bound = int(ceil(log2(double(*delta))));
    // int upper_bound = log2v2(*delta);
    int threshold = pow(2,upper_bound) - range;
    int shift = range - threshold/2;
    int result;

    if(*P < shift){
        result = (*P + threshold)*2 + 1;
    }
    else if(P > delta - shift){
        result = (*delta - *P + threshold)*2;
    }
    else{
        result = *P - shift;
    }
    return result;
}

__host__ __device__ int SEC(int* P){
    int result;
    int b =0;
    int u =0;
    int k = 12;
    int bitlen = ceil(log2(*P+1)) - 1;
    if(bitlen < k){
        b = k;
        u = 0;
    }
    else{
        b = bitlen;
        u = b - k + 1;
    }

    // const int const_u = u;
    // int result_arr[] = ; 
    // int result_arr[const_u+b+1];
    int* result_arr = (int*) malloc((u+b+1)*sizeof(int));
        for (int i = 0; i < u+b+1; i++)
        {
            result_arr[i] = 5;
        }
    
    for(int i = 0; i < u; i++){
        result_arr[i] = 1;
    }
    result_arr[u] = 0;

        // for (int i = 0; i < u+b+1; i++)
        // {
        //     std::cout<<result_arr[i];
        // }
    int sum;
    std::bitset<32> A=*P;
    for (int i=u+1,j=b-1; i < u+b+1; i++, j--)
    {
            result_arr[i]=A[j];
    }
    for(int i =0,j=u+b; i<u+b+1; i++,j--){
        sum = sum + result_arr[j]*pow(2,i);
    }
    free(result_arr);
    result = sum;
    return result;
}




__host__ __device__ int* flipfloep (uint8_t N1, uint8_t N2, uint8_t P){
    //bepalen welke hoogste is (N1 of N2) voor door te sturen naar ABC
    //-> hier delta al berekenen, maar 1 getal meegeven ipv 2
    int H, L;
    if( N1 > N2){
            H = N1;
            L = N2;
    }
    else{
        H = N2;
        L = N1;
    }
    int delta; 
    delta = int(H - L);
    if( delta < 255){
        delta = 255;
    }
    int result;
    int res_str[35];
    //array maken van 33 -> int is max 32 groot, 33ste getal wordt gebruikt om bitlengte in te steken -> zo bij uitlezen weet men welke van de eerste X bits men moet uitlezen
    //1ste 2 bits van de array overlaten om dat dit dan de prefix wordt 0->00 voor de makkelijkheid
    if(P < H && P > L){
            int x = P-L;
            result = ABC(&delta, &x);
            *res_str = *toBinary(result);
            //res_str = bitset<8>(result);
            res_str[0] = 0;
            res_str[1] = 0;
        
    }
    else if(P < L ){
            int x = L-P;
            result = SEC(&x);
            *res_str = *toBinary(result);
            //res_str = bitset<8>(result);
            res_str[0] = 1;
            res_str[1] = 0;
    }
    else {
            int x = P-H;
            result = SEC(&x);
            *res_str = *toBinary(result);
            //res_str = bitset<8>(result);
            res_str[0] = 1;
            res_str[1] = 1;
    }
    int *x =res_str;
    return x;
}


// __global__ void felics(uint8_t *image_array, string *output_array){
__global__ void felics(uint8_t *image_array, int *output_array){
    // void felics(uint8_t *image_array, string *output_array){
    /*
    ontvangt rij van pixels
    selecteert telkens 3 pixels om naar flipfloep te sturen
    ontvangt van flipfloep terug gecodeerde pixel om in output array te steken
    */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;

    // output_array[0] = image_array[0];
    // output_array[1] = image_array[1];


    for(int i = tid; i < width*height; i+= stride){
        if (i%width == 0 || i%width == 1){
            output_array[i] = image_array[i];
        }
        else{
             output_array[i] =  *flipfloep(image_array[i-2*stride], image_array[i-stride], image_array[i]);
    
        }
        
    }
}





int main (void){
    uint8_t* image_array = get_image_array();

    int* output_array = (int*)malloc(width*height*(sizeof(int)+3));

    uint8_t *d_image_array;
    cudaMalloc(&d_image_array,width*height*sizeof(uint8_t));
    cudaMemcpy(d_image_array,image_array,width*height*sizeof(uint8_t),cudaMemcpyHostToDevice);

    int *d_output_array;
    cudaMalloc(&d_output_array,width*height*(sizeof(int)+3));

    const auto start = std::chrono::high_resolution_clock::now();
    felics<<<gridSize,blockSize>>>(d_image_array,d_output_array);
    cudaMemcpy(output_array,d_output_array,width*height*(sizeof(int)+3),cudaMemcpyDeviceToHost);
    
    string s;
    for (int i = 0; i < width*height; i++){
        if (i%width ==0 || i%width ==1){
            std::cout << "onveranderde pixel" << output_array[i] << std::endl;
        }
        else{
            s = "";
            for (int i = 0; i < output_array[34]; i++){
                s+= to_string(output_array[i]);
            }
        std::cout << s << std::endl;
            
        }
        
    }
   
   cudaFree(d_image_array);
   cudaFree(d_output_array);
   const auto end = std::chrono::high_resolution_clock::now();

    return 0;
}