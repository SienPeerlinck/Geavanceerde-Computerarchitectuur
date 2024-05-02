#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <bitset>
using namespace std;

#define width 384
#define height 288

#define gridSize 1
#define blockSize 96

string toBinary(int n){
    string r;
    while(n != 0){
        r=(n%2==0 ?"0":"1")+r;
        n/=2;
    }
    return r;
}

double log2v2(int n){
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

__global__ void felics(uint8_t *image_array, string *output_array){
   /*
   ontvangt rij van pixels
   selecteert telkens 3 pixls om naar flipfloep te sturen
   ontvangt van flipfloep terug gecodeerde pixel om in output array te steken
   */
   int tid = blockIdx.x * blockDim.x + threadIdx.x;
   int stride = blockDim.x*gridDim.x;

   output_array[0] = image_array[0];
   output_array[1] = image_array[1];

   string coded_pixel[width];

   for(int i = tid + 2*stride; i < height; i+= stride){
       flipfloep(image_array[i-2*stride], image_array[i-stride], image_array[i], coded_pixel[i]);
       output_array[i] = coded_pixel[i];

   }
}

__global__ void flipfloep (uint8_t N1, uint8_t N2, uint8_t P, string pixel){

   for(int i= 2; i < width; i++){
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
       string res_str;
       if(P < H && P > L){
            int x = P-L;
            ABC(&delta, &x, &result);
            res_str = toBinary(result);
            //res_str = bitset<8>(result);
            res_str = "0"+res_str;
           
       }
       else if(P < L ){
            int x = L-P;
            SEC(&x, &result);
            res_str = toBinary(result);
            //res_str = bitset<8>(result);
            res_str = "10"+res_str;
       }
       else {
            int x = P-H;
            SEC(&x, &result);
            res_str = toBinary(result);
            //res_str = bitset<8>(result);
            res_str = "11"+res_str;
       }
       pixel = res_str;
       
   }
}

__global__ void ABC(int *delta, int* P, int *result){

   int range = *delta + 1;
   int upper_bound = int(ceil(log2(double(int(delta)))));
   int threshold = pow(2,upper_bound) - range;
   int shift = range - threshold/2;


   if(*P < shift){
       *result = (*P + threshold)*2 + 1;
   }
   else if(P > delta - shift){
       *result = (*delta - *P + threshold)*2;
   }
   else{
       *result = *P - shift;
   }
}

__global__ void SEC(int* P, int *result){
   int b,u =0;
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
   int result_arr[u+b+1];
       for (int i = 0; i < u+b+1; i++)
       {
           result_arr[i] = 5;
       }
   
   for(int i = 0; i < u; i++){
       result_arr[i] = 1;
   }
   result_arr[u] = 0;

       for (int i = 0; i < u+b+1; i++)
       {
           std::cout<<result_arr[i];
       }
       std::cout << std::endl;
   int sum;
   std::bitset<32> A=*P;
   for (int i=u+1,j=b-1; i < u+b+1; i++, j--)
   {
           result_arr[i]=A[j];
   }
   for(int i =0,j=u+b; i<u+b+1; i++,j--){
       sum = sum + result_arr[j]*pow(2,i);
   }
   *result = sum;
}




int main (void){
   uint8_t* image_array = get_image_array();

   string* output_array = (string*)malloc(width*height*sizeof(string));

   uint8_t *d_image_array;
   cudaMalloc(&d_image_array,width*height*sizeof(uint8_t));
   cudaMemcpy(d_image_array,image_array,width*height*sizeof(uint8_t),cudaMemcpyHostToDevice);

   string *d_output_array;
   cudaMalloc(&d_output_array,width*height*sizeof(string));

   const auto start = std::chrono::high_resolution_clock::now();
   felics<<<gridSize,blockSize>>>(d_image_array,d_output_array);
   cudaMemcpy(output_array,d_output_array,width*height*sizeof(string),cudaMemcpyDeviceToHost);
   cudaFree(d_image_array);
   cudaFree(d_output_array);
   const auto end = std::chrono::high_resolution_clock::now();

    return 0;
}