#include "autocorrelation_gpu.cuh"
#include <stdio.h>

texture<float, 1, cudaReadModeElementType> texIx;
texture<float, 1, cudaReadModeElementType> texIy;

__global__ void compute_autocorrelation_matrix_kernel_texture(float *A, float *B, float *C) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int p = x + y * blockDim.x * gridDim.x;

    float Ix_value = tex1D(texIx, p);  // Read from texture memory
    float Iy_value = tex1D(texIy, p);

    A[p] = Ix_value * Ix_value;
    B[p] = Ix_value * Iy_value;
    C[p] = Iy_value * Iy_value;

    if (p == 0)
        printf("%f, %f ", Ix_value, A[0]);
}

// __global__ void compute_autocorrelation_matrix_kernel(float *Ix, float *Iy, float *A, float *B, float *C, int nx, int ny) {
//     int x = blockIdx.x * blockDim.x + threadIdx.x ;
//     int y = blockIdx.y * blockDim.y + threadIdx.y ;
//     int p = x + y*blockDim.x*gridDim.x ;
//     A[p] = Ix[p]*Ix[p];
//     B[p] = Ix[p]*Iy[p];
//     C[p] = Iy[p]*Iy[p];
//     if(p == 0)
//         printf("%f, %f ",Ix[0], A[0]);
// }

//avec texture
void compute_autocorrelation_matrix_cuda(float *Ix, float *Iy, float *A, float *B, float *C, int nx, int ny){
    int imageSize = nx * ny;
    // Allocate device memory
    float *d_A, *d_B, *d_C;

    // Texture
    cudaArray *d_IxArray, *d_IyArray;
    cudaMallocArray(&d_IxArray, &texIx.channelDesc, imageSize, 1);
    cudaMallocArray(&d_IyArray, &texIy.channelDesc, imageSize, 1);
    // Copy input data to texture memory
    cudaMemcpyToArray(d_IxArray, 0, 0, Ix, imageSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToArray(d_IyArray, 0, 0, Iy, imageSize * sizeof(float), cudaMemcpyHostToDevice);

    // Bind texture references to the arrays
    cudaBindTextureToArray(texIx, d_IxArray);
    cudaBindTextureToArray(texIy, d_IyArray);

    cudaMalloc((void**)&d_A, imageSize * sizeof(float));
    cudaMalloc((void**)&d_B, imageSize * sizeof(float));
    cudaMalloc((void**)&d_C, imageSize * sizeof(float));
    
    // Define block and grid dimensions
    dim3 blockSize(10, 10); // Adjust block size based on your GPU architecture
    dim3 gridSize((int)(nx/blockSize.x), (int)(ny / blockSize.y));
    // Launch the kernel
    compute_autocorrelation_matrix_kernel_texture<<<gridSize, blockSize>>>(d_A, d_B, d_C);

    // Wait for kernel completion
    cudaDeviceSynchronize();
    // Copy the results back to host
    cudaMemcpy(A, d_A, imageSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(B, d_B, imageSize * sizeof(float), cudaMemcpyDeviceToHost);    
    cudaMemcpy(C, d_C, imageSize * sizeof(float), cudaMemcpyDeviceToHost);

    // Unbind texture references
    cudaUnbindTexture(texIx);
    cudaUnbindTexture(texIy);
    // printf("A:\n");
    // for (int j = 0; j < 100; j++){
	// 	printf("%f",A[j]);
	// }
    // printf("B:\n");
    // for (int j = 0; j < imageSize; j++){
	// 	printf("%f",B[j]);
	// }
    // printf("C:\n");
    // for (int j = 0; j < imageSize; j++){
	// 	printf("%f",C[j]);
	// }
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeArray(d_IxArray);
    cudaFreeArray(d_IyArray);
}

// sans texture 
// void compute_autocorrelation_matrix_cuda(float *Ix, float *Iy, float *A, float *B, float *C, int nx, int ny){
//     int imageSize = nx * ny;
//     // Allocate device memory
//     float *d_Ix, *d_Iy, *d_A, *d_B, *d_C;

//     cudaMalloc((void**)&d_Ix, imageSize * sizeof(float));
//     cudaMalloc((void**)&d_Iy, imageSize * sizeof(float));
//     cudaMalloc((void**)&d_A, imageSize * sizeof(float));
//     cudaMalloc((void**)&d_B, imageSize * sizeof(float));
//     cudaMalloc((void**)&d_C, imageSize * sizeof(float));
    
//     // copy in the memory
//     cudaMemcpy(d_Ix, Ix, imageSize * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_Iy, Iy, imageSize * sizeof(float), cudaMemcpyHostToDevice);
//     // Define block and grid dimensions
//     dim3 blockSize(10, 10); // Adjust block size based on your GPU architecture
//     dim3 gridSize((int)(nx/blockSize.x), (int)(ny / blockSize.y));
//     // Launch the kernel
//     compute_autocorrelation_matrix_kernel<<<gridSize, blockSize>>>(d_Ix, d_Iy, d_A, d_B, d_C, nx, ny);
//     // Wait for kernel completion
//     cudaDeviceSynchronize();
//     // Copy the results back to host
//     cudaMemcpy(A, d_A, imageSize * sizeof(float), cudaMemcpyDeviceToHost);
//     cudaMemcpy(B, d_B, imageSize * sizeof(float), cudaMemcpyDeviceToHost);    
//     cudaMemcpy(C, d_C, imageSize * sizeof(float), cudaMemcpyDeviceToHost);
//     // printf("A:\n");
//     // for (int j = 0; j < 100; j++){
// 	// 	printf("%f",A[j]);
// 	// }
//     // printf("B:\n");
//     // for (int j = 0; j < imageSize; j++){
// 	// 	printf("%f",B[j]);
// 	// }
//     // printf("C:\n");
//     // for (int j = 0; j < imageSize; j++){
// 	// 	printf("%f",C[j]);
// 	// }
//     // Free device memory
//     cudaFree(d_Ix);
//     cudaFree(d_Iy);
//     cudaFree(d_A);
//     cudaFree(d_B);
//     cudaFree(d_C);
// }
