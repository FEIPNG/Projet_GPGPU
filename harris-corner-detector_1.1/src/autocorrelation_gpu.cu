#include "autocorrelation_gpu.cuh"

__global__
void compute_autocorrelation_matrix_kernel(float *Ix, float *Iy, float *A, float *B, float *C, int nx, int ny) {
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int p = i * nx + j;
    A[p] = Ix[p]*Ix[p];
    B[p] = Ix[p]*Iy[p];
    C[p] = Iy[p]*Iy[p];
}

void compute_autocorrelation_matrix_cuda(float *Ix, float *Iy, float *A, float *B, float *C, int nx, int ny){
    int imageSize = nx * ny;
    // Allocate device memory
    float *d_Ix, *d_Iy, *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_Ix, imageSize * sizeof(float));
    cudaMalloc((void**)&d_Iy, imageSize * sizeof(float));
    cudaMalloc((void**)&d_A, imageSize * sizeof(float));
    cudaMalloc((void**)&d_B, imageSize * sizeof(float));
    cudaMalloc((void**)&d_C, imageSize * sizeof(float));
    // copy in the memory
    cudaMemcpy(Ix, d_Ix, imageSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Iy, d_Iy, imageSize * sizeof(float), cudaMemcpyDeviceToHost);
    // Define block and grid dimensions
    dim3 blockSize(16, 16); // Adjust block size based on your GPU architecture
    dim3 gridSize(nx / blockSize.x, ny / blockSize.y);
    // Launch the kernel
    compute_autocorrelation_matrix_kernel<<<gridSize, blockSize>>>(d_Ix, d_Iy, d_A, d_B, d_C, nx, ny);
    // Wait for kernel completion
    cudaDeviceSynchronize();
    // Copy the results back to host
    cudaMemcpy(A, d_A, imageSize * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(B, d_B, imageSize * sizeof(float), cudaMemcpyDeviceToHost);    
    cudaMemcpy(C, d_C, imageSize * sizeof(float), cudaMemcpyDeviceToHost);    
    // Free device memory
    cudaFree(d_Ix);
    cudaFree(d_Iy);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
