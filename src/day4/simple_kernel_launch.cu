#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(float *A, float *B, float *C, int N) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(globalIdx < N) {
        C[globalIdx] = A[globalIdx] + B[globalIdx];
    }
}

int main() {
    int N = 1000000;
    size_t size = N * sizeof(float);
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    // Allocate host memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // Initialize host vectors
    for(int i = 0; i < N; i++) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }

    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy host vectors to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify the result
    bool success = true;
    for(int i = 0; i < N; i++) {
        if(h_C[i] != h_A[i] + h_B[i]) {
            success = false;
            printf("Error at index %d: %f != %f + %f\n", i, h_C[i], h_A[i], h_B[i]);
            break;
        }
    }

    if(success) {
        printf("Vector addition successful!\n");
    }

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}