#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

void vectorAddCPU(float* A, float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

__global__ void vectorAddGPU(float* A, float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int N = 100000000; // 100 million
    float* A = (float*)malloc(N * sizeof(float));
    float* B = (float*)malloc(N * sizeof(float));
    float* C = (float*)malloc(N * sizeof(float));

    // Initialize vectors
    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    clock_t CPU_start = clock();
    vectorAddCPU(A, B, C, N);
    clock_t CPU_end = clock();

    printf("CPU Vector Add Time: %lf seconds\n", ((double)(CPU_end - CPU_start)) / CLOCKS_PER_SEC);

    free(A);
    free(B);
    free(C);




    size_t size = N * sizeof(float);
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialize vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float* d_A, * d_B, * d_C;
    cudaError_t err;

    err = cudaMalloc(&d_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for A: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc(&d_B, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for B: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc(&d_C, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for C: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    clock_t GPU_start = clock();
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAddGPU << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, N);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        free(h_A);
        free(h_B);
        free(h_C);
        exit(EXIT_FAILURE);
    }
    clock_t GPU_end = clock();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("GPU Vector Add Time: %lf seconds\n", ((double)(GPU_end - GPU_start)) / CLOCKS_PER_SEC);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;

}