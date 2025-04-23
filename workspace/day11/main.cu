#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TILE_WIDTH 16

__global__ void tiledMatrixMulSync(const float *A, const float *B, float *C, int width) {
    // Shared memory allocation for tiles of A and B
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    // Compute the row and column indices for the output matrix C
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float value = 0.0f;

    // Loop over the tiles required to compute C[row][col]
    for (int m = 0; m < (width + TILE_WIDTH - 1) / TILE_WIDTH; m++) {
        // Load tile from matrix A into shared memory
        if (row < width && m * TILE_WIDTH + threadIdx.x < width)
            tileA[threadIdx.y][threadIdx.x] = A[row * width + m * TILE_WIDTH + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile from matrix B into shared memory
        if (col < width && m * TILE_WIDTH + threadIdx.y < width)
            tileB[threadIdx.y][threadIdx.x] = B[(m * TILE_WIDTH + threadIdx.y) * width + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        // Synchronize threads to ensure complete tile loading
        __syncthreads();

        // Multiply the tiles together
        for (int k = 0; k < TILE_WIDTH; k++)
            value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        // Synchronize threads before loading new tiles
        __syncthreads();
    }

    // Write the computed value to global memory
    if (row < width && col < width)
        C[row * width + col] = value;
}

void initializeMatrix(float *mat, int width) {
    for (int i = 0; i < width * width; i++) {
        mat[i] = (float)(rand() % 100) / 100.0f;
    }
}

int main() {
    int width = 1024;  // Define matrix dimensions: width x width
    size_t size = width * width * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize matrices with random values
    srand(time(NULL));
    initializeMatrix(h_A, width);
    initializeMatrix(h_B, width);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaError_t err;
    err = cudaMalloc((void**)&d_A, size);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_A error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }
    err = cudaMalloc((void**)&d_B, size);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_B error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }
    err = cudaMalloc((void**)&d_C, size);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc d_C error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }

    // Copy matrices A and B from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Configure grid and block dimensions
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((width + TILE_WIDTH - 1) / TILE_WIDTH, (width + TILE_WIDTH - 1) / TILE_WIDTH);

    // Set up CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event and launch the kernel
    cudaEventRecord(start);
    tiledMatrixMulSync<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Synchronized tiled matrix multiplication time: %f ms\n", milliseconds);

    // Copy result matrix C back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // (Optional) Validate results here...

    // Clean up device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Clean up host memory
    free(h_A);
    free(h_B);
    free(h_C);

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}