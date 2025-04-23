#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Define tile width (adjust according to hardware capabilities)
#define TILE_WIDTH 16

__global__ void tiledMatrixMul(const float *A, const float *B, float *C, int width) {
    // Allocate shared memory for tiles of A and B
    __shared__ float tileA[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tileB[TILE_WIDTH][TILE_WIDTH];

    // Calculate row and column indices of the element of C computed by this thread
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    float Pvalue = 0.0f;

    // Loop over all tiles needed to compute C[row][col]
    // 'm' indexes the tiles along the shared dimension
    for (int m = 0; m < (width + TILE_WIDTH - 1) / TILE_WIDTH; m++) {
        // Load tile from matrix A into shared memory if within bounds
        if (row < width && (m * TILE_WIDTH + threadIdx.x) < width) {
            tileA[threadIdx.y][threadIdx.x] = A[row * width + m * TILE_WIDTH + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        // Load tile from matrix B into shared memory if within bounds
        if (col < width && (m * TILE_WIDTH + threadIdx.y) < width) {
            tileB[threadIdx.y][threadIdx.x] = B[(m * TILE_WIDTH + threadIdx.y) * width + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Synchronize to ensure all data is loaded into shared memory
        __syncthreads();

        // Perform multiplication for the tile and accumulate the result
        for (int k = 0; k < TILE_WIDTH; k++) {
            Pvalue += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        // Synchronize to make sure computation is done before loading new tiles
        __syncthreads();
    }

    // Write the result to global memory if within bounds
    if (row < width && col < width) {
        C[row * width + col] = Pvalue;
    }
}

void initializeMatrix(float *mat, int width) {
    for (int i = 0; i < width * width; i++) {
        mat[i] = (float)(rand() % 100) / 100.0f;
    }
}

int main() {
    // Set matrix dimensions (square matrix)
    int width = 1024; // Example: 1024 x 1024 matrix
    size_t size = width * width * sizeof(float);

    // Allocate host memory for matrices A, B, and C
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host memory allocation failed!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize matrices A and B with random values
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

    // Copy matrices A and B to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Configure kernel launch parameters
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((width + TILE_WIDTH - 1) / TILE_WIDTH, (width + TILE_WIDTH - 1) / TILE_WIDTH);

    // Launch the tiled matrix multiplication kernel
    tiledMatrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, width);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }

    // Copy result matrix C back to host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // (Optional) Validate results here...

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Tile-based matrix multiplication completed successfully!\n");
    return 0;
}