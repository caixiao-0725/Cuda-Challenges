#include <cuda_runtime.h>
#include <stdio.h>

// Kernel: Multiply each element by a scale factor
__global__ void scaleVector(const float *input, float *output, float scale, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        output[idx] = input[idx] * scale;
    }
}

int main() {
    int N = 1024;                   // Number of elements in the vector
    size_t size = N * sizeof(float);  // Total size in bytes

    // Allocate host memory
    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);

    // Initialize the host array
    for (int i = 0; i < N; i++) {
        h_input[i] = i * 1.0f;
    }

    // Allocate device (global) memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    // Transfer input data from host to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Configure kernel execution parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel to scale the vector by 2.0
    scaleVector<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, 2.0f, N);

    // Synchronize to ensure kernel completion
    cudaDeviceSynchronize();

    // Transfer the result back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Verify and print the first 10 results
    for (int i = 0; i < 10; i++) {
        printf("h_output[%d] = %f\n", i, h_output[i]);
    }

    // Free device memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Free host memory
    free(h_input);
    free(h_output);

    return 0;
}