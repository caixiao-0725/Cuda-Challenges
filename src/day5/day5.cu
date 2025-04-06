#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>

#define CHECK_CUDA_ERROR(val) check_cuda_error((val), #val, __FILE__, __LINE__)
void check_cuda_error(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error = %d at %s:%d: '%s'\n", 
                static_cast<unsigned int>(result), file, line, func);
        exit(EXIT_FAILURE);
    }
}

__global__ void noDivergenceKernel(float *A, float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        float a = A[idx];
        float b = B[idx];
        C[idx] = a * (a > 0 ? 1.0f : -1.0f) + b;
    }
}

__global__ void divergenceKernel(float *A, float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        if(A[idx] > 0) {
            C[idx] = A[idx] + B[idx];
        } else {
            C[idx] = A[idx] - B[idx];
        }
    }
}


void calculateStats(const std::vector<float>& times, float& mean, float& stddev, float& min, float& max) {
    mean = std::accumulate(times.begin(), times.end(), 0.0f) / times.size();
    float sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0f);
    stddev = std::sqrt(sq_sum / times.size() - mean * mean);
    min = *std::min_element(times.begin(), times.end());
    max = *std::max_element(times.begin(), times.end());
}

int main() {
    const int N = 1024 * 1024;  
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    const int numIterations = 10000; 

    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    
    for(int i = 0; i < N; i++) {
        h_A[i] = (i % 2 == 0) ? 1.0f : -1.0f;  
        h_B[i] = static_cast<float>(i);
    }

    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, N * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

  
    std::vector<float> noDivergenceTimes;
    std::vector<float> divergenceTimes;

   
    printf("Running benchmarks for %d iterations...\n", numIterations);
    
    for(int i = 0; i < numIterations; i++) {
        cudaEvent_t start, stop;
        CHECK_CUDA_ERROR(cudaEventCreate(&start));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop));
        
        CHECK_CUDA_ERROR(cudaEventRecord(start));
        noDivergenceKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
        noDivergenceTimes.push_back(milliseconds);
        
        CHECK_CUDA_ERROR(cudaEventDestroy(start));
        CHECK_CUDA_ERROR(cudaEventDestroy(stop));

        CHECK_CUDA_ERROR(cudaEventCreate(&start));
        CHECK_CUDA_ERROR(cudaEventCreate(&stop));
        
        CHECK_CUDA_ERROR(cudaEventRecord(start));
        divergenceKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        CHECK_CUDA_ERROR(cudaEventRecord(stop));
        CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
        
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
        divergenceTimes.push_back(milliseconds);
        
        CHECK_CUDA_ERROR(cudaEventDestroy(start));
        CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    }

    float mean, stddev, min, max;

    printf("\nNo Divergence Kernel Statistics:\n");
    calculateStats(noDivergenceTimes, mean, stddev, min, max);
    printf("Mean: %.3f ms\n", mean);
    printf("Std Dev: %.3f ms\n", stddev);
    printf("Min: %.3f ms\n", min);
    printf("Max: %.3f ms\n", max);

    printf("\nDivergence Kernel Statistics:\n");
    calculateStats(divergenceTimes, mean, stddev, min, max);
    printf("Mean: %.3f ms\n", mean);
    printf("Std Dev: %.3f ms\n", stddev);
    printf("Min: %.3f ms\n", min);
    printf("Max: %.3f ms\n", max);

    float noDivMean = std::accumulate(noDivergenceTimes.begin(), noDivergenceTimes.end(), 0.0f) / noDivergenceTimes.size();
    float divMean = std::accumulate(divergenceTimes.begin(), divergenceTimes.end(), 0.0f) / divergenceTimes.size();
    printf("\nSpeedup (No Divergence vs Divergence): %.2fx\n", divMean / noDivMean);

    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
