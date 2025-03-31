#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void vectorAddCPU(float *A, float *B, float *C, int N) {
    for(int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 100000000; // 100 million
    float *A = (float*)malloc(N * sizeof(float));
    float *B = (float*)malloc(N * sizeof(float));
    float *C = (float*)malloc(N * sizeof(float));

    // Initialize vectors
    for(int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    clock_t start = clock();
    vectorAddCPU(A, B, C, N);
    clock_t end = clock();

    printf("CPU Vector Add Time: %lf seconds\n", ((double)(end - start))/CLOCKS_PER_SEC);

    free(A);
    free(B);
    free(C);
    return 0;
}