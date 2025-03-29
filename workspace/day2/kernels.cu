#include <stdio.h>
#include "kernels.cuh"

// A dummy kernel that prints block and thread indices
__global__ void dummyKernel() {
    printf("Block %d, Thread %d\n", blockIdx.x, threadIdx.x);
}