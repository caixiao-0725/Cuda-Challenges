#include <cuda_runtime.h>
#include "kernel.h"

int main() {
    dummyKernel<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}