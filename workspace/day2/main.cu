#include <cuda_runtime.h>
#include "kernels.cuh"

int main() {
    dummyKernel << <2, 4 >> > ();
    cudaDeviceSynchronize();
    return 0;
}