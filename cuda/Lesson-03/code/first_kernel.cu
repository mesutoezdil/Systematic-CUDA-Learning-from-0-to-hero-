#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

int main()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("Device name:           %s\n",   prop.name);
    printf("Compute capability:    %d.%d\n", prop.major, prop.minor);
    printf("Multiprocessors (SMs): %d\n",   prop.multiProcessorCount);
    printf("Warp size:             %d\n",   prop.warpSize);
    printf("Max threads per block: %d\n",   prop.maxThreadsPerBlock);
    printf("Max threads per SM:    %d\n",   prop.maxThreadsPerMultiProcessor);
    printf("Registers per SM:      %d\n",   prop.regsPerMultiprocessor);
    printf("Shared memory per SM:  %zu bytes\n", prop.sharedMemPerMultiprocessor);

    return 0;
}
