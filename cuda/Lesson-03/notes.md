# Lesson 03: Compute Capability

Compute capability is a version number that identifies the feature set and hardware limits of a GPU. It determines what instructions are available, how much shared memory each SM gets, and what the register limits are. Every CUDA feature has a minimum compute capability requirement.

## What the number means

The format is major.minor — for example, 9.0 for Hopper. A higher major version means a new architecture generation with new hardware. A higher minor version is a revision within the same generation. Code compiled for CC 7.0 runs on any GPU with CC 7.0 or higher. Code that uses features from CC 9.0 will not run on older hardware.

## GPU generation comparison

| Spec                      | P100 (CC 6.0)       | V100 (CC 7.0)       | A100 (CC 8.0)        | H100 (CC 9.0)        |
|---------------------------|---------------------|---------------------|----------------------|----------------------|
| GPU                       | NVIDIA Tesla P100   | NVIDIA Tesla V100   | NVIDIA A100          | NVIDIA H100          |
| Codename                  | GP100               | GV100               | GA100                | GH100                |
| Architecture              | Pascal              | Volta               | Ampere               | Hopper               |
| Threads / Warp            | 32                  | 32                  | 32                   | 32                   |
| Max Warps / SM            | 64                  | 64                  | 64                   | 64                   |
| Max Threads / SM          | 2048                | 2048                | 2048                 | 2048                 |
| Max Thread Blocks / SM    | 32                  | 32                  | 32                   | 32                   |
| Max Registers / SM        | 65536               | 65536               | 65536                | 65536                |
| Max Registers / Block     | 65536               | 65536               | 65536                | 65536                |
| Max Registers / Thread    | 255                 | 255                 | 255                  | 255                  |
| Max Thread Block Size     | 1024                | 1024                | 1024                 | 1024                 |
| FP32 Cores / SM           | 64                  | 64                  | 64                   | 128                  |
| Shared Memory / SM        | 64 KB               | up to 96 KB         | up to 164 KB         | up to 228 KB         |

Blackwell (B100/B200, CC 10.0) is the current data center generation as of 2026. It increases FP32 throughput and shared memory per SM beyond Hopper, and requires CUDA Toolkit 12.8 or later.

## What each spec means

Threads/Warp is fixed at 32 across all generations. The GPU always processes threads in groups of 32. This has not changed since Fermi (CC 2.0) and shapes how every CUDA kernel is written.

Max Warps/SM and Max Threads/SM set how many threads can be alive on a single SM at once. With 64 warps at 32 threads each, the cap is 2048 threads per SM. More active threads give the scheduler more work to hide memory latency.

Max Thread Block Size stays at 1024 regardless of compute capability. A single block can never exceed 1024 threads. This limit comes from the SM register file: one block must fit entirely on one SM.

FP32 Cores/SM doubled from 64 to 128 starting with Hopper. More FP32 cores per SM means more floating-point operations per clock cycle per SM. Older data center GPUs (Pascal through Ampere) kept this at 64.

Shared Memory/SM grew from 64 KB (Pascal) to 228 KB (Hopper). Shared memory is fast on-chip memory that all threads in a block can read and write. Larger shared memory lets kernels work with bigger data chunks without going to slow global memory.

## Querying at runtime

`cudaGetDeviceProperties()` fills a `cudaDeviceProp` struct with every spec for the device. The `major` and `minor` fields give the compute capability. The struct also holds SM count, max threads per block, shared memory size, register counts, and more. This is how your code can adapt to whatever GPU it runs on.

## Code

```c
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
```

## Compile and run

```bash
nvcc first_kernel.cu -o first_kernel
./first_kernel
```

Output on an H100 SXM5:

```
Device name:           NVIDIA H100 80GB HBM3
Compute capability:    9.0
Multiprocessors (SMs): 132
Warp size:             32
Max threads per block: 1024
Max threads per SM:    2048
Registers per SM:      65536
Shared memory per SM:  233472 bytes
```

Output varies by GPU. `233472 bytes` is 228 KB.

## Visual

```
cudaDeviceProp struct
+------------------------------------------+
|  name:                 "NVIDIA H100 ..." |
|  major / minor:        9  /  0           |
|  multiProcessorCount:  132               |
|  warpSize:             32                |
|  maxThreadsPerBlock:   1024              |
|  maxThreadsPerMP:      2048              |
|  regsPerMultiprocessor:65536             |
|  sharedMemPerMP:       233472 bytes      |
+------------------------------------------+

Compute Capability progression (data center GPUs):

  Pascal    Volta     Ampere    Hopper    Blackwell
  CC 6.0    CC 7.0    CC 8.0    CC 9.0    CC 10.0
    |         |         |         |          |
    +----+----+----+----+----+----+----+-----+
         FP32/SM:  64   64   64   128    ...
         Shmem/SM: 64K  96K  164K 228K   ...
```

## Glossary

- compute capability: a version number (e.g. 9.0) that describes the feature set and hardware limits of a GPU. Higher means newer and more capable.
- `cudaDeviceProp`: a struct in the CUDA runtime API that holds all properties of a GPU device. Filled by calling `cudaGetDeviceProperties()`.
- `cudaGetDeviceProperties(&prop, n)`: fills the `prop` struct with specs for device number `n`. Device 0 is the first GPU in the system.
- shared memory: fast on-chip memory inside each SM. All threads in a block share it. Much faster than global memory but limited in size.
- FP32 core: a hardware unit that executes one 32-bit floating-point operation per clock cycle. More FP32 cores per SM means more parallel compute.
- SM count: the number of streaming multiprocessors on the GPU. Total GPU throughput scales roughly with SM count.
