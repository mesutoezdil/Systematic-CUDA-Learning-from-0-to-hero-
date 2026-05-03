# Lesson 01: One Block, Four Threads

One change from Lesson 00: the thread count goes from 1 to 4. The block count stays at 1. Four threads run the same kernel simultaneously, each with a different `threadIdx.x`.

---

## What changes

```c
printIDs<<<1, 4>>>();
//          ^  ^
//  blocks -+  +- threads per block (was 1, now 4)
```

The GPU creates 4 copies of `printIDs` and runs them at the same time. Each copy gets a unique `threadIdx.x`: 0, 1, 2, or 3. `blockIdx.x` stays 0 for all of them because there is still only one block.

---

## SIMT: Single Instruction, Multiple Threads

Each thread is an independent execution unit. They do not wait for each other and do not coordinate. They all execute the exact same instructions at the same time, but with different ID values. This model is called SIMT: Single Instruction, Multiple Threads.

---

## Warps

The GPU processes threads in groups of 32 called warps. A warp is the actual scheduling unit at the hardware level, not a block. When you launch 4 threads, the GPU creates a full warp of 32 lanes but activates only 4 of them. If threads in a warp take different branches (if/else), the GPU serializes the diverging paths, which is called warp divergence. With 4 identical threads here there is no divergence.

---

## Why output order changes every run

`printf` inside a kernel does not print immediately. Each thread writes into a shared circular buffer in GPU memory. The buffer flushes when `cudaDeviceSynchronize()` is called. The order in which threads write to the buffer is not guaranteed, even within a single warp. Output order changes between runs.

---

## Code

```c
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void printIDs()
{
    printf("\nBlock ID: %d  ===  Thread ID: %d", blockIdx.x, threadIdx.x);
}

int main()
{
    printIDs<<<1, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

---

## Compile and run

```bash
nvcc first_kernel.cu -o first_kernel
./first_kernel
```

Output (4 lines, `blockIdx.x` always 0, `threadIdx.x` 0-3 in some order):

```
Block ID: 0  ===  Thread ID: 2
Block ID: 0  ===  Thread ID: 0
Block ID: 0  ===  Thread ID: 3
Block ID: 0  ===  Thread ID: 1
```

---

## Visual

```
printIDs<<<1, 4>>>
              |  |
    blocks ---+  +--- threads per block

GPU Grid
+-------------------------------------------------------------+
|  Block 0  (blockIdx=0)                                      |
|  +----------+ +----------+ +----------+ +----------+        |
|  | Thread 0 | | Thread 1 | | Thread 2 | | Thread 3 |        |
|  | tIdx=0   | | tIdx=1   | | tIdx=2   | | tIdx=3   |        |
|  +----------+ +----------+ +----------+ +----------+        |
|                                                             |
|  all 4 fit in one warp (warp size = 32)                     |
+-------------------------------------------------------------+
```
