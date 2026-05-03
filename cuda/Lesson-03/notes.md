# Lesson 03: Compute Capability

Compute capability is a version number that identifies the feature set and hardware limits of a GPU. It determines what instructions are available, how much shared memory each SM gets, and what the register limits are. Every CUDA feature has a minimum compute capability requirement.

## What the number means

The format is major.minor, for example 9.0 for Hopper. A higher major version means a new architecture generation with new hardware. A higher minor version is a revision within the same generation. Code compiled for CC 7.0 runs on any GPU with CC 7.0 or higher. Code that uses features from CC 9.0 will not run on older hardware.

## GPU generation comparison

| Spec                   | P100 (CC 6.0)     | V100 (CC 7.0)     | A100 (CC 8.0)     | H100 (CC 9.0)     | B100 (CC 10.0)    |
|------------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
| GPU                    | Tesla P100        | Tesla V100        | A100              | H100              | B100              |
| Codename               | GP100             | GV100             | GA100             | GH100             | GB100             |
| Architecture           | Pascal            | Volta             | Ampere            | Hopper            | Blackwell         |
| Threads / Warp         | 32                | 32                | 32                | 32                | 32                |
| Max Warps / SM         | 64                | 64                | 64                | 64                | 64                |
| Max Threads / SM       | 2048              | 2048              | 2048              | 2048              | 2048              |
| Max Thread Blocks / SM | 32                | 32                | 32                | 32                | 32                |
| Max Registers / SM     | 65536             | 65536             | 65536             | 65536             | 65536             |
| Max Registers / Block  | 65536             | 65536             | 65536             | 65536             | 65536             |
| Max Registers / Thread | 255               | 255               | 255               | 255               | 255               |
| Max Thread Block Size  | 1024              | 1024              | 1024              | 1024              | 1024              |
| FP32 Cores / SM        | 64                | 64                | 64                | 128               | 128               |
| Shared Memory / SM     | 64 KB             | up to 96 KB       | up to 164 KB      | up to 228 KB      | up to 228 KB      |

Source: NVIDIA CUDA Programming Guide, Blackwell Tuning Guide, Hopper Tuning Guide (CUDA Toolkit 13.2, 2025-2026).

H100 and B100 share the same basic SM threading and memory limits. Blackwell's gains over Hopper are in total SM count (192 vs 132 for H100 SXM5), 5th generation Tensor Cores, HBM3e bandwidth, and NVLink 5.0, not in per-SM thread or register counts.

## What each spec means

Threads/Warp is fixed at 32 across all generations. The GPU always processes threads in groups of 32. This constraint has not changed since Fermi (CC 2.0) and shapes how every kernel is written.

Max Warps/SM and Max Threads/SM set how many threads can be alive on one SM at once. With 64 warps at 32 threads each, the cap is 2048 threads per SM. More active threads give the scheduler more work to hide memory latency.

Max Thread Block Size stays at 1024 across all generations. A single block can never exceed 1024 threads. This comes from the SM register file: one block must fit entirely on one SM, and the SM has a fixed register budget.

FP32 Cores/SM went from 64 (Pascal through Ampere) to 128 (Hopper and Blackwell). More FP32 cores per SM means more parallel floating-point operations per clock cycle per SM.

Shared Memory/SM grew from 64 KB (Pascal) to 228 KB (Hopper and Blackwell). Shared memory is fast on-chip memory that all threads in a block can read and write. Larger shared memory lets kernels work with bigger data chunks without going to global memory.

## Visual

```
Compute Capability progression (data center GPUs):

  Pascal    Volta    Ampere    Hopper   Blackwell
  CC 6.0   CC 7.0   CC 8.0   CC 9.0   CC 10.0
    |        |        |        |          |
    +--------+--------+--------+-----------+
FP32/SM:   64       64       64      128       128
Shmem/SM:  64K      96K     164K     228K      228K

Specs that have NOT changed since CC 6.0:
  - Threads per warp:    32
  - Max warps per SM:    64
  - Max threads per SM:  2048
  - Max registers per SM: 65536
  - Max thread block size: 1024
```

## Glossary

- compute capability: a version number (major.minor) that describes which CUDA features a GPU supports and what its hardware limits are.
- SM (Streaming Multiprocessor): the physical processor unit inside the GPU. All thread execution happens on SMs.
- warp: a group of 32 threads the GPU schedules and executes together.
- FP32 core: a hardware unit that executes one 32-bit floating-point operation per clock cycle.
- shared memory: fast on-chip memory inside each SM, shared by all threads in a block. Much faster than global (device) memory.
- register file: a pool of fast storage per SM used to hold thread-local variables. 65536 registers per SM across all generations shown.
- CUDA Toolkit 12.8+: required to compile code targeting Blackwell (CC 10.0).
