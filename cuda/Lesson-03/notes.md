# Lesson 03: Compute Capability

Lessons 00 through 02 introduced threads, warps, blocks, and the 1024-thread block limit. Compute capability is the version number that defines where those limits come from. It identifies the feature set and hardware boundaries of a GPU generation, and every CUDA feature has a minimum compute capability requirement.

## What the number means

The format is major.minor, for example 9.0 for Hopper. A higher major version means a new architecture generation with new hardware. A higher minor version is a revision within the same generation. Code compiled for CC 7.0 runs on any GPU with CC 7.0 or higher. Code that uses features from CC 9.0 will not run on older hardware.

## GPU generations

The table below covers data center GPUs from Pascal through Blackwell. Specs come from the NVIDIA CUDA Programming Guide, Blackwell Tuning Guide, and Hopper Tuning Guide (CUDA Toolkit 13.2, 2025-2026).

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

H100 and B100 share the same per-SM threading and memory limits. Blackwell's gains over Hopper are in total SM count (192 vs 132 for H100 SXM5), 5th generation Tensor Cores, HBM3e bandwidth, and NVLink 5.0, not in per-SM thread or register counts.

## Threads per warp

Lesson-01 introduced the warp as the group of 32 threads the GPU processes together. That number is not arbitrary — it is fixed by the hardware and encoded in the compute capability spec. The GPU never schedules individual threads. It always schedules warps of 32 at a time, and this has not changed since Fermi (CC 2.0).

## Warps and threads per SM

Lesson-02 introduced the SM as the physical processor that blocks run on. Each SM can hold up to 64 active warps at once, which works out to 2048 threads. More active warps give the warp scheduler more choices when some warps stall waiting on memory, which keeps the execution units busy.

## Thread block size limit

Lesson-02 showed that `<<<1, 2048>>>` compiles without error but launches nothing at runtime. That silent failure happens because the max thread block size is 1024, a hard limit from the compute capability spec. A block must fit entirely on one SM, and the SM has a fixed register budget that caps how large one block can be.

## FP32 cores per SM

Pascal, Volta, and Ampere each have 64 FP32 cores per SM. Hopper and Blackwell have 128. More FP32 cores per SM means more floating-point operations complete per clock cycle on each SM.

## Shared memory per SM

Shared memory capacity grew across generations: 64 KB on Pascal, up to 96 KB on Volta, up to 164 KB on Ampere, and up to 228 KB on Hopper and Blackwell. Shared memory sits on-chip inside each SM and is accessible by all threads in a block. Larger capacity lets kernels keep bigger working sets on-chip without going to global memory.

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
  - Threads per warp:     32
  - Max warps per SM:     64
  - Max threads per SM:   2048
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
