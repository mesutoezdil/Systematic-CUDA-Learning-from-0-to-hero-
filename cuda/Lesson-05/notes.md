# Lesson 05: The CUDA Platform Stack

The CUDA Toolkit is not just a compiler and a few API calls. It is a layered platform: programming languages at the top, hardware features at the bottom, and development and deployment tools in between. The diagram covers the full stack as shipped with CUDA Toolkit 13.x targeting Blackwell.

## Programming languages

Four languages can target CUDA directly. CUDA C/C++ is the closest to the hardware and the most common path for writing kernels. OpenACC uses compiler directives to offload loops to the GPU without writing explicit kernels. CUDA Fortran does the same for Fortran codebases. Python targets CUDA through libraries like CuPy, PyCUDA, and Numba, and is the dominant language for AI workloads.

## Development tools

Three tools ship with the toolkit for finding performance problems and bugs. Nsight Systems is a timeline profiler: it records CPU and GPU activity side by side and shows where time is spent across the full application. Nsight Compute is a kernel-level profiler: it collects hardware counters for a specific kernel and reports memory throughput, warp efficiency, and occupancy. Compute Sanitizer detects out-of-bounds memory access, race conditions, and uninitialized memory reads inside kernels at runtime.

## Compiler toolchain

`nvcc` is the NVIDIA CUDA compiler driver. It splits each source file into host code, compiled by the system C++ compiler, and device code, compiled by the NVIDIA backend. Device code compiles first to PTX (Parallel Thread Execution), a virtual instruction set that is not tied to any specific GPU generation. PTX is then assembled into SASS (Streaming ASSembler), the native machine code for a specific architecture. Shipping PTX inside a binary lets the driver JIT-compile it for any future GPU without recompiling the application.

## Hardware capabilities

The hardware layer lists features available on Blackwell SMs. MIG (Multi-Instance GPU) partitions one physical GPU into up to seven isolated instances, each with its own SM slice and memory bandwidth. Tensor Cores are dedicated matrix-multiply units, 5th generation in Blackwell, accelerating FP8, FP16, BF16, and FP32 matrix operations at much higher throughput than regular FP32 cores. Dynamic Parallelism lets a kernel launch other kernels directly from the GPU without returning control to the CPU. GPU Direct allows GPUs to transfer data to each other or to a network adapter without routing through system memory or the CPU.

## AI framework layer

The bottom layer is a set of GPU-accelerated libraries that AI and HPC workloads build on top of. cuDNN provides optimized primitives for deep learning: convolutions, attention, normalization, and activation functions. PyTorch, TensorFlow, and JAX all call cuDNN under the hood. TensorRT is an inference optimizer: it takes a trained model, fuses layers, quantizes weights, and produces a low-latency engine tuned for a target GPU. NCCL (NVIDIA Collective Communications Library) handles multi-GPU and multi-node communication patterns like all-reduce and broadcast, which are the building blocks of distributed training.

## Visual

![CUDA Platform Stack](05.png)

## Glossary

- PTX (Parallel Thread Execution): virtual instruction set that CUDA device code compiles to first. Architecture-independent. The GPU driver compiles it to SASS at install time or first run.
- SASS (Streaming ASSembler): native machine code for a specific GPU architecture. The final output of the CUDA compilation pipeline.
- `nvcc`: NVIDIA CUDA compiler driver. Separates host and device code, calls the system C++ compiler for host code, and the NVIDIA backend for device code.
- Nsight Systems: timeline profiler. Shows CPU and GPU activity together across the full application.
- Nsight Compute: kernel-level profiler. Measures hardware counters inside a single kernel invocation.
- Compute Sanitizer: runtime tool that detects memory errors inside kernels. Replaces the older `cuda-memcheck`.
- MIG (Multi-Instance GPU): hardware feature that partitions one GPU into isolated compute instances. Available on A100, H100, and B100.
- Tensor Core: dedicated matrix-multiply unit inside each SM. Operates on tiles of FP8/FP16/BF16/FP32 data per clock cycle. Much faster than regular FP32 cores for matrix math.
- Dynamic Parallelism: the ability for a kernel running on the GPU to launch another kernel without returning to the CPU.
- GPU Direct: technology that lets GPUs transfer data directly to each other or to a network card, bypassing the CPU and system memory.
- NCCL: library for collective GPU communication. Used in distributed training for all-reduce, broadcast, and reduce-scatter operations.
- cuDNN: GPU-accelerated deep learning primitives. Called by PyTorch, TensorFlow, and JAX for convolutions, attention, and similar operations.
- TensorRT: NVIDIA inference optimizer. Takes a trained model and compiles it into a low-latency engine for a specific GPU.
