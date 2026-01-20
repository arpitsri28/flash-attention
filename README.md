# FlashAttention Kernel 1 (CUDA)

This project implements a FlashAttention-style forward kernel (Kernel 1). It includes a CPU reference for correctness checks and a small test runner.

## Build

From the repository root:

```
cmake -S flash_attention_kernel1 -B flash_attention_kernel1/build
cmake --build flash_attention_kernel1/build -j
```

If CMake cannot find CUDA, pass `-DCUDAToolkit_ROOT=/path/to/cuda` or ensure `nvcc` is on your `PATH`.

## Run

The test executable runs two cases by default: `seq_len=64` and `seq_len=128` for FP16 and BF16.

```
flash_attention_kernel1/build/flash_test
```

Optional flags:

- `--seq_len <N>`: run a single sequence length.
- `--dtype <fp16|bf16|all>`: control which dtype(s) run.

Example:

```
flash_attention_kernel1/build/flash_test --seq_len 64 --dtype fp16
```

The test prints `max_abs_err`, `mean_abs_err`, and `rmse` and exits non-zero if thresholds are exceeded.

## Notes

- Default target architecture is `sm80` (set in `CMakeLists.txt`).
- The kernel expects `d_head=128`, `B_r=64`, and `B_c=64`.
