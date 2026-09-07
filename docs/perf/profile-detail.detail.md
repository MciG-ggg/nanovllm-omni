# Per-stage kernel breakdown (/tmp/profiler-originals/profile-detail.trace.json)

| stage | wall_us | n_steps | kernel_count | total_kernel_us | top_kernel | top_kernel_us |
| --- | --- | --- | --- | --- | --- | --- |
| tokenize | 2143.8 | 0 | 0 | 0.0 | (none) | - |
| generate | 1288974.4 | 15 | 25757 | 391260.5 | cudaLaunchKernel | 192496.5 |
| decode | 60545.1 | 0 | 1009 | 17772.2 | cudaLaunchKernel | 10624.8 |
| wav | 1648.1 | 0 | 0 | 0.0 | (none) | - |


### tokenize  (wall=2143.8us, kernels=0)

_no kernels captured_

### generate  (wall=1288974.4us, kernels=25757)

| kernel | total_us | count |
| --- | ---: | ---: |
| cudaLaunchKernel | 192496.5 | 9689 |
| void gemv2T_kernel_val<int, int, __half, __half, __half, float, 128, 16, 4, 4, false, false, cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float> >(cublasGemvParamsEx<int, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half const>, cublasGemvTensorStridedBatched<__half>, float>, float, float) | 33542.3 | 180 |
| cudaMemcpyAsync | 29546.3 | 1215 |
| void cutlass::Kernel2<cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x64_32x6_tn_align8>(cutlass_80_tensorop_f16_s16816gemm_relu_f16_64x64_32x6_tn_align8::Params) | 25252.8 | 480 |
| cudaStreamSynchronize | 17550.5 | 327 |

### decode  (wall=60545.1us, kernels=1009)

| kernel | total_us | count |
| --- | ---: | ---: |
| cudaLaunchKernel | 10624.8 | 389 |
| cudaStreamSynchronize | 1710.1 | 23 |
| cudaMemcpyAsync | 1580.8 | 28 |
| void cutlass::Kernel2<cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_tn_align8>(cutlass_80_wmma_tensorop_f16_s161616gemm_f16_32x32_128x2_tn_align8::Params) | 425.9 | 40 |
| void cudnn::engines_precompiled::nchwToNhwcKernel<__half, __half, float, false, true, (cudnnKernelDataType_t)0>(cudnn::engines_precompiled::nchw2nhwc_params_t<float>, __half const*, __half*) | 366.0 | 12 |

### wav  (wall=1648.1us, kernels=0)

_no kernels captured_

