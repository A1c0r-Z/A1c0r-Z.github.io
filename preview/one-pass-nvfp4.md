---
layout: post
title: "One-pass dynamic NVFP4 quantization (working title)"
date: 2026-09-24
permalink: /draft/one-pass-nvfp4/
draft: true
math: true
sitemap: false
---

Recently I worked on one problem from [SOL-ExecBench](https://research.nvidia.com/benchmarks/sol-execbench), NVIDIA's benchmark that scores GPU kernels against their speed-of-light bounds: [kernel #196](https://research.nvidia.com/benchmarks/sol-execbench/kernel/196), an NVFP4 linear layer on B200. Each call takes bf16 activations, quantizes them to NVFP4 on the fly, and multiplies them with pre-quantized FP4 weights.

## How NVFP4 works

On B200, NVFP4 matrix multiplies run on the tensor-core instruction [`tcgen05.mma`](https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-block-scaling), in its block-scaled FP4 mode (`.kind::mxf4nvf4`, `.block16`). Both operands are E2M1 values in ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}, and every 16 consecutive elements along K share one UE4M3 scale, which the tensor core applies during accumulation.

<figure class="fig">
<div class="fig-scroll">
{% include figs/nvfp4-mma.svg %}
</div>
<figcaption><span class="fig-num">Figure 1</span>Each pair of blocks along K adds <em>a</em>·<em>b</em> × (its 16-element dot product) to the FP32 accumulator.</figcaption>
</figure>

A UE4M3 value lies between $$2^{-9}$$ and 448, about 17.8 binades. A block's scale should be about $$\max\lvert x_b\rvert/6$$, where $$x_b$$ are its 16 input values, and for a real tensor that can be far outside this range: gradients are often much smaller, and activations with outliers can be larger. NVFP4 therefore pairs the block scales with a second, FP32 factor, which multiplies them before they are rounded to UE4M3 and slides them into range.

<figure class="fig">
<div class="fig-scroll">
{% include figs/nvfp4-ranges.svg %}
</div>
<figcaption><span class="fig-num">Figure 2</span>A block's raw scale can fall anywhere in a range of about 261 binades; UE4M3 covers 17.8 of them. The tensor scale chooses which.</figcaption>
</figure>

This factor is handled by software, not by the tensor core. It only has to be constant along K, so it factors out of the sum in Figure 1 and the GEMM epilogue applies it to the output. The standard choice is one per tensor, the case this post is about; per-row variants also exist.
