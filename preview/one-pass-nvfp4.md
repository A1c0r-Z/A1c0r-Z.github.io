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

Blackwell's FP4 tensor-core instruction takes its inputs in one specific form: 4-bit E2M1 values, plus one 8-bit E4M3 scale for every 16 consecutive values along the reduction dimension.

Four bits are very coarse. An E2M1 value is one of ±{0, 0.5, 1, 1.5, 2, 3, 4, 6}, so the largest magnitude is only 12 times the smallest nonzero one. The block scale stretches this grid over each group of 16 values. Every block gets a grid that fits its own range, and an outlier only costs precision inside its own block.

<figure class="fig">
<div class="fig-scroll">
{% include figs/e2m1-grid.svg %}
</div>
<figcaption>Quantizing one block: divide by the block scale so that the block's largest magnitude lands at 6, then round each value to the nearest E2M1 value. Eight of the 16 values are shown.</figcaption>
</figure>

The block scale changes along the reduction dimension, so it cannot be pulled out of the dot product and applied afterwards. The tensor core applies it inside the instruction: every 16-element partial dot product is multiplied by the two block scales before it is accumulated.

$$
y_{mn} \;=\; \sum_{b} s^{x}_{m,b}\, s^{w}_{n,b} \sum_{k \in b} q^{x}_{m,k}\, q^{w}_{n,k}
$$

E4M3 scales have three mantissa bits, so they are much finer than power-of-two scales (the MXFP4 choice). The price is range. They only span about 17.8 binades, from 2⁻⁹ to 448, and a real tensor can sit anywhere. So NVFP4 adds a second, FP32 scale that slides this window to where the data are.

This second scale is constant along the reduction dimension, so it factors out of the dot product and is applied after the GEMM. The tensor core never sees it. It could be one per row, but the usual choice, and the case this post is about, is one per tensor.

<figure class="fig">
<div class="fig-scroll">
{% include figs/nvfp4-layout.svg %}
</div>
<figcaption>The two levels of NVFP4 scaling. The hardware format is the E2M1 codes and E4M3 block scales. The tensor scale lives in software.</figcaption>
</figure>

The hardware fixes the format. How to choose the scales is up to software. The standard dynamic recipe, used in NVIDIA's NVFP4 training recipe and in the benchmark's reference, is:

```
amax = max |x|                      # over the whole tensor
g    = 6 * 448 / amax               # tensor scale (encode direction)
s_b  = E4M3( max|x_b| / 6 * g )     # one scale per 16-value block
q    = E2M1( x * g / s_b )          # the 4-bit codes
```

A value then decodes as $$\hat{x} = q \cdot s_b / g$$. The tensor scale $$g$$ sends the largest value in the tensor to 6 × 448, the largest number NVFP4 can express, and each block scale sends its own block's maximum to about 6.

Notice the first line: nothing below it can start until the whole tensor has been read.
