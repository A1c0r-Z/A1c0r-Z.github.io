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

A UE4M3 value lies between $$2^{-9}$$ and 448, about 17.8 binades. A block's scale should be about $$\max\lvert x_b\rvert/6$$, where $$x_b$$ are its 16 input values, and for a real tensor that can be far outside this range. NVFP4 therefore pairs the block scales with a second, FP32 factor, which multiplies them before they are rounded to UE4M3 and slides them into range.

<figure class="fig">
<div class="fig-scroll">
{% include figs/nvfp4-ranges.svg %}
</div>
<figcaption><span class="fig-num">Figure 2</span>A block's raw scale can fall anywhere in a range of about 261 binades; UE4M3 covers 17.8 of them. The tensor scale chooses which.</figcaption>
</figure>

This factor is handled by software, not by the tensor core. It only has to be constant along K, so it factors out of the sum in Figure 1 and the GEMM epilogue applies it to the output. The standard choice is one per tensor; Table 1 lists the common variants, and this post is about the dynamic per-tensor one.

{% include tables/tensor-scales.html %}

The standard dynamic recipe derives everything from the tensor's largest magnitude. It is what NVIDIA's NVFP4 training recipe uses for activations, and it is also the algorithm of [kernel #196](https://research.nvidia.com/benchmarks/sol-execbench/kernel/196): the benchmark's reference quantizes x exactly this way.

```
amax = max |x|                     # over the whole tensor
g    = 6 * 448 / amax              # tensor scale
s_b  = UE4M3( max|x_b| / 6 * g )   # stored block scale
q    = E2M1( x * g / s_b )         # 4-bit values
```

## A power-of-two tensor scale

Write a positive number as $$y = (1+f)\,2^{E}$$ with $$0 \le f < 1$$, and let R keep three mantissa bits:

$$
R\big((1+f)\,2^{E}\big) = \big(1 + \lfloor f \rceil_{3}\big)\,2^{E},
$$

where $$\lfloor f \rceil_{3}$$ rounds f to the nearest multiple of 1/8, ties to even. On the normal range $$[2^{-6}, 448]$$, R is exactly rounding to UE4M3. Now take a block with $$v_b = \max\lvert x_b\rvert/6 = (1+f_b)\,2^{E_b}$$ and a power-of-two tensor scale $$g = 2^k$$:

$$
\begin{aligned}
s_b &= R(v_b\,g) = \big(1+\lfloor f_b \rceil_3\big)\,2^{E_b+k} = g\,R(v_b), \\
q &= \mathrm{E2M1}(x\,g/s_b) = \mathrm{E2M1}\big(x/R(v_b)\big).
\end{aligned}
$$

Nothing on the right depends on k: the 4-bit values and the dequantized values $$\hat{x} = q\,R(v_b)$$ are fixed by the block alone, and k only enters the exponent of the stored byte. For a general $$g = (1+f_g)\,2^{E_g}$$, the mantissa of $$v_b\,g$$ is $$(1+f_b)(1+f_g)$$, whose rounding depends on $$f_g$$.

This needs $$v_b\,g \in [2^{-6}, 448]$$. Choosing $$k = \lfloor \log_2(448/v_{\max}) \rfloor$$, with $$v_{\max} = \mathrm{amax}/6$$, gives $$v_b\,g \le 448$$ for every block, so the top edge is never crossed. Below $$2^{-6}$$, UE4M3 is subnormal and rounds to a fixed step of $$2^{-9}$$ instead of three mantissa bits, so R no longer applies and $$s_b$$ depends on k. Since $$g > 224/v_{\max}$$, only blocks with $$\max\lvert x_b\rvert < \mathrm{amax}\cdot 2^{-6}/224 \approx \mathrm{amax}/14{,}000$$ can land there.
