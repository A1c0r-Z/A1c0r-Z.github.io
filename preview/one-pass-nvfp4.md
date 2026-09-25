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

{% include interactive/pow2-anim.html %}

UE4M3 has three kinds of values:

$$
\begin{aligned}
\mathrm{UE4M3} &= \{0\} \cup S \cup N, \\
S &= \{\, m \cdot 2^{-9} \;:\; m = 1, \dots, 7 \,\}, \\
N &= \{\, (1 + \tfrac{i}{8})\, 2^{e} \;:\; i = 0, \dots, 7,\ e = -6, \dots, 8 \,\} \cap [0, 448].
\end{aligned}
$$

N keeps three mantissa bits; S has a fixed step of $$2^{-9}$$. Write $$y = (1+f)\,2^{E}$$ with $$0 \le f < 1$$, and let R keep three mantissa bits:

$$
R\big((1+f)\,2^{E}\big) = \big(1 + \lfloor f \rceil_{3}\big)\,2^{E},
$$

where $$\lfloor f \rceil_{3}$$ rounds f to a multiple of 1/8, ties to even. On $$[2^{-6}, 448]$$, $$\mathrm{UE4M3}(y) = R(y)$$. For a block with $$v_b = \max\lvert x_b\rvert/6 = (1+f_b)\,2^{E_b}$$ and $$g = 2^k$$, $$v_b\,g \in [2^{-6}, 448]$$:

$$
\begin{aligned}
s_b &= \mathrm{UE4M3}(v_b\,g) = R(v_b\,g) = \big(1+\lfloor f_b \rceil_3\big)\,2^{E_b+k} = g\,R(v_b), \\
q &= \mathrm{E2M1}(x\,g/s_b) = \mathrm{E2M1}\big(x/R(v_b)\big).
\end{aligned}
$$

So the 4-bit values depend on the block alone, and k only sets the exponent of the stored byte. A general g would change the mantissa to $$(1+f_b)(1+f_g)$$.

Taking $$k = \lfloor \log_2(2688/\mathrm{amax}) \rfloor$$ keeps $$v_b\,g \le 448$$. Only blocks with $$\max\lvert x_b\rvert < \mathrm{amax}/14{,}000$$ can fall below $$2^{-6}$$ into S, where this fails.

<figure class="fig">
<div class="fig-scroll">
{% include figs/nvfp4-kernels.svg %}
</div>
<figcaption><span class="fig-num">Figure 3</span>With an exact tensor scale, quantize depends on amax (orange), so x is read twice. With a power-of-two scale, quantize and the amax reduction run side by side in one pass over x; only a small finalize, which adds k to the scale exponents and redoes the few blocks that fall into S, waits for amax.</figcaption>
</figure>

## Results

**Exactness.** The one-pass quantizer is compared byte for byte with the two-pass quantizer that uses the same $$g = 2^k$$.

<figure class="tbl-fig">
<div class="fig-scroll">
<table class="tbl">
<thead><tr><th>Input</th><th>Implementation</th><th>Blocks in S</th><th>Mismatched bytes</th></tr></thead>
<tbody>
<tr><td>Qwen2.5-0.5B, every Linear input<span class="sub">4 × 1024 WikiText-2 tokens</span></td><td>PyTorch emulation</td><td>0.05%</td><td>0</td></tr>
<tr><td>Qwen2.5-1.5B, every Linear input<span class="sub">4 × 1024 WikiText-2 tokens</span></td><td>PyTorch emulation</td><td>1.8%</td><td>0</td></tr>
<tr><td>Gaussian, 2048 × 8192</td><td>B200 kernel</td><td>0%</td><td>0</td></tr>
<tr><td>Planted channel outliers, 2048 × 8192</td><td>B200 kernel</td><td>0.004%</td><td>0</td></tr>
<tr><td>Lognormal row scales, 2048 × 8192</td><td>B200 kernel</td><td>12.4%</td><td>0</td></tr>
</tbody>
</table>
</div>
<figcaption><span class="fig-num">Table 2</span>A negative control that perturbs the provisional scale of 1% of blocks by one mantissa step is caught in every trial (1352/1352 and 1576/1576). In Qwen2.5-1.5B the blocks in S concentrate in the <code>down_proj</code> inputs of layers 1 and 2 (63% and 42%).</figcaption>
</figure>

**Accuracy.** Every Linear input is fake-quantized (activations only), and the models are evaluated on WikiText-2, 240 windows of 1024 tokens.

<figure class="tbl-fig">
<div class="fig-scroll">
<table class="tbl">
<thead><tr><th></th><th>Qwen2.5-0.5B</th><th>Qwen2.5-1.5B</th></tr></thead>
<tbody>
<tr><td>Perplexity, bf16</td><td>14.80</td><td>10.47</td></tr>
<tr><td>Perplexity, standard NVFP4</td><td>16.83</td><td>11.53</td></tr>
<tr><td>Perplexity, power-of-two</td><td>16.85</td><td>11.52</td></tr>
<tr><td>KL to bf16, power-of-two / standard</td><td>1.011</td><td>1.008</td></tr>
<tr><td>ΔNLL, power-of-two − standard</td><td>+0.0017 ± 0.0015</td><td>−0.0007 ± 0.0013</td></tr>
</tbody>
</table>
</div>
<figcaption><span class="fig-num">Table 3</span>Both KL increases are significant (paired 95% bootstrap CI); neither ΔNLL is. ± is one standard error.</figcaption>
</figure>

**Speed.** A standalone quantizer on B200, M × 8192 bf16 input; L2 flushed between runs, runs interleaved, CUPTI timing. The two-pass baseline is an amax kernel and a quantize kernel chained with programmatic dependent launch.

<figure class="tbl-fig">
<div class="fig-scroll">
<table class="tbl">
<thead><tr><th>M</th><th>Two-pass <span class="unit">(µs)</span></th><th>One-pass <span class="unit">(µs)</span></th><th>Change</th></tr></thead>
<tbody>
<tr><td>1024</td><td>11.74</td><td>10.20</td><td>−13%</td></tr>
<tr><td>2048</td><td>18.41</td><td>14.14</td><td>−23%</td></tr>
<tr><td>4096</td><td>31.12</td><td>23.48</td><td>−25%</td></tr>
<tr><td>8192</td><td>55.45</td><td>40.95</td><td>−26%</td></tr>
<tr><td>16384</td><td>103.22</td><td>79.78</td><td>−23%</td></tr>
</tbody>
</table>
</div>
<figcaption><span class="fig-num">Table 4</span>Gaussian input, so no block takes the fix-up path: this is the best case. A bandwidth model that counts the provisional scales staged between the two kernels bounds the gain at about 38%.</figcaption>
</figure>
