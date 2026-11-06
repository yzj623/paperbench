# Stochastic-interpolants：人工 rubric-assisted checklist

> 用于诊断高质量清单的效果；使用了 rubric，不能作为 rubric-blind 方法的结果。完整来源/冲突说明见 AUDIT.md。

13 个任务，59 个检查项；58 个 CodeDev 叶节点均有审阅位置，其中 5 项有实质来源冲突，不能称为完全满足 rubric。

保留普通 RA 的任务 ID、名称、顺序；修正三处描述，逐字差异见 coverage.json。每项检查须追踪实际代码调用路径，不以注释或未接线 helper 代替实现。

## task_001 — Project scaffolding and configuration management

Create the project structure with configuration files, dependency specifications (requirements.txt), and a central configuration module. The configuration module should define all hyperparameters, model architecture parameters, training settings, dataset paths, and experiment-specific settings as described in the paper and addendum. This includes batch size, gradient steps, learning rate, optimizer settings, U-Net architecture parameters (dimension multipliers, channels, resnet block groups, attention heads, sinusoidal conditioning), interpolation coefficients, and task-specific parameters for inpainting and super-resolution experiments.

### task_001_check_001

Provide importable, version-compatible dependencies for the referenced U-Net, PyTorch, torchdiffeq, HuggingFace ImageNet access and FID. The actual velocity-model import must resolve to the intended implementation, not a similarly named placeholder or a silent tiny-network fallback.

来源性质：`paper_or_addendum`

**paper.md**

```text
Architecture For the velocity model we use the U-net from (Ho et al., 2020b) as implemented in lucidrain's denoising-diffusion-pytorch repository; this variant of the architecture includes embeddings to condition on class labels. We use the following hyperparameters:
- Dim Mults: (1,1,2,3,4)
- Dim (channels): 256
- Resnet block groups: 8
- Leanred Sinusoidal Cond: True
- Learned Sinusoidal Dim: 32
- Attention Dim Head: 64
- Attention Heads: 4
- Random Fourier Features: False
```

**paper.md**

```text
Integration for sampling We use the Dopri solver from the torchdiffeq library (Chen, 2018).
```

**addendum.md**

```text
You should download ImageNet using HuggingFace. Below is some example code of how you can do this. You should use use `trust_remote_code=True` if you want to avoid the code waiting for stdin:
```python
from datasets import load_dataset
dataset = load_dataset("imagenet-1k", trust_remote_code=True)
```
```

### task_001_check_002

Provide explicit production configurations for dependent inpainting, uncoupled inpainting and dependent super-resolution. Every production run uses batch size 32 and 200000 optimizer updates; small/debug settings must be opt-in and must not replace production defaults. Check configuration consumption, not merely the presence of constants.

来源性质：`paper_or_addendum`

**addendum.md**

```text
All experiments should use a batch size of 32.
```

**addendum.md**

```text
All experiments should run for 200,000 gradient steps
```

**addendum.md**

```text
You do not have to replicate the baseline results for Improved DDPM, SR3, ADM, Cascaded Diffusion, or I^2SB.
```

### task_001_check_003

Separate image state, observation mask, fixed low-resolution condition, class label and time in the data/model interfaces. Document their shapes and the xi=1 observed / xi=0 missing convention so training and sampling cannot silently swap mask meanings or evolve conditioning channels.

来源性质：`paper_or_addendum`

**paper.md**

```text
For simplicity, the mask takes the same value for all channels in a given spatial location in the image. We define the base density by the relation $x_{0}=\xi \circ x_{1}+(1-\xi) \circ \zeta$, where $\circ$ denotes the Hadamard (elementwise) product and $\zeta \in \mathbb{R}^{C \times W \times H}, \zeta \sim \mathrm{~N}(0, I d)$ denotes random noise used to initialize the pixels within the masked region (separate noise for each channel). During
training, the mask is drawn randomly by tiling the image into 64 tiles; each tile is selected to enter the mask with probability $p=0.3$.
```

**paper.md**

```text
Image-shaped conditioning in the Unet. For image-shaped conditioning, we follow (Ho et al., 2022a) and append upsampled low-resolution images to the input $x_{t}$ at each time step to the velocity model. We also condition on the missingness masks for in-painting by appending them to $x_{t}$.
```

**paper.md**

```text
tion 4.1. We denote by $\mathcal{D}: \mathbb{R}^{C \times W \times H} \rightarrow \mathbb{R}^{C \times W_{\text {low }} \times H_{\text {low }}}$ and $\mathcal{U}: \mathbb{R}^{C \times W_{\text {low }} \times H_{\text {low }}} \rightarrow \mathbb{R}^{C \times W \times H}$ image downsampling and upsampling operations, where $W_{\text {low }}$ and $H_{\text {low }}$ denote the width and height of a low-resolution image. To define the base density, we then set $x_{0}=\mathcal{U}\left(\mathcal{D}\left(x_{1}\right)\right)+\sigma \zeta$ with $\zeta \in \mathbb{R}^{C \times W \times H}, \zeta \sim \mathrm{~N}(0, I d)$, and $\sigma>0$. Defining $x_{0}$ in this way frames the transport problem such that each starting pixel is proximal to its intended target. Notice in particular that, with $\sigma=0$, each $x_{0}$ would correspond to a lowerdimensional sample embedded in a higher-dimensional space, and the corresponding distribution would be concentrated on a lower-dimensional manifold. Working with $\sigma>0$ alleviates the associated singularities by adding a small amount of Gaussian noise to smooth the base density so it is well-defined over the entire higher-dimensional ambient space. In addition, we give the model access to the lowresolution image at all times; this problem setting then corresponds to using $\rho\left(x_{0}, x_{1} \mid \xi\right)=\rho_{1}\left(x_{1}\right) \rho_{0}\left(x_{0} \mid x_{1}, \xi\right)$ with $\xi=\mathcal{U}\left(\mathcal{D}\left(x_{1}\right)\right)$. In the experiments, we set $\rho_{1}$ to correspond to ImageNet ( 256 or 512 ), following prior work (Saharia et al., 2022; Ho et al., 2022a).
```

### task_001_check_004

This is a rubric-assisted CodeDev diagnostic: implement the full production paths, but do not require executing training or 50000-image sampling to complete a task. Implementation checks must follow callable code and configuration; prose, stub functions and cached claimed metrics are not implementations.

来源性质：`paper_or_addendum`

**instructions.txt**

```text
Your submission will be graded based on whether the code necessary for reproducing the paper has been correctly implemented. The code will not be executed during grading.
```

## task_002 — ImageNet dataset loading and preprocessing pipeline

Implement the ImageNet dataset loader using HuggingFace datasets library as specified in the addendum. Create data loading utilities that provide batches of high-resolution images from ImageNet. Include image normalization, resizing to target resolutions, and efficient batching. Implement both train and validation split handling. Include utility functions for computing FID scores against reference sets. This module provides the foundational data access layer used by all experiments.

### task_002_check_001

Implement authenticated HuggingFace imagenet-1k loading with genuine train and validation split access and labels preserved. Follow the addendum remote-code handling when applicable to the selected compatible datasets version; a missing dataset must not silently select synthetic data or the wrong split.

来源性质：`paper_or_addendum`

**addendum.md**

```text
You should download ImageNet using HuggingFace. Below is some example code of how you can do this. You should use use `trust_remote_code=True` if you want to avoid the code waiting for stdin:
```python
from datasets import load_dataset
dataset = load_dataset("imagenet-1k", trust_remote_code=True)
```
```

### task_002_check_002

Provide RGB ImageNet preprocessing at both 256x256 and 512x512, with explicit consistent normalization and inverse conversion for visualization/FID. Keep the initial ImageNet resolution preparation distinct from the later task-specific low-resolution degradation.

来源性质：`paper_or_addendum`

**paper.md**

```text
Figure 3: Image inpainting: ImageNet- $256 \times 256$ and ImageNet- $512 \times 512$. Top panels: Six examples of image in-filling at resolution $256 \times 256$, where the left columns display masked images, the center corresponds to in-filled model samples, and the right shows full reference images. The aims are not to recover the precise content of the reference image, but instead, to provide a conditionally valid in-filling. Bottom panels: Four examples at resolution $512 \times 512$.
![](assets/asset_4.jpg)
```

**paper.md**

```text
Figure 4: Super-resolution: Top four rows: Super-resolved images from resolution $64 \times 64 \mapsto 256 \times 256$, where the left-most image is the lower resolution version, the middle is the model output, and the right is the ground truth. Examples for $256 \times 256 \mapsto 512 \times 512$ are given in Fig. 6 .
```

**paper.md**

```text
Figure 6: Super-resolution: Top four rows: Super-resolved images from resolution $256 \times 256 \mapsto 512 \times 512$, where the left-most image is the lower resolution version, the middle is the model output, and the right is the ground truth.
```

### task_002_check_003

Training receives training images; held-out dependent sampling receives validation images with their actual observations and labels. Split, sample identity, mask and class must stay aligned through batching; validation must never silently alias the training set.

来源性质：`rubric_assisted_protocol`

**addendum.md**

```text
You should download ImageNet using HuggingFace. Below is some example code of how you can do this. You should use use `trust_remote_code=True` if you want to avoid the code waiting for stdin:
```python
from datasets import load_dataset
dataset = load_dataset("imagenet-1k", trust_remote_code=True)
```
```

**paper.md**

```text
Figure 3: Image inpainting: ImageNet- $256 \times 256$ and ImageNet- $512 \times 512$. Top panels: Six examples of image in-filling at resolution $256 \times 256$, where the left columns display masked images, the center corresponds to in-filled model samples, and the right shows full reference images. The aims are not to recover the precise content of the reference image, but instead, to provide a conditionally valid in-filling. Bottom panels: Four examples at resolution $512 \times 512$.
![](assets/asset_4.jpg)
```

## task_003 — Stochastic interpolant construction and time-sampling utilities

Implement the core stochastic interpolant framework. Create classes/functions for constructing the interpolated state given base samples, target samples, noise, and time. Implement configurable interpolant coefficient schedules with boundary conditions alpha_0=beta_1=1, alpha_1=beta_0=gamma_0=gamma_1=0. Implement uniform time sampling from U(0,1). Compute the time derivative analytically from the coefficient derivatives. Provide utilities for evaluating the empirical loss function as specified in Algorithm 1. Support both independent coupling (standard baseline) and dependent coupling modes.

### task_003_check_001

Construct the general interpolant I(t)=alpha(t)*x0+beta(t)*x1+gamma(t)*z with analytic derivatives and Definition 3.1 endpoint conditions. Broadcast a separate time per sample without mixing batch and image axes; any general noise term must be independent of the coupled endpoints.

来源性质：`paper_or_addendum`

**paper.md**

```text
Definition 3.1 (Stochastic interpolant with coupling). The stochastic interpolant $I_{t}$ is the process defined as ${ }^{1}$
\[
I_{t}=\alpha_{t} x_{0}+\beta_{t} x_{1}+\gamma_{t} z \quad t \in[0,1]
\]
where
- $\alpha_{t}, \beta_{t}$, and $\gamma_{t}^{2}$ are differentiable functions of time such that $\alpha_{0}=\beta_{1}=1, \alpha_{1}=\beta_{0}=\gamma_{0}=\gamma_{1}=0$, and $\alpha_{t}^{2}+\beta_{t}^{2}+\gamma_{t}^{2}>0$ for all $t \in[0,1]$.
- The pair $\left(x_{0}, x_{1}\right)$ is jointly drawn from a probability density $\rho\left(x_{0}, x_{1}\right)$ with finite second moments and such that
\[
\begin{aligned}
& \int_{\mathbb{R}^{d}} \rho\left(x_{0}, x_{1}\right) d x_{1}=\rho_{0}\left(x_{0}\right) \\
& \int_{\mathbb{R}^{d}} \rho\left(x_{0}, x_{1}\right) d x_{0}=\rho_{1}\left(x_{1}\right)
\end{aligned}
\]
- $z \sim \mathrm{~N}(0$, Id $)$, independent of $\left(x_{0}, x_{1}\right)$.
```

**paper.md**

```text
Algorithm 1 Training
    Input: Interpolant coefficients $\alpha_{t}, \beta_{t}$; velocity model $\hat{b}$;
    batch size $n_{b}$;
    repeat
        for $i=1, \ldots, n_{\mathrm{b}}$ do
            Draw $x_{1}^{i} \sim \rho_{1}\left(x_{1}\right), \zeta_{i} \sim \mathcal{N}(0, I d), t_{i} \sim U(0,1)$.
            Compute $x_{0}^{i}=m\left(x_{1}^{i}\right)+\sigma \zeta^{i}$.
            Compute $I_{t_{i}}=\alpha_{t_{i}} x_{0}^{i}+\beta_{t_{i}} x_{1}^{i}$.
        end for
        Compute empirical loss
        $\hat{L}_{b}(\hat{b})=n_{\mathrm{b}}^{-1} \sum_{i=1}^{n_{\mathrm{b}}}\left[\left|\hat{b}_{t_{i}}\left(I_{t_{i}}\right)\right|^{2}-2 \dot{I}_{t_{i}} \cdot \hat{b}_{t_{i}}\left(I_{t_{i}}\right)\right]$.
        Take gradient step on $\hat{L}_{b}(\hat{b})$ to update $\hat{b}$.
    until converged
    Return: Velocity $\hat{b}$.
```

### task_003_check_002

For the image experiments use gamma=0 and a coherent forward convention: x0 is the base, x1 the target, I(t)=(1-t)*x0+t*x1, dI/dt=x1-x0, and integrate from t=0 to t=1. Section 4.1 instead prints alpha=t,beta=1-t, inconsistent with Definition 3.1 and Algorithm 2. A wholly consistent reversed-time reparameterization is acceptable, but mixing its state formula with the forward derivative/sampler is not. Document the conflict rather than enforcing contradictory formulas.

来源性质：`source_conflict_resolution`

**paper.md**

```text
Definition 3.1 (Stochastic interpolant with coupling). The stochastic interpolant $I_{t}$ is the process defined as ${ }^{1}$
\[
I_{t}=\alpha_{t} x_{0}+\beta_{t} x_{1}+\gamma_{t} z \quad t \in[0,1]
\]
where
- $\alpha_{t}, \beta_{t}$, and $\gamma_{t}^{2}$ are differentiable functions of time such that $\alpha_{0}=\beta_{1}=1, \alpha_{1}=\beta_{0}=\gamma_{0}=\gamma_{1}=0$, and $\alpha_{t}^{2}+\beta_{t}^{2}+\gamma_{t}^{2}>0$ for all $t \in[0,1]$.
- The pair $\left(x_{0}, x_{1}\right)$ is jointly drawn from a probability density $\rho\left(x_{0}, x_{1}\right)$ with finite second moments and such that
\[
\begin{aligned}
& \int_{\mathbb{R}^{d}} \rho\left(x_{0}, x_{1}\right) d x_{1}=\rho_{0}\left(x_{0}\right) \\
& \int_{\mathbb{R}^{d}} \rho\left(x_{0}, x_{1}\right) d x_{0}=\rho_{1}\left(x_{1}\right)
\end{aligned}
\]
- $z \sim \mathrm{~N}(0$, Id $)$, independent of $\left(x_{0}, x_{1}\right)$.
```

**paper.md**

```text
The model sees the mask; we note that we do not need to additionally input the partial image as extra conditioning because it is present, uncorrupted, in $x_{t}$ for each $t$ because the values are present in $x_{0}$ and $x_{1}$. In the interpolant (20), we set $\alpha_{t}=t$ and $\beta_{t}=1-t$. In this setup, the velocity field $b_{t}(x, \xi)$ is such that $b_{t}(x, \xi)=0$ except in the masked regions. This follows because $\xi \circ I_{t}=\xi \circ x_{1}$ for every $t$, i.e., the unmasked pixels in $I_{t}$ are always those of $x_{1}$ for which $\dot{I}_{t}=0$. To take this structural information into account, we can build this property into our neural network model, and mask the output of the approximate velocity field to enforce that the unmasked pixels remain fixed. We note that this method does not necessitate any inference time corrections, such as the replacement method or MCMC.
```

**paper.md**

```text
Algorithm 2 Sampling (via forward Euler method)
    Input: model $\hat{b}$, corrupted sample $m\left(x_{1}\right), N \in \mathbb{N}$.
    Draw noise $\zeta \sim \mathcal{N}(0, I d)$
    Initialize $\hat{X}_{0}=m\left(x_{1}\right)+\sigma \zeta$
    for $n=0, \ldots, N-1$ do
        $\hat{X}_{i+1}=\hat{X}_{i}+N^{-1} \hat{b}_{i / N}\left(\hat{X}_{i}\right)$
    end for
    Return: clean sample $\hat{X}_{N}$.
```

**paper.md**

```text
In this context, we can use the interpolant from (1) with $\gamma_{t}=0$, which reduces to:
\[
I_{t}=\alpha_{t}\left(m\left(x_{1}\right)+\sigma \zeta\right)+\beta_{t} x_{1}
\]

Note that the score associated to (20) is still available because of the factor of $\sigma \zeta$, so long as $\sigma$ is invertible.
```

### task_003_check_003

Uniformly sample t_i in [0,1] independently for every minibatch element in all three training modes. The actual mode-specific training calls must use this continuous-time draw, not one fixed time, a shared batch scalar or diffusion-index sampling.

来源性质：`paper_or_addendum`

**paper.md**

```text
Algorithm 1 Training
    Input: Interpolant coefficients $\alpha_{t}, \beta_{t}$; velocity model $\hat{b}$;
    batch size $n_{b}$;
    repeat
        for $i=1, \ldots, n_{\mathrm{b}}$ do
            Draw $x_{1}^{i} \sim \rho_{1}\left(x_{1}\right), \zeta_{i} \sim \mathcal{N}(0, I d), t_{i} \sim U(0,1)$.
            Compute $x_{0}^{i}=m\left(x_{1}^{i}\right)+\sigma \zeta^{i}$.
            Compute $I_{t_{i}}=\alpha_{t_{i}} x_{0}^{i}+\beta_{t_{i}} x_{1}^{i}$.
        end for
        Compute empirical loss
        $\hat{L}_{b}(\hat{b})=n_{\mathrm{b}}^{-1} \sum_{i=1}^{n_{\mathrm{b}}}\left[\left|\hat{b}_{t_{i}}\left(I_{t_{i}}\right)\right|^{2}-2 \dot{I}_{t_{i}} \cdot \hat{b}_{t_{i}}\left(I_{t_{i}}\right)\right]$.
        Take gradient step on $\hat{L}_{b}(\hat{b})$ to update $\hat{b}$.
    until converged
    Return: Velocity $\hat{b}$.
```

### task_003_check_004

Image-experiment noise belongs to the construction of x0; do not also add an unrequested nonzero gamma(t)*z to the linear experimental bridge. Dependent and independent endpoint constructors remain distinct and are selected by the actual experiment mode.

来源性质：`paper_or_addendum`

**paper.md**

```text
In this context, we can use the interpolant from (1) with $\gamma_{t}=0$, which reduces to:
\[
I_{t}=\alpha_{t}\left(m\left(x_{1}\right)+\sigma \zeta\right)+\beta_{t} x_{1}
\]

Note that the score associated to (20) is still available because of the factor of $\sigma \zeta$, so long as $\sigma$ is invertible.
```

**paper.md**

```text
Algorithm 1 Training
    Input: Interpolant coefficients $\alpha_{t}, \beta_{t}$; velocity model $\hat{b}$;
    batch size $n_{b}$;
    repeat
        for $i=1, \ldots, n_{\mathrm{b}}$ do
            Draw $x_{1}^{i} \sim \rho_{1}\left(x_{1}\right), \zeta_{i} \sim \mathcal{N}(0, I d), t_{i} \sim U(0,1)$.
            Compute $x_{0}^{i}=m\left(x_{1}^{i}\right)+\sigma \zeta^{i}$.
            Compute $I_{t_{i}}=\alpha_{t_{i}} x_{0}^{i}+\beta_{t_{i}} x_{1}^{i}$.
        end for
        Compute empirical loss
        $\hat{L}_{b}(\hat{b})=n_{\mathrm{b}}^{-1} \sum_{i=1}^{n_{\mathrm{b}}}\left[\left|\hat{b}_{t_{i}}\left(I_{t_{i}}\right)\right|^{2}-2 \dot{I}_{t_{i}} \cdot \hat{b}_{t_{i}}\left(I_{t_{i}}\right)\right]$.
        Take gradient step on $\hat{L}_{b}(\hat{b})$ to update $\hat{b}$.
    until converged
    Return: Velocity $\hat{b}$.
```

**paper.md**

```text
Table 2: FID for Inpainting Task. FID comparison between under two paradigms: a baseline, where $\rho_{0}$ is a Gaussian with independent coupling to $\rho_{1}$, and our datadependent coupling detailed in Section 4.1.
\begin{tabular}{ll}
\hline Model & FID-50k \\
\hline Uncoupled Interpolant (Baseline) & 1.35 \\
Dependent Coupling (Ours) & $\mathbf{1 . 1 3}$ \\
\hline
\end{tabular}
```

## task_004 — U-Net velocity field model architecture

Implement the U-Net based velocity field model following the architecture specification from Appendix B. The model takes as input the interpolated state at time t, the time step via learned sinusoidal positional embeddings, and optional conditioning information such as class labels, masks, or low-resolution images appended as extra channels. Architecture details include dimension multipliers, channel dimensions, residual block groups with GroupNorm, learned sinusoidal time conditioning, attention mechanism with specified head count and dimension, and class embedding support. The model outputs a velocity prediction matching the spatial dimensions of the input. Follow the denoising-diffusion-pytorch variant referenced in Appendix B.

### task_004_check_001

Use the Appendix B lucidrain denoising-diffusion-pytorch U-Net velocity architecture, or a faithful explicit implementation of that variant, with its multiscale residual/attention/skip structure. A generic smaller CNN, similarly named U-Net or diffusion noise-predictor wrapper is not a substitute for the stated velocity model.

来源性质：`paper_or_addendum`

**paper.md**

```text
Architecture For the velocity model we use the U-net from (Ho et al., 2020b) as implemented in lucidrain's denoising-diffusion-pytorch repository; this variant of the architecture includes embeddings to condition on class labels. We use the following hyperparameters:
- Dim Mults: (1,1,2,3,4)
- Dim (channels): 256
- Resnet block groups: 8
- Leanred Sinusoidal Cond: True
- Learned Sinusoidal Dim: 32
- Attention Dim Head: 64
- Attention Heads: 4
- Random Fourier Features: False
```

### task_004_check_002

Instantiate the production U-Net with dim=256 and dim_mults=(1,1,2,3,4), not repository defaults. Verify the instantiated channel hierarchy, not only config declarations.

来源性质：`paper_or_addendum`

**paper.md**

```text
Architecture For the velocity model we use the U-net from (Ho et al., 2020b) as implemented in lucidrain's denoising-diffusion-pytorch repository; this variant of the architecture includes embeddings to condition on class labels. We use the following hyperparameters:
- Dim Mults: (1,1,2,3,4)
- Dim (channels): 256
- Resnet block groups: 8
- Leanred Sinusoidal Cond: True
- Learned Sinusoidal Dim: 32
- Attention Dim Head: 64
- Attention Heads: 4
- Random Fourier Features: False
```

### task_004_check_003

Use eight GroupNorm groups in the ResNet blocks, not eight blocks or eight channels; preserve valid divisibility in all actual residual blocks.

来源性质：`paper_or_addendum`

**paper.md**

```text
Architecture For the velocity model we use the U-net from (Ho et al., 2020b) as implemented in lucidrain's denoising-diffusion-pytorch repository; this variant of the architecture includes embeddings to condition on class labels. We use the following hyperparameters:
- Dim Mults: (1,1,2,3,4)
- Dim (channels): 256
- Resnet block groups: 8
- Leanred Sinusoidal Cond: True
- Learned Sinusoidal Dim: 32
- Attention Dim Head: 64
- Attention Heads: 4
- Random Fourier Features: False
```

**addendum.md**

```text
The "Resnet block groups" that is discussed in the appendix is the number of groups that goes into the GroupNorm in the ResNet layers (their ResNet blocks use GroupNorm)
```

### task_004_check_004

Use learned sinusoidal time conditioning with learned_sinusoidal_cond=True, learned_sinusoidal_dim=32 and random_fourier_features=False. The real model forward must consume this trainable time representation.

来源性质：`paper_or_addendum`

**paper.md**

```text
Architecture For the velocity model we use the U-net from (Ho et al., 2020b) as implemented in lucidrain's denoising-diffusion-pytorch repository; this variant of the architecture includes embeddings to condition on class labels. We use the following hyperparameters:
- Dim Mults: (1,1,2,3,4)
- Dim (channels): 256
- Resnet block groups: 8
- Leanred Sinusoidal Cond: True
- Learned Sinusoidal Dim: 32
- Attention Dim Head: 64
- Attention Heads: 4
- Random Fourier Features: False
```

### task_004_check_005

Use attention head dimension 64 and four heads in the specified U-Net attention modules; configuration overrides must reach the modules.

来源性质：`paper_or_addendum`

**paper.md**

```text
Architecture For the velocity model we use the U-net from (Ho et al., 2020b) as implemented in lucidrain's denoising-diffusion-pytorch repository; this variant of the architecture includes embeddings to condition on class labels. We use the following hyperparameters:
- Dim Mults: (1,1,2,3,4)
- Dim (channels): 256
- Resnet block groups: 8
- Leanred Sinusoidal Cond: True
- Learned Sinusoidal Dim: 32
- Attention Dim Head: 64
- Attention Heads: 4
- Random Fourier Features: False
```

### task_004_check_006

Append the inpainting observation mask and the super-resolution fixed upsampled low-resolution image as spatial input channels in their respective modes at every training and sampling call. Output only C image-velocity channels with the original spatial resolution, never velocities for mask, label or low-resolution channels.

来源性质：`paper_or_addendum`

**paper.md**

```text
Image-shaped conditioning in the Unet. For image-shaped conditioning, we follow (Ho et al., 2022a) and append upsampled low-resolution images to the input $x_{t}$ at each time step to the velocity model. We also condition on the missingness masks for in-painting by appending them to $x_{t}$.
```

**paper.md**

```text
Results. Similarly to the previous experiment, we append the upsampled low-resolution images $\xi$ to the channel dimension of the input $x$ of the velocity model, and likewise include the ImageNet class labels.
```

**paper.md**

```text
The model sees the mask; we note that we do not need to additionally input the partial image as extra conditioning because it is present, uncorrupted, in $x_{t}$ for each $t$ because the values are present in $x_{0}$ and $x_{1}$. In the interpolant (20), we set $\alpha_{t}=t$ and $\beta_{t}=1-t$. In this setup, the velocity field $b_{t}(x, \xi)$ is such that $b_{t}(x, \xi)=0$ except in the masked regions. This follows because $\xi \circ I_{t}=\xi \circ x_{1}$ for every $t$, i.e., the unmasked pixels in $I_{t}$ are always those of $x_{1}$ for which $\dot{I}_{t}=0$. To take this structural information into account, we can build this property into our neural network model, and mask the output of the approximate velocity field to enforce that the unmasked pixels remain fixed. We note that this method does not necessitate any inference time corrections, such as the replacement method or MCMC.
```

### task_004_check_007

For this rubric-assisted diagnostic, retain class conditioning and also append one HxW channel uniformly filled with each sample's ImageNet class value in all class-conditioned training and sampling paths. A learned embedding alone does not meet this diagnostic channel contract. Appendix B supports embeddings; the additional constant-value channel is a rubric-specified refinement, not a verbatim requirement of Appendix B.

来源性质：`rubric_assisted_protocol`

**paper.md**

```text
Architecture For the velocity model we use the U-net from (Ho et al., 2020b) as implemented in lucidrain's denoising-diffusion-pytorch repository; this variant of the architecture includes embeddings to condition on class labels. We use the following hyperparameters:
- Dim Mults: (1,1,2,3,4)
- Dim (channels): 256
- Resnet block groups: 8
- Leanred Sinusoidal Cond: True
- Learned Sinusoidal Dim: 32
- Attention Dim Head: 64
- Attention Heads: 4
- Random Fourier Features: False
```

**paper.md**

```text
Results. Similarly to the previous experiment, we append the upsampled low-resolution images $\xi$ to the channel dimension of the input $x$ of the velocity model, and likewise include the ImageNet class labels.
```

## task_005 — Training loop for stochastic interpolants with dependent couplings

Implement the complete training procedure as specified in Algorithm 1. For each training step: sample a batch of target data points, sample base corruption according to the coupling scheme, sample random noise, sample uniform time steps, compute the interpolated states, compute the empirical loss over the batch using the velocity model predictions, perform gradient descent update on the velocity model parameters. Include gradient clipping as specified in Appendix B, Adam optimizer with specified learning rate schedule, and logging infrastructure. Handle both conditional with class labels and unconditional training modes. Track training loss curves and save model checkpoints at regular intervals throughout the full training run.

### task_005_check_001

The minibatch update must construct paired endpoints from the selected coupling, sample per-example times, evaluate the coherent interpolant and its derivative, call the actual velocity model with the matching conditions, backpropagate and update parameters. All three mode-specific training entry points must reach this update, not an unused correct helper.

来源性质：`paper_or_addendum`

**paper.md**

```text
Algorithm 1 Training
    Input: Interpolant coefficients $\alpha_{t}, \beta_{t}$; velocity model $\hat{b}$;
    batch size $n_{b}$;
    repeat
        for $i=1, \ldots, n_{\mathrm{b}}$ do
            Draw $x_{1}^{i} \sim \rho_{1}\left(x_{1}\right), \zeta_{i} \sim \mathcal{N}(0, I d), t_{i} \sim U(0,1)$.
            Compute $x_{0}^{i}=m\left(x_{1}^{i}\right)+\sigma \zeta^{i}$.
            Compute $I_{t_{i}}=\alpha_{t_{i}} x_{0}^{i}+\beta_{t_{i}} x_{1}^{i}$.
        end for
        Compute empirical loss
        $\hat{L}_{b}(\hat{b})=n_{\mathrm{b}}^{-1} \sum_{i=1}^{n_{\mathrm{b}}}\left[\left|\hat{b}_{t_{i}}\left(I_{t_{i}}\right)\right|^{2}-2 \dot{I}_{t_{i}} \cdot \hat{b}_{t_{i}}\left(I_{t_{i}}\right)\right]$.
        Take gradient step on $\hat{L}_{b}(\hat{b})$ to update $\hat{b}$.
    until converged
    Return: Velocity $\hat{b}$.
```

**paper.md**

```text
Table 2: FID for Inpainting Task. FID comparison between under two paradigms: a baseline, where $\rho_{0}$ is a Gaussian with independent coupling to $\rho_{1}$, and our datadependent coupling detailed in Section 4.1.
\begin{tabular}{ll}
\hline Model & FID-50k \\
\hline Uncoupled Interpolant (Baseline) & 1.35 \\
Dependent Coupling (Ours) & $\mathbf{1 . 1 3}$ \\
\hline
\end{tabular}
```

### task_005_check_002

Compute the empirical quadratic velocity objective mean_i[sum_pixels,channels(b_hat^2)-2*sum_pixels,channels(dI_dt*b_hat)] over image channels only. Squared-error regression is acceptable only with the same parameter gradients/reduction (its target-norm term is parameter independent); do not substitute image reconstruction, score/noise prediction, or omit the cross term. Check every training mode.

来源性质：`paper_or_addendum`

**paper.md**

```text
Algorithm 1 Training
    Input: Interpolant coefficients $\alpha_{t}, \beta_{t}$; velocity model $\hat{b}$;
    batch size $n_{b}$;
    repeat
        for $i=1, \ldots, n_{\mathrm{b}}$ do
            Draw $x_{1}^{i} \sim \rho_{1}\left(x_{1}\right), \zeta_{i} \sim \mathcal{N}(0, I d), t_{i} \sim U(0,1)$.
            Compute $x_{0}^{i}=m\left(x_{1}^{i}\right)+\sigma \zeta^{i}$.
            Compute $I_{t_{i}}=\alpha_{t_{i}} x_{0}^{i}+\beta_{t_{i}} x_{1}^{i}$.
        end for
        Compute empirical loss
        $\hat{L}_{b}(\hat{b})=n_{\mathrm{b}}^{-1} \sum_{i=1}^{n_{\mathrm{b}}}\left[\left|\hat{b}_{t_{i}}\left(I_{t_{i}}\right)\right|^{2}-2 \dot{I}_{t_{i}} \cdot \hat{b}_{t_{i}}\left(I_{t_{i}}\right)\right]$.
        Take gradient step on $\hat{L}_{b}(\hat{b})$ to update $\hat{b}$.
    until converged
    Return: Velocity $\hat{b}$.
```

### task_005_check_003

Use Adam with initial learning rate 2e-4 and zero weight decay for dependent inpainting, independent inpainting and super-resolution; AdamW defaults or weight decay on any active parameter group do not satisfy this requirement.

来源性质：`paper_or_addendum`

**paper.md**

```text
Optimization. We use Adam optimizer (Kingma \& Ba, 2014), starting at learning rate 2e-4 with the StepLR scheduler which scales the learning rate by $\gamma=.99$ every $N=1000$ steps. We use no weight decay. We clip gradient norms at 10,000 (this is the norm of the entire set of parameters taken as a vector, the default type of norm clipping in PyTorch library).
```

### task_005_check_004

StepLR must multiply the learning rate by 0.99 after every 1000 optimizer updates, across 200000 updates. Do not step it on every minibatch as if one step were 1000, on each epoch, or per accumulated microbatch; resume must preserve its update count.

来源性质：`paper_or_addendum`

**paper.md**

```text
Optimization. We use Adam optimizer (Kingma \& Ba, 2014), starting at learning rate 2e-4 with the StepLR scheduler which scales the learning rate by $\gamma=.99$ every $N=1000$ steps. We use no weight decay. We clip gradient norms at 10,000 (this is the norm of the entire set of parameters taken as a vector, the default type of norm clipping in PyTorch library).
```

**addendum.md**

```text
All experiments should run for 200,000 gradient steps
```

### task_005_check_005

Clip the global gradient norm of the entire velocity model parameter vector at 10000 before optimizer.step; do not clip parameter values, each tensor independently, or use a different threshold.

来源性质：`paper_or_addendum`

**paper.md**

```text
Optimization. We use Adam optimizer (Kingma \& Ba, 2014), starting at learning rate 2e-4 with the StepLR scheduler which scales the learning rate by $\gamma=.99$ every $N=1000$ steps. We use no weight decay. We clip gradient norms at 10,000 (this is the norm of the entire set of parameters taken as a vector, the default type of norm clipping in PyTorch library).
```

### task_005_check_006

The production loop must actually honor batch size 32 and 200000 gradient updates, cycle the dataset as needed, and save model/config/optimizer/scheduler/update-counter state so sampling loads the matching learned architecture and a resumed run does not restart its schedule.

来源性质：`paper_or_addendum`

**paper.md**

```text
Algorithm 1 Training
    Input: Interpolant coefficients $\alpha_{t}, \beta_{t}$; velocity model $\hat{b}$;
    batch size $n_{b}$;
    repeat
        for $i=1, \ldots, n_{\mathrm{b}}$ do
            Draw $x_{1}^{i} \sim \rho_{1}\left(x_{1}\right), \zeta_{i} \sim \mathcal{N}(0, I d), t_{i} \sim U(0,1)$.
            Compute $x_{0}^{i}=m\left(x_{1}^{i}\right)+\sigma \zeta^{i}$.
            Compute $I_{t_{i}}=\alpha_{t_{i}} x_{0}^{i}+\beta_{t_{i}} x_{1}^{i}$.
        end for
        Compute empirical loss
        $\hat{L}_{b}(\hat{b})=n_{\mathrm{b}}^{-1} \sum_{i=1}^{n_{\mathrm{b}}}\left[\left|\hat{b}_{t_{i}}\left(I_{t_{i}}\right)\right|^{2}-2 \dot{I}_{t_{i}} \cdot \hat{b}_{t_{i}}\left(I_{t_{i}}\right)\right]$.
        Take gradient step on $\hat{L}_{b}(\hat{b})$ to update $\hat{b}$.
    until converged
    Return: Velocity $\hat{b}$.
```

**addendum.md**

```text
All experiments should use a batch size of 32.
```

**addendum.md**

```text
All experiments should run for 200,000 gradient steps
```

## task_006 — Inpainting experiment mask generation and data-dependent coupling

Implement the image inpainting specific components. Create a mask generation system that tiles the image into tiles and randomly selects each tile to enter the mask with the specified probability. Define the inpainting coupling where x0 = xi*x1 + (1-xi)*zeta, where xi=1 denotes observed pixels and xi=0 denotes missing pixels, ensuring unmasked pixels remain unchanged across all interpolant times. Implement the structural property that the velocity field output is masked so that the velocity is zero in unmasked regions, enforcing that unmasked pixel values remain fixed during transport. The conditioning variable the mask is appended as extra channels to the model input. This module handles the specific data preparation for the inpainting experiment.

### task_006_check_001

At both training and dependent sampling, create exactly 64 equal tiles (an 8x8 grid for these square images), with independent Bernoulli choices per tile and per sample at p=0.3. Define xi as the observed-pixel indicator and document how selected tiles map to it. Never interpret 64 as tile width, which yields only 16 tiles on 256x256 images.

来源性质：`paper_or_addendum`

**paper.md**

```text
For simplicity, the mask takes the same value for all channels in a given spatial location in the image. We define the base density by the relation $x_{0}=\xi \circ x_{1}+(1-\xi) \circ \zeta$, where $\circ$ denotes the Hadamard (elementwise) product and $\zeta \in \mathbb{R}^{C \times W \times H}, \zeta \sim \mathrm{~N}(0, I d)$ denotes random noise used to initialize the pixels within the masked region (separate noise for each channel). During
training, the mask is drawn randomly by tiling the image into 64 tiles; each tile is selected to enter the mask with probability $p=0.3$.
```

### task_006_check_002

The spatial observation mask is identical across RGB channels for each sample, whereas missing-pixel Gaussian noise is independently drawn across channels and pixels. Apply the same convention in training and sampling.

来源性质：`paper_or_addendum`

**paper.md**

```text
For simplicity, the mask takes the same value for all channels in a given spatial location in the image. We define the base density by the relation $x_{0}=\xi \circ x_{1}+(1-\xi) \circ \zeta$, where $\circ$ denotes the Hadamard (elementwise) product and $\zeta \in \mathbb{R}^{C \times W \times H}, \zeta \sim \mathrm{~N}(0, I d)$ denotes random noise used to initialize the pixels within the masked region (separate noise for each channel). During
training, the mask is drawn randomly by tiling the image into 64 tiles; each tile is selected to enter the mask with probability $p=0.3$.
```

### task_006_check_003

Construct dependent training x0=xi*x1+(1-xi)*zeta from the actual ImageNet training image, with standard-normal zeta. Known pixels equal x1 exactly; missing pixels are noise. Do not use xi*xi, replace the observation with zeros, or use an entirely independent noisy image.

来源性质：`paper_or_addendum`

**paper.md**

```text
For simplicity, the mask takes the same value for all channels in a given spatial location in the image. We define the base density by the relation $x_{0}=\xi \circ x_{1}+(1-\xi) \circ \zeta$, where $\circ$ denotes the Hadamard (elementwise) product and $\zeta \in \mathbb{R}^{C \times W \times H}, \zeta \sim \mathrm{~N}(0, I d)$ denotes random noise used to initialize the pixels within the masked region (separate noise for each channel). During
training, the mask is drawn randomly by tiling the image into 64 tiles; each tile is selected to enter the mask with probability $p=0.3$.
```

### task_006_check_004

Build the dependent image interpolant so observed pixels remain exactly fixed at every time and their analytic target velocity is zero. Mask the predicted image velocity with (1-xi) during loss computation AND sampling; masking only the regression target is insufficient. No replacement/MCMC correction may substitute for this structural velocity constraint.

来源性质：`paper_or_addendum`

**paper.md**

```text
The model sees the mask; we note that we do not need to additionally input the partial image as extra conditioning because it is present, uncorrupted, in $x_{t}$ for each $t$ because the values are present in $x_{0}$ and $x_{1}$. In the interpolant (20), we set $\alpha_{t}=t$ and $\beta_{t}=1-t$. In this setup, the velocity field $b_{t}(x, \xi)$ is such that $b_{t}(x, \xi)=0$ except in the masked regions. This follows because $\xi \circ I_{t}=\xi \circ x_{1}$ for every $t$, i.e., the unmasked pixels in $I_{t}$ are always those of $x_{1}$ for which $\dot{I}_{t}=0$. To take this structural information into account, we can build this property into our neural network model, and mask the output of the approximate velocity field to enforce that the unmasked pixels remain fixed. We note that this method does not necessitate any inference time corrections, such as the replacement method or MCMC.
```

**paper.md**

```text
For simplicity, the mask takes the same value for all channels in a given spatial location in the image. We define the base density by the relation $x_{0}=\xi \circ x_{1}+(1-\xi) \circ \zeta$, where $\circ$ denotes the Hadamard (elementwise) product and $\zeta \in \mathbb{R}^{C \times W \times H}, \zeta \sim \mathrm{~N}(0, I d)$ denotes random noise used to initialize the pixels within the masked region (separate noise for each channel). During
training, the mask is drawn randomly by tiling the image into 64 tiles; each tile is selected to enter the mask with probability $p=0.3$.
```

### task_006_check_005

The dependent training model must receive the same sample's mask, class conditioning and diagnostic constant class channel, while predicting only image velocity. Mask and label channels are fixed conditions, not interpolated targets.

来源性质：`rubric_assisted_protocol`

**paper.md**

```text
Results. For implementation, we parameterize $b_{t}(x, \xi)$ using the basic U-Net architecture from (Ho et al., 2020b), where $\xi$ is given to the model as appended channels of the image $x$.
```

**paper.md**

```text
Architecture For the velocity model we use the U-net from (Ho et al., 2020b) as implemented in lucidrain's denoising-diffusion-pytorch repository; this variant of the architecture includes embeddings to condition on class labels. We use the following hyperparameters:
- Dim Mults: (1,1,2,3,4)
- Dim (channels): 256
- Resnet block groups: 8
- Leanred Sinusoidal Cond: True
- Learned Sinusoidal Dim: 32
- Attention Dim Head: 64
- Attention Heads: 4
- Random Fourier Features: False
```

## task_007 — Super-resolution experiment downsampling upsampling coupling

Implement the image super-resolution specific components. Create downsampling operation that maps high-resolution images to low-resolution images, and upsampling operation that maps low-resolution back to high-resolution. Define the super-resolution coupling providing the low-resolution image as conditioning information appended to the model input. The fixed upsampled low-resolution condition xi=U(D(x1)) is explicitly appended to the evolving image at every model call; it is not part of the evolved ODE state. Support multiple resolution pairs. Include class label conditioning alongside the low-resolution image conditioning. This module handles the specific data preparation for the super-resolution experiment.

### task_007_check_001

For this rubric-assisted diagnostic profile, construct low-resolution training observations from the prepared 256x256 image by a genuine 64x64 crop, and from 512x512 by a 256x256 crop. Make crop placement deterministic/configured and documented (the rubric does not specify its location). This crop protocol refines the paper's unspecified D operator; resize, average pooling or crop after already resizing to the low resolution is not this diagnostic operation.

来源性质：`rubric_assisted_protocol`

**paper.md**

```text
tion 4.1. We denote by $\mathcal{D}: \mathbb{R}^{C \times W \times H} \rightarrow \mathbb{R}^{C \times W_{\text {low }} \times H_{\text {low }}}$ and $\mathcal{U}: \mathbb{R}^{C \times W_{\text {low }} \times H_{\text {low }}} \rightarrow \mathbb{R}^{C \times W \times H}$ image downsampling and upsampling operations, where $W_{\text {low }}$ and $H_{\text {low }}$ denote the width and height of a low-resolution image. To define the base density, we then set $x_{0}=\mathcal{U}\left(\mathcal{D}\left(x_{1}\right)\right)+\sigma \zeta$ with $\zeta \in \mathbb{R}^{C \times W \times H}, \zeta \sim \mathrm{~N}(0, I d)$, and $\sigma>0$. Defining $x_{0}$ in this way frames the transport problem such that each starting pixel is proximal to its intended target. Notice in particular that, with $\sigma=0$, each $x_{0}$ would correspond to a lowerdimensional sample embedded in a higher-dimensional space, and the corresponding distribution would be concentrated on a lower-dimensional manifold. Working with $\sigma>0$ alleviates the associated singularities by adding a small amount of Gaussian noise to smooth the base density so it is well-defined over the entire higher-dimensional ambient space. In addition, we give the model access to the lowresolution image at all times; this problem setting then corresponds to using $\rho\left(x_{0}, x_{1} \mid \xi\right)=\rho_{1}\left(x_{1}\right) \rho_{0}\left(x_{0} \mid x_{1}, \xi\right)$ with $\xi=\mathcal{U}\left(\mathcal{D}\left(x_{1}\right)\right)$. In the experiments, we set $\rho_{1}$ to correspond to ImageNet ( 256 or 512 ), following prior work (Saharia et al., 2022; Ho et al., 2022a).
```

### task_007_check_002

Use nearest-neighbor interpolation to upsample those low-resolution observations back to 256x256 or 512x512 respectively, in both preparation and sampling. For this rubric-assisted profile nearest is fixed explicitly; the paper itself denotes only U. Bilinear or bicubic paths must not silently be the production default.

来源性质：`rubric_assisted_protocol`

**paper.md**

```text
tion 4.1. We denote by $\mathcal{D}: \mathbb{R}^{C \times W \times H} \rightarrow \mathbb{R}^{C \times W_{\text {low }} \times H_{\text {low }}}$ and $\mathcal{U}: \mathbb{R}^{C \times W_{\text {low }} \times H_{\text {low }}} \rightarrow \mathbb{R}^{C \times W \times H}$ image downsampling and upsampling operations, where $W_{\text {low }}$ and $H_{\text {low }}$ denote the width and height of a low-resolution image. To define the base density, we then set $x_{0}=\mathcal{U}\left(\mathcal{D}\left(x_{1}\right)\right)+\sigma \zeta$ with $\zeta \in \mathbb{R}^{C \times W \times H}, \zeta \sim \mathrm{~N}(0, I d)$, and $\sigma>0$. Defining $x_{0}$ in this way frames the transport problem such that each starting pixel is proximal to its intended target. Notice in particular that, with $\sigma=0$, each $x_{0}$ would correspond to a lowerdimensional sample embedded in a higher-dimensional space, and the corresponding distribution would be concentrated on a lower-dimensional manifold. Working with $\sigma>0$ alleviates the associated singularities by adding a small amount of Gaussian noise to smooth the base density so it is well-defined over the entire higher-dimensional ambient space. In addition, we give the model access to the lowresolution image at all times; this problem setting then corresponds to using $\rho\left(x_{0}, x_{1} \mid \xi\right)=\rho_{1}\left(x_{1}\right) \rho_{0}\left(x_{0} \mid x_{1}, \xi\right)$ with $\xi=\mathcal{U}\left(\mathcal{D}\left(x_{1}\right)\right)$. In the experiments, we set $\rho_{1}$ to correspond to ImageNet ( 256 or 512 ), following prior work (Saharia et al., 2022; Ho et al., 2022a).
```

### task_007_check_003

Construct super-resolution x0=U(D(x1))+sigma*zeta using independent standard Gaussian image noise. Use sigma=1 in the diagnostic production profile to match the rubric; retain its explicit value in configuration. The paper permits sigma>0 and does not set it to one. Do not initialize from pure noise or omit the nonzero noise.

来源性质：`rubric_assisted_protocol`

**paper.md**

```text
tion 4.1. We denote by $\mathcal{D}: \mathbb{R}^{C \times W \times H} \rightarrow \mathbb{R}^{C \times W_{\text {low }} \times H_{\text {low }}}$ and $\mathcal{U}: \mathbb{R}^{C \times W_{\text {low }} \times H_{\text {low }}} \rightarrow \mathbb{R}^{C \times W \times H}$ image downsampling and upsampling operations, where $W_{\text {low }}$ and $H_{\text {low }}$ denote the width and height of a low-resolution image. To define the base density, we then set $x_{0}=\mathcal{U}\left(\mathcal{D}\left(x_{1}\right)\right)+\sigma \zeta$ with $\zeta \in \mathbb{R}^{C \times W \times H}, \zeta \sim \mathrm{~N}(0, I d)$, and $\sigma>0$. Defining $x_{0}$ in this way frames the transport problem such that each starting pixel is proximal to its intended target. Notice in particular that, with $\sigma=0$, each $x_{0}$ would correspond to a lowerdimensional sample embedded in a higher-dimensional space, and the corresponding distribution would be concentrated on a lower-dimensional manifold. Working with $\sigma>0$ alleviates the associated singularities by adding a small amount of Gaussian noise to smooth the base density so it is well-defined over the entire higher-dimensional ambient space. In addition, we give the model access to the lowresolution image at all times; this problem setting then corresponds to using $\rho\left(x_{0}, x_{1} \mid \xi\right)=\rho_{1}\left(x_{1}\right) \rho_{0}\left(x_{0} \mid x_{1}, \xi\right)$ with $\xi=\mathcal{U}\left(\mathcal{D}\left(x_{1}\right)\right)$. In the experiments, we set $\rho_{1}$ to correspond to ImageNet ( 256 or 512 ), following prior work (Saharia et al., 2022; Ho et al., 2022a).
```

### task_007_check_004

Keep xi=U(D(x1)) as a fixed, clean upsampled low-resolution condition; append it to the evolving image I(t)/X(t) at every velocity evaluation, with class conditioning and the diagnostic constant class channel. Evolve only the image channels. The paper explicitly specifies clean xi; do not silently replace it by the noisy base or leak the unknown high-resolution target into sampling. The rubric's noisy-appending wording conflicts with this paper specification.

来源性质：`source_conflict_resolution`

**paper.md**

```text
tion 4.1. We denote by $\mathcal{D}: \mathbb{R}^{C \times W \times H} \rightarrow \mathbb{R}^{C \times W_{\text {low }} \times H_{\text {low }}}$ and $\mathcal{U}: \mathbb{R}^{C \times W_{\text {low }} \times H_{\text {low }}} \rightarrow \mathbb{R}^{C \times W \times H}$ image downsampling and upsampling operations, where $W_{\text {low }}$ and $H_{\text {low }}$ denote the width and height of a low-resolution image. To define the base density, we then set $x_{0}=\mathcal{U}\left(\mathcal{D}\left(x_{1}\right)\right)+\sigma \zeta$ with $\zeta \in \mathbb{R}^{C \times W \times H}, \zeta \sim \mathrm{~N}(0, I d)$, and $\sigma>0$. Defining $x_{0}$ in this way frames the transport problem such that each starting pixel is proximal to its intended target. Notice in particular that, with $\sigma=0$, each $x_{0}$ would correspond to a lowerdimensional sample embedded in a higher-dimensional space, and the corresponding distribution would be concentrated on a lower-dimensional manifold. Working with $\sigma>0$ alleviates the associated singularities by adding a small amount of Gaussian noise to smooth the base density so it is well-defined over the entire higher-dimensional ambient space. In addition, we give the model access to the lowresolution image at all times; this problem setting then corresponds to using $\rho\left(x_{0}, x_{1} \mid \xi\right)=\rho_{1}\left(x_{1}\right) \rho_{0}\left(x_{0} \mid x_{1}, \xi\right)$ with $\xi=\mathcal{U}\left(\mathcal{D}\left(x_{1}\right)\right)$. In the experiments, we set $\rho_{1}$ to correspond to ImageNet ( 256 or 512 ), following prior work (Saharia et al., 2022; Ho et al., 2022a).
```

**paper.md**

```text
Results. Similarly to the previous experiment, we append the upsampled low-resolution images $\xi$ to the channel dimension of the input $x$ of the velocity model, and likewise include the ImageNet class labels.
```

**paper.md**

```text
Image-shaped conditioning in the Unet. For image-shaped conditioning, we follow (Ho et al., 2022a) and append upsampled low-resolution images to the input $x_{t}$ at each time step to the velocity model. We also condition on the missingness masks for in-painting by appending them to $x_{t}$.
```

### task_007_check_005

Use identical degradation/upsampling, class encoding, image normalization and condition layout at training and inference for each resolution pair. Verify that the SR training call receives the constant class channel; implementing it only in a generic model adapter is insufficient.

来源性质：`rubric_assisted_protocol`

**paper.md**

```text
Results. Similarly to the previous experiment, we append the upsampled low-resolution images $\xi$ to the channel dimension of the input $x$ of the velocity model, and likewise include the ImageNet class labels.
```

**paper.md**

```text
Architecture For the velocity model we use the U-net from (Ho et al., 2020b) as implemented in lucidrain's denoising-diffusion-pytorch repository; this variant of the architecture includes embeddings to condition on class labels. We use the following hyperparameters:
- Dim Mults: (1,1,2,3,4)
- Dim (channels): 256
- Resnet block groups: 8
- Leanred Sinusoidal Cond: True
- Learned Sinusoidal Dim: 32
- Attention Dim Head: 64
- Attention Heads: 4
- Random Fourier Features: False
```

## task_008 — Sampling and inference engine for probability flow ODE

Implement the sampling procedure as specified in Algorithm 2 and Section 4. Create a sampler that generates samples from the trained velocity model via forward integration of the probability flow ODE. Use the Dopri solver from torchdiffeq as specified in Appendix B. For inpainting: initialize X0 with known pixels from the observed image and masked pixels initialized with Gaussian noise. For super-resolution: initialize X0 with upsampled low-resolution plus noise. Support both Euler method stepping as in Algorithm 2 and the Dopri adaptive solver. Generate samples conditioned on the provided corrupted observation and any conditioning variables. Output generated samples at t=1.

### task_008_check_001

Implement Algorithm 2 Euler updates X[n+1]=X[n]+b(n/N,X[n])/N for exactly N steps, n=0,...,N-1, returning X[N]. The state starts at the selected base and reaches t=1 under the forward convention; time conditioning must follow n/N. Provide working dependent-inpainting, uncoupled and SR dispatches, not just a generic unused loop.

来源性质：`paper_or_addendum`

**paper.md**

```text
Algorithm 2 Sampling (via forward Euler method)
    Input: model $\hat{b}$, corrupted sample $m\left(x_{1}\right), N \in \mathbb{N}$.
    Draw noise $\zeta \sim \mathcal{N}(0, I d)$
    Initialize $\hat{X}_{0}=m\left(x_{1}\right)+\sigma \zeta$
    for $n=0, \ldots, N-1$ do
        $\hat{X}_{i+1}=\hat{X}_{i}+N^{-1} \hat{b}_{i / N}\left(\hat{X}_{i}\right)$
    end for
    Return: clean sample $\hat{X}_{N}$.
```

**paper.md**

```text
Definition 3.1 (Stochastic interpolant with coupling). The stochastic interpolant $I_{t}$ is the process defined as ${ }^{1}$
\[
I_{t}=\alpha_{t} x_{0}+\beta_{t} x_{1}+\gamma_{t} z \quad t \in[0,1]
\]
where
- $\alpha_{t}, \beta_{t}$, and $\gamma_{t}^{2}$ are differentiable functions of time such that $\alpha_{0}=\beta_{1}=1, \alpha_{1}=\beta_{0}=\gamma_{0}=\gamma_{1}=0$, and $\alpha_{t}^{2}+\beta_{t}^{2}+\gamma_{t}^{2}>0$ for all $t \in[0,1]$.
- The pair $\left(x_{0}, x_{1}\right)$ is jointly drawn from a probability density $\rho\left(x_{0}, x_{1}\right)$ with finite second moments and such that
\[
\begin{aligned}
& \int_{\mathbb{R}^{d}} \rho\left(x_{0}, x_{1}\right) d x_{1}=\rho_{0}\left(x_{0}\right) \\
& \int_{\mathbb{R}^{d}} \rho\left(x_{0}, x_{1}\right) d x_{0}=\rho_{1}\left(x_{1}\right)
\end{aligned}
\]
- $z \sim \mathrm{~N}(0$, Id $)$, independent of $\left(x_{0}, x_{1}\right)$.
```

### task_008_check_002

Use the torchdiffeq Dopri solver (dopri5) for the Appendix B production sampling path, with the trained velocity RHS and the same state/condition semantics as Euler. Do not label Euler as Dopri, silently fall back on import failure, or introduce an unrequested stochastic/SDE sampler.

来源性质：`paper_or_addendum`

**paper.md**

```text
Integration for sampling We use the Dopri solver from the torchdiffeq library (Chen, 2018).
```

**paper.md**

```text
Algorithm 2 Sampling (via forward Euler method)
    Input: model $\hat{b}$, corrupted sample $m\left(x_{1}\right), N \in \mathbb{N}$.
    Draw noise $\zeta \sim \mathcal{N}(0, I d)$
    Initialize $\hat{X}_{0}=m\left(x_{1}\right)+\sigma \zeta$
    for $n=0, \ldots, N-1$ do
        $\hat{X}_{i+1}=\hat{X}_{i}+N^{-1} \hat{b}_{i / N}\left(\hat{X}_{i}\right)$
    end for
    Return: clean sample $\hat{X}_{N}$.
```

### task_008_check_003

Dependent inpainting inference must load actual held-out ImageNet validation/test observations, create the specified 64-tile shared-channel mask, and initialize X0=xi*x_observed+(1-xi)*zeta. Trace the CLI through loader, constructor and sampler: a correct helper that is not called does not satisfy this item. Never use the binary mask itself as the image or substitute training images.

来源性质：`rubric_assisted_protocol`

**paper.md**

```text
For simplicity, the mask takes the same value for all channels in a given spatial location in the image. We define the base density by the relation $x_{0}=\xi \circ x_{1}+(1-\xi) \circ \zeta$, where $\circ$ denotes the Hadamard (elementwise) product and $\zeta \in \mathbb{R}^{C \times W \times H}, \zeta \sim \mathrm{~N}(0, I d)$ denotes random noise used to initialize the pixels within the masked region (separate noise for each channel). During
training, the mask is drawn randomly by tiling the image into 64 tiles; each tile is selected to enter the mask with probability $p=0.3$.
```

**paper.md**

```text
Algorithm 2 Sampling (via forward Euler method)
    Input: model $\hat{b}$, corrupted sample $m\left(x_{1}\right), N \in \mathbb{N}$.
    Draw noise $\zeta \sim \mathcal{N}(0, I d)$
    Initialize $\hat{X}_{0}=m\left(x_{1}\right)+\sigma \zeta$
    for $n=0, \ldots, N-1$ do
        $\hat{X}_{i+1}=\hat{X}_{i}+N^{-1} \hat{b}_{i / N}\left(\hat{X}_{i}\right)$
    end for
    Return: clean sample $\hat{X}_{N}$.
```

### task_008_check_004

Inpainting sampling must pass the corresponding real class labels, diagnostic uniform class-value channel and mask at every velocity evaluation. Multiply image velocity by the missing-region mask so observed pixels remain fixed throughout integration, not merely pasted back after generation.

来源性质：`rubric_assisted_protocol`

**paper.md**

```text
The model sees the mask; we note that we do not need to additionally input the partial image as extra conditioning because it is present, uncorrupted, in $x_{t}$ for each $t$ because the values are present in $x_{0}$ and $x_{1}$. In the interpolant (20), we set $\alpha_{t}=t$ and $\beta_{t}=1-t$. In this setup, the velocity field $b_{t}(x, \xi)$ is such that $b_{t}(x, \xi)=0$ except in the masked regions. This follows because $\xi \circ I_{t}=\xi \circ x_{1}$ for every $t$, i.e., the unmasked pixels in $I_{t}$ are always those of $x_{1}$ for which $\dot{I}_{t}=0$. To take this structural information into account, we can build this property into our neural network model, and mask the output of the approximate velocity field to enforce that the unmasked pixels remain fixed. We note that this method does not necessitate any inference time corrections, such as the replacement method or MCMC.
```

**paper.md**

```text
Results. For implementation, we parameterize $b_{t}(x, \xi)$ using the basic U-Net architecture from (Ho et al., 2020b), where $\xi$ is given to the model as appended channels of the image $x$.
```

**paper.md**

```text
Architecture For the velocity model we use the U-net from (Ho et al., 2020b) as implemented in lucidrain's denoising-diffusion-pytorch repository; this variant of the architecture includes embeddings to condition on class labels. We use the following hyperparameters:
- Dim Mults: (1,1,2,3,4)
- Dim (channels): 256
- Resnet block groups: 8
- Leanred Sinusoidal Cond: True
- Learned Sinusoidal Dim: 32
- Attention Dim Head: 64
- Attention Heads: 4
- Random Fourier Features: False
```

### task_008_check_005

The held-out SR sampling entry point must obtain real validation/test images and derive the diagnostic 256-to-64 or 512-to-256 crop, nearest upsampling, noisy base and clean fixed condition consistently with training. The separate training-reference FID mode is explicitly named and must not replace this held-out path.

来源性质：`rubric_assisted_protocol`

**paper.md**

```text
tion 4.1. We denote by $\mathcal{D}: \mathbb{R}^{C \times W \times H} \rightarrow \mathbb{R}^{C \times W_{\text {low }} \times H_{\text {low }}}$ and $\mathcal{U}: \mathbb{R}^{C \times W_{\text {low }} \times H_{\text {low }}} \rightarrow \mathbb{R}^{C \times W \times H}$ image downsampling and upsampling operations, where $W_{\text {low }}$ and $H_{\text {low }}$ denote the width and height of a low-resolution image. To define the base density, we then set $x_{0}=\mathcal{U}\left(\mathcal{D}\left(x_{1}\right)\right)+\sigma \zeta$ with $\zeta \in \mathbb{R}^{C \times W \times H}, \zeta \sim \mathrm{~N}(0, I d)$, and $\sigma>0$. Defining $x_{0}$ in this way frames the transport problem such that each starting pixel is proximal to its intended target. Notice in particular that, with $\sigma=0$, each $x_{0}$ would correspond to a lowerdimensional sample embedded in a higher-dimensional space, and the corresponding distribution would be concentrated on a lower-dimensional manifold. Working with $\sigma>0$ alleviates the associated singularities by adding a small amount of Gaussian noise to smooth the base density so it is well-defined over the entire higher-dimensional ambient space. In addition, we give the model access to the lowresolution image at all times; this problem setting then corresponds to using $\rho\left(x_{0}, x_{1} \mid \xi\right)=\rho_{1}\left(x_{1}\right) \rho_{0}\left(x_{0} \mid x_{1}, \xi\right)$ with $\xi=\mathcal{U}\left(\mathcal{D}\left(x_{1}\right)\right)$. In the experiments, we set $\rho_{1}$ to correspond to ImageNet ( 256 or 512 ), following prior work (Saharia et al., 2022; Ho et al., 2022a).
```

**paper.md**

```text
Figure 4: Super-resolution: Top four rows: Super-resolved images from resolution $64 \times 64 \mapsto 256 \times 256$, where the left-most image is the lower resolution version, the middle is the model output, and the right is the ground truth. Examples for $256 \times 256 \mapsto 512 \times 512$ are given in Fig. 6 .
```

**paper.md**

```text
Figure 6: Super-resolution: Top four rows: Super-resolved images from resolution $256 \times 256 \mapsto 512 \times 512$, where the left-most image is the lower resolution version, the middle is the model output, and the right is the ground truth.
```

### task_008_check_006

SR sampling receives the corresponding class labels and diagnostic constant class-value channel; the model sees evolving RGB state plus fixed low-resolution/class conditions, and outputs only RGB velocity. Unknown high-resolution pixels are not an extra inference input. Preserve conditions unchanged throughout all solver calls.

来源性质：`rubric_assisted_protocol`

**paper.md**

```text
Results. Similarly to the previous experiment, we append the upsampled low-resolution images $\xi$ to the channel dimension of the input $x$ of the velocity model, and likewise include the ImageNet class labels.
```

**paper.md**

```text
Image-shaped conditioning in the Unet. For image-shaped conditioning, we follow (Ho et al., 2022a) and append upsampled low-resolution images to the input $x_{t}$ at each time step to the velocity model. We also condition on the missingness masks for in-painting by appending them to $x_{t}$.
```

### task_008_check_007

Load the selected trained checkpoint/config in evaluation mode and disable gradients for inference without detaching training-time predictions. Sampling outputs must be the solver endpoint, not the reference image, a nearest-upsample-only proxy or an unrelated cached tensor.

来源性质：`paper_or_addendum`

**paper.md**

```text
Algorithm 2 Sampling (via forward Euler method)
    Input: model $\hat{b}$, corrupted sample $m\left(x_{1}\right), N \in \mathbb{N}$.
    Draw noise $\zeta \sim \mathcal{N}(0, I d)$
    Initialize $\hat{X}_{0}=m\left(x_{1}\right)+\sigma \zeta$
    for $n=0, \ldots, N-1$ do
        $\hat{X}_{i+1}=\hat{X}_{i}+N^{-1} \hat{b}_{i / N}\left(\hat{X}_{i}\right)$
    end for
    Return: clean sample $\hat{X}_{N}$.
```

**paper.md**

```text
Figure 3: Image inpainting: ImageNet- $256 \times 256$ and ImageNet- $512 \times 512$. Top panels: Six examples of image in-filling at resolution $256 \times 256$, where the left columns display masked images, the center corresponds to in-filled model samples, and the right shows full reference images. The aims are not to recover the precise content of the reference image, but instead, to provide a conditionally valid in-filling. Bottom panels: Four examples at resolution $512 \times 512$.
![](assets/asset_4.jpg)
```

**paper.md**

```text
Figure 4: Super-resolution: Top four rows: Super-resolved images from resolution $64 \times 64 \mapsto 256 \times 256$, where the left-most image is the lower resolution version, the middle is the model output, and the right is the ground truth. Examples for $256 \times 256 \mapsto 512 \times 512$ are given in Fig. 6 .
```

## task_009 — FID evaluation and metric computation

Implement FID Fréchet Inception Distance score computation for evaluating generated samples. Create an evaluation pipeline that computes FID between generated samples and real ImageNet samples. Support generating fifty thousand samples for evaluation as referenced in Tables 2 and 3. Implement the evaluation protocol used in the paper for both inpainting and super-resolution tasks. Include utilities for comparing dependent coupling results against the independent coupling baseline. Report FID scores formatted consistently with the paper tables.

### task_009_check_001

Implement genuine Fréchet Inception Distance using a declared standard pretrained Inception feature extractor, empirical means/covariances and a numerically sound covariance square-root term. Correctly convert model image ranges to the extractor's expected preprocessing. A random/untrained feature network, pixel MSE or hard-coded scalar is not FID.

来源性质：`paper_or_addendum`

**paper.md**

```text
Table 2: FID for Inpainting Task. FID comparison between under two paradigms: a baseline, where $\rho_{0}$ is a Gaussian with independent coupling to $\rho_{1}$, and our datadependent coupling detailed in Section 4.1.
\begin{tabular}{ll}
\hline Model & FID-50k \\
\hline Uncoupled Interpolant (Baseline) & 1.35 \\
Dependent Coupling (Ours) & $\mathbf{1 . 1 3}$ \\
\hline
\end{tabular}
```

**paper.md**

```text
Table 3: FID-50k for Super-resolution, 64x64 to 256x256. FIDs for baselines taken from (Saharia et al., 2022; Ho et al., 2022a; Liu et al., 2023a).
\begin{tabular}{lll}
\hline Model & Train & Valid \\
\hline Improved DDPM (Nichol \& Dhariwal, 2021) & 12.26 & - \\
SR3 (Saharia et al., 2022) & 11.30 & 5.20 \\
ADM (Dhariwal \& Nichol, 2021) & 7.49 & 3.10 \\
Cascaded Diffusion (Ho et al., 2022a) & 4.88 & 4.63 \\
I $^{2}$ SB (Liu et al., 2023a) & - & 2.70 \\
Dependent Coupling (Ours) & $\mathbf{2 . 1 3}$ & $\mathbf{2 . 0 5}$ \\
\hline
\end{tabular}
```

### task_009_check_002

Provide FID-50k evaluation paths for both independent and dependent inpainting against the held-out ImageNet validation reference. Support exactly 50000 generated/reference samples with streaming/batched features, including partial final batches, and distinguish model outputs from source/reference images.

来源性质：`rubric_assisted_protocol`

**paper.md**

```text
Table 2: FID for Inpainting Task. FID comparison between under two paradigms: a baseline, where $\rho_{0}$ is a Gaussian with independent coupling to $\rho_{1}$, and our datadependent coupling detailed in Section 4.1.
\begin{tabular}{ll}
\hline Model & FID-50k \\
\hline Uncoupled Interpolant (Baseline) & 1.35 \\
Dependent Coupling (Ours) & $\mathbf{1 . 1 3}$ \\
\hline
\end{tabular}
```

### task_009_check_003

For 64-to-256 SR, implement separately labeled Train and Valid FID-50k reports; the diagnostic train reference uses 50000 selected training samples and validation uses its own reference statistics. Cache keys must distinguish split, resolution and feature-extractor settings. Do not reuse train statistics as validation or compare references with themselves.

来源性质：`rubric_assisted_protocol`

**paper.md**

```text
Table 3: FID-50k for Super-resolution, 64x64 to 256x256. FIDs for baselines taken from (Saharia et al., 2022; Ho et al., 2022a; Liu et al., 2023a).
\begin{tabular}{lll}
\hline Model & Train & Valid \\
\hline Improved DDPM (Nichol \& Dhariwal, 2021) & 12.26 & - \\
SR3 (Saharia et al., 2022) & 11.30 & 5.20 \\
ADM (Dhariwal \& Nichol, 2021) & 7.49 & 3.10 \\
Cascaded Diffusion (Ho et al., 2022a) & 4.88 & 4.63 \\
I $^{2}$ SB (Liu et al., 2023a) & - & 2.70 \\
Dependent Coupling (Ours) & $\mathbf{2 . 1 3}$ & $\mathbf{2 . 0 5}$ \\
\hline
\end{tabular}
```

### task_009_check_004

Evaluation must call the selected experiment's actual checkpoint/sampler with the matching coupling and resolution. Report measured values only when computed; paper values are labeled reference values. Achieving published FIDs or running expensive evaluation is not a CodeDev completion requirement.

来源性质：`paper_or_addendum`

**instructions.txt**

```text
Your submission will be graded based on whether the code necessary for reproducing the paper has been correctly implemented. The code will not be executed during grading.
```

**paper.md**

```text
Table 2: FID for Inpainting Task. FID comparison between under two paradigms: a baseline, where $\rho_{0}$ is a Gaussian with independent coupling to $\rho_{1}$, and our datadependent coupling detailed in Section 4.1.
\begin{tabular}{ll}
\hline Model & FID-50k \\
\hline Uncoupled Interpolant (Baseline) & 1.35 \\
Dependent Coupling (Ours) & $\mathbf{1 . 1 3}$ \\
\hline
\end{tabular}
```

**paper.md**

```text
Table 3: FID-50k for Super-resolution, 64x64 to 256x256. FIDs for baselines taken from (Saharia et al., 2022; Ho et al., 2022a; Liu et al., 2023a).
\begin{tabular}{lll}
\hline Model & Train & Valid \\
\hline Improved DDPM (Nichol \& Dhariwal, 2021) & 12.26 & - \\
SR3 (Saharia et al., 2022) & 11.30 & 5.20 \\
ADM (Dhariwal \& Nichol, 2021) & 7.49 & 3.10 \\
Cascaded Diffusion (Ho et al., 2022a) & 4.88 & 4.63 \\
I $^{2}$ SB (Liu et al., 2023a) & - & 2.70 \\
Dependent Coupling (Ours) & $\mathbf{2 . 1 3}$ & $\mathbf{2 . 0 5}$ \\
\hline
\end{tabular}
```

## task_010 — Independent coupling baseline experiment

Implement the independent coupling baseline experiment as described in Table 2. This runs the same training and sampling pipeline but with x0 sampled independently from the base distribution standard Gaussian rather than from the data-dependent coupling. For inpainting: compare uncoupled interpolant baseline versus dependent coupling. For super-resolution: the dependent coupling approach is compared against prior work baselines Improved DDPM SR3 ADM Cascaded Diffusion I2SB which are out of scope per the addendum; implement only the dependent coupling version. Use identical architecture, hyperparameters, and training protocol as the main experiments to ensure fair comparison.

### task_010_check_001

In independent-baseline training, draw x0~N(0,Id) independently of x1 over the entire RGB image. Do not reuse the dependent observed-pixel/noise constructor, initialize from U(D(x1)), or change only an experiment label while retaining dependent coupling.

来源性质：`paper_or_addendum`

**paper.md**

```text
Table 2: FID for Inpainting Task. FID comparison between under two paradigms: a baseline, where $\rho_{0}$ is a Gaussian with independent coupling to $\rho_{1}$, and our datadependent coupling detailed in Section 4.1.
\begin{tabular}{ll}
\hline Model & FID-50k \\
\hline Uncoupled Interpolant (Baseline) & 1.35 \\
Dependent Coupling (Ours) & $\mathbf{1 . 1 3}$ \\
\hline
\end{tabular}
```

**paper.md**

```text
Definition 3.1 (Stochastic interpolant with coupling). The stochastic interpolant $I_{t}$ is the process defined as ${ }^{1}$
\[
I_{t}=\alpha_{t} x_{0}+\beta_{t} x_{1}+\gamma_{t} z \quad t \in[0,1]
\]
where
- $\alpha_{t}, \beta_{t}$, and $\gamma_{t}^{2}$ are differentiable functions of time such that $\alpha_{0}=\beta_{1}=1, \alpha_{1}=\beta_{0}=\gamma_{0}=\gamma_{1}=0$, and $\alpha_{t}^{2}+\beta_{t}^{2}+\gamma_{t}^{2}>0$ for all $t \in[0,1]$.
- The pair $\left(x_{0}, x_{1}\right)$ is jointly drawn from a probability density $\rho\left(x_{0}, x_{1}\right)$ with finite second moments and such that
\[
\begin{aligned}
& \int_{\mathbb{R}^{d}} \rho\left(x_{0}, x_{1}\right) d x_{1}=\rho_{0}\left(x_{0}\right) \\
& \int_{\mathbb{R}^{d}} \rho\left(x_{0}, x_{1}\right) d x_{0}=\rho_{1}\left(x_{1}\right)
\end{aligned}
\]
- $z \sim \mathrm{~N}(0$, Id $)$, independent of $\left(x_{0}, x_{1}\right)$.
```

### task_010_check_002

The independent baseline uses per-sample uniform times, the same coherent image bridge/analytic derivative and quadratic image-velocity loss as task_003/task_005, optimized with Adam. The class label must reach both the class conditioning and the diagnostic uniform class channel, not remain a disconnected dataloader field.

来源性质：`rubric_assisted_protocol`

**paper.md**

```text
Algorithm 1 Training
    Input: Interpolant coefficients $\alpha_{t}, \beta_{t}$; velocity model $\hat{b}$;
    batch size $n_{b}$;
    repeat
        for $i=1, \ldots, n_{\mathrm{b}}$ do
            Draw $x_{1}^{i} \sim \rho_{1}\left(x_{1}\right), \zeta_{i} \sim \mathcal{N}(0, I d), t_{i} \sim U(0,1)$.
            Compute $x_{0}^{i}=m\left(x_{1}^{i}\right)+\sigma \zeta^{i}$.
            Compute $I_{t_{i}}=\alpha_{t_{i}} x_{0}^{i}+\beta_{t_{i}} x_{1}^{i}$.
        end for
        Compute empirical loss
        $\hat{L}_{b}(\hat{b})=n_{\mathrm{b}}^{-1} \sum_{i=1}^{n_{\mathrm{b}}}\left[\left|\hat{b}_{t_{i}}\left(I_{t_{i}}\right)\right|^{2}-2 \dot{I}_{t_{i}} \cdot \hat{b}_{t_{i}}\left(I_{t_{i}}\right)\right]$.
        Take gradient step on $\hat{L}_{b}(\hat{b})$ to update $\hat{b}$.
    until converged
    Return: Velocity $\hat{b}$.
```

**paper.md**

```text
Architecture For the velocity model we use the U-net from (Ho et al., 2020b) as implemented in lucidrain's denoising-diffusion-pytorch repository; this variant of the architecture includes embeddings to condition on class labels. We use the following hyperparameters:
- Dim Mults: (1,1,2,3,4)
- Dim (channels): 256
- Resnet block groups: 8
- Leanred Sinusoidal Cond: True
- Learned Sinusoidal Dim: 32
- Attention Dim Head: 64
- Attention Heads: 4
- Random Fourier Features: False
```

**paper.md**

```text
Table 2: FID for Inpainting Task. FID comparison between under two paradigms: a baseline, where $\rho_{0}$ is a Gaussian with independent coupling to $\rho_{1}$, and our datadependent coupling detailed in Section 4.1.
\begin{tabular}{ll}
\hline Model & FID-50k \\
\hline Uncoupled Interpolant (Baseline) & 1.35 \\
Dependent Coupling (Ours) & $\mathbf{1 . 1 3}$ \\
\hline
\end{tabular}
```

### task_010_check_003

Independent sampling starts from pure full-image Gaussian noise, then integrates its own trained velocity forward using the implemented Euler/Dopri path. It must not inherit dependent initialization or a known-pixel zero-velocity constraint, which would freeze initially noisy pixels. Any additional observation-conditioning policy for this underspecified baseline must be explicit and consistent between training and sampling, not a disguised dependent base.

来源性质：`paper_or_addendum`

**paper.md**

```text
Table 2: FID for Inpainting Task. FID comparison between under two paradigms: a baseline, where $\rho_{0}$ is a Gaussian with independent coupling to $\rho_{1}$, and our datadependent coupling detailed in Section 4.1.
\begin{tabular}{ll}
\hline Model & FID-50k \\
\hline Uncoupled Interpolant (Baseline) & 1.35 \\
Dependent Coupling (Ours) & $\mathbf{1 . 1 3}$ \\
\hline
\end{tabular}
```

**paper.md**

```text
Algorithm 2 Sampling (via forward Euler method)
    Input: model $\hat{b}$, corrupted sample $m\left(x_{1}\right), N \in \mathbb{N}$.
    Draw noise $\zeta \sim \mathcal{N}(0, I d)$
    Initialize $\hat{X}_{0}=m\left(x_{1}\right)+\sigma \zeta$
    for $n=0, \ldots, N-1$ do
        $\hat{X}_{i+1}=\hat{X}_{i}+N^{-1} \hat{b}_{i / N}\left(\hat{X}_{i}\right)$
    end for
    Return: clean sample $\hat{X}_{N}$.
```

### task_010_check_004

Maintain separately identified dependent and uncoupled checkpoints/results with the same stated backbone, optimizer schedule, batch size and training budget. Implement the baseline of Table 2, but do not require retraining Improved DDPM, SR3, ADM, Cascaded Diffusion or I2SB, which the addendum excludes.

来源性质：`paper_or_addendum`

**paper.md**

```text
Table 2: FID for Inpainting Task. FID comparison between under two paradigms: a baseline, where $\rho_{0}$ is a Gaussian with independent coupling to $\rho_{1}$, and our datadependent coupling detailed in Section 4.1.
\begin{tabular}{ll}
\hline Model & FID-50k \\
\hline Uncoupled Interpolant (Baseline) & 1.35 \\
Dependent Coupling (Ours) & $\mathbf{1 . 1 3}$ \\
\hline
\end{tabular}
```

**addendum.md**

```text
You do not have to replicate the baseline results for Improved DDPM, SR3, ADM, Cascaded Diffusion, or I^2SB.
```

**paper.md**

```text
Architecture For the velocity model we use the U-net from (Ho et al., 2020b) as implemented in lucidrain's denoising-diffusion-pytorch repository; this variant of the architecture includes embeddings to condition on class labels. We use the following hyperparameters:
- Dim Mults: (1,1,2,3,4)
- Dim (channels): 256
- Resnet block groups: 8
- Leanred Sinusoidal Cond: True
- Learned Sinusoidal Dim: 32
- Attention Dim Head: 64
- Attention Heads: 4
- Random Fourier Features: False
```

**paper.md**

```text
Optimization. We use Adam optimizer (Kingma \& Ba, 2014), starting at learning rate 2e-4 with the StepLR scheduler which scales the learning rate by $\gamma=.99$ every $N=1000$ steps. We use no weight decay. We clip gradient norms at 10,000 (this is the norm of the entire set of parameters taken as a vector, the default type of norm clipping in PyTorch library).
```

**addendum.md**

```text
All experiments should use a batch size of 32.
```

**addendum.md**

```text
All experiments should run for 200,000 gradient steps
```

## task_011 — Experiment orchestration and reproducibility scripts

Create orchestration scripts that configure and launch each experiment end-to-end. Implement separate entry points for inpainting experiment with dependent coupling, inpainting experiment with independent coupling baseline, and super-resolution experiment at each resolution pair. Each script should load the appropriate configuration, set up data loaders, instantiate the model, run training, perform sampling, and compute evaluation metrics. Support command-line arguments for selecting experiment type, resolution, device, and checkpoint path. Ensure all experiments use the prescribed batch size and total gradient steps. Include checkpoint saving and loading to enable resuming interrupted training.

### task_011_check_001

Wire actual runnable experiment entry points for dependent inpainting at 256 and 512, uncoupled inpainting baseline, and dependent SR at 64-to-256 and 256-to-512. The main configurations must select the correct coupling, loader, model channels, trainer, sampler and evaluator; detached demonstration helpers do not constitute an end-to-end path.

来源性质：`paper_or_addendum`

**paper.md**

```text
Figure 3: Image inpainting: ImageNet- $256 \times 256$ and ImageNet- $512 \times 512$. Top panels: Six examples of image in-filling at resolution $256 \times 256$, where the left columns display masked images, the center corresponds to in-filled model samples, and the right shows full reference images. The aims are not to recover the precise content of the reference image, but instead, to provide a conditionally valid in-filling. Bottom panels: Four examples at resolution $512 \times 512$.
![](assets/asset_4.jpg)
```

**paper.md**

```text
Figure 4: Super-resolution: Top four rows: Super-resolved images from resolution $64 \times 64 \mapsto 256 \times 256$, where the left-most image is the lower resolution version, the middle is the model output, and the right is the ground truth. Examples for $256 \times 256 \mapsto 512 \times 512$ are given in Fig. 6 .
```

**paper.md**

```text
Figure 6: Super-resolution: Top four rows: Super-resolved images from resolution $256 \times 256 \mapsto 512 \times 512$, where the left-most image is the lower resolution version, the middle is the model output, and the right is the ground truth.
```

**paper.md**

```text
Table 2: FID for Inpainting Task. FID comparison between under two paradigms: a baseline, where $\rho_{0}$ is a Gaussian with independent coupling to $\rho_{1}$, and our datadependent coupling detailed in Section 4.1.
\begin{tabular}{ll}
\hline Model & FID-50k \\
\hline Uncoupled Interpolant (Baseline) & 1.35 \\
Dependent Coupling (Ours) & $\mathbf{1 . 1 3}$ \\
\hline
\end{tabular}
```

### task_011_check_002

Resolve CLI/config choices once and propagate them consistently across training, checkpoint save/load, sampling and evaluation: architecture, class encoding, resolution, mask convention, interpolation orientation and low-resolution degradation. Production must not silently replace these with hard-coded sampling defaults.

来源性质：`paper_or_addendum`

**paper.md**

```text
Algorithm 1 Training
    Input: Interpolant coefficients $\alpha_{t}, \beta_{t}$; velocity model $\hat{b}$;
    batch size $n_{b}$;
    repeat
        for $i=1, \ldots, n_{\mathrm{b}}$ do
            Draw $x_{1}^{i} \sim \rho_{1}\left(x_{1}\right), \zeta_{i} \sim \mathcal{N}(0, I d), t_{i} \sim U(0,1)$.
            Compute $x_{0}^{i}=m\left(x_{1}^{i}\right)+\sigma \zeta^{i}$.
            Compute $I_{t_{i}}=\alpha_{t_{i}} x_{0}^{i}+\beta_{t_{i}} x_{1}^{i}$.
        end for
        Compute empirical loss
        $\hat{L}_{b}(\hat{b})=n_{\mathrm{b}}^{-1} \sum_{i=1}^{n_{\mathrm{b}}}\left[\left|\hat{b}_{t_{i}}\left(I_{t_{i}}\right)\right|^{2}-2 \dot{I}_{t_{i}} \cdot \hat{b}_{t_{i}}\left(I_{t_{i}}\right)\right]$.
        Take gradient step on $\hat{L}_{b}(\hat{b})$ to update $\hat{b}$.
    until converged
    Return: Velocity $\hat{b}$.
```

**paper.md**

```text
Algorithm 2 Sampling (via forward Euler method)
    Input: model $\hat{b}$, corrupted sample $m\left(x_{1}\right), N \in \mathbb{N}$.
    Draw noise $\zeta \sim \mathcal{N}(0, I d)$
    Initialize $\hat{X}_{0}=m\left(x_{1}\right)+\sigma \zeta$
    for $n=0, \ldots, N-1$ do
        $\hat{X}_{i+1}=\hat{X}_{i}+N^{-1} \hat{b}_{i / N}\left(\hat{X}_{i}\right)$
    end for
    Return: clean sample $\hat{X}_{N}$.
```

**paper.md**

```text
Image-shaped conditioning in the Unet. For image-shaped conditioning, we follow (Ho et al., 2022a) and append upsampled low-resolution images to the input $x_{t}$ at each time step to the velocity model. We also condition on the missingness masks for in-painting by appending them to $x_{t}$.
```

### task_011_check_003

Allow separate training/sampling/evaluation commands and checkpoint resume, with required inputs and failures handled explicitly. Do not automatically run real training or large-image inference merely to satisfy CodeDev checks; keep production implementation intact without presenting synthetic smoke output as experimental evidence.

来源性质：`paper_or_addendum`

**instructions.txt**

```text
Your submission will be graded based on whether the code necessary for reproducing the paper has been correctly implemented. The code will not be executed during grading.
```

**instructions.txt**

```text
Finally, please also include a README.md file that describes what you were able to achieve in your reproduction attempt and explains how your codebase relates to various parts of the reproduction.
```

## task_012 — Visualization and result reporting utilities

Create visualization tools for displaying experimental results consistent with the paper figures. Implement utilities for creating figure layouts showing triplets of images: base sample input, model output, and ground truth for both inpainting and super-resolution experiments. For inpainting: show masked input, filled model output, and reference. For super-resolution: show low-resolution input, super-resolved output, and high-resolution ground truth. Include temporal slice visualization capability as shown in Figure 5. Generate figure panels with proper labeling consistent with the paper presentation style. Create summary tables for FID results formatted like Tables 2 and 3.

### task_012_check_001

Generate inpainting triplets from the same example: corrupted/base input, actual integrated model sample, and original reference, for 256 and 512. Missing-region output need only be conditionally plausible, not identical to the reference; do not reject the method solely for not copying the target.

来源性质：`paper_or_addendum`

**paper.md**

```text
Figure 3: Image inpainting: ImageNet- $256 \times 256$ and ImageNet- $512 \times 512$. Top panels: Six examples of image in-filling at resolution $256 \times 256$, where the left columns display masked images, the center corresponds to in-filled model samples, and the right shows full reference images. The aims are not to recover the precise content of the reference image, but instead, to provide a conditionally valid in-filling. Bottom panels: Four examples at resolution $512 \times 512$.
![](assets/asset_4.jpg)
```

### task_012_check_002

Generate SR triplets at both main-text resolution pairs with true low-resolution input, model output and high-resolution reference in the correct order; preserve their sample correspondence and show real generated outputs rather than just nearest/bicubic enlargement.

来源性质：`paper_or_addendum`

**paper.md**

```text
Figure 4: Super-resolution: Top four rows: Super-resolved images from resolution $64 \times 64 \mapsto 256 \times 256$, where the left-most image is the lower resolution version, the middle is the model output, and the right is the ground truth. Examples for $256 \times 256 \mapsto 512 \times 512$ are given in Fig. 6 .
```

**paper.md**

```text
Figure 6: Super-resolution: Top four rows: Super-resolved images from resolution $256 \times 256 \mapsto 512 \times 512$, where the left-most image is the lower resolution version, the middle is the model output, and the right is the ground truth.
```

### task_012_check_003

Provide temporal-slice plotting from states of one genuine inpainting trajectory as in Figure 5, not independent random images or a hand-drawn interpolation. This is a visualization utility, not a requirement to train a new appendix-only experiment.

来源性质：`paper_or_addendum`

**paper.md**

```text
Figure 5: Additional examples of in-filling on the $256 \times 256$ resolution images, with temporal slices of the probability flow.
![](assets/asset_6.jpg)
```

**instructions.txt**

```text
Any experiments that are only introduced in the Appendix are not considered core contributions of the paper, and so are out of scope. This is the case even if the experiment is referenced in the main text. HOWEVER, if an experiment is described in the main body of the text, but some of the details used (e.g. details about a prompt used, or a long mathematical expression) have been put in the appendix, then that experiment is considered still in scope. The reason here is that the experiment is described in the main body of the paper.
```

### task_012_check_004

Populate Tables 2/3-style reports from measured evaluation artifacts with split/coupling labels; distinguish paper reference numbers from measurements and unavailable values. The paper reports dependent inpainting 1.13 and SR train/valid 2.13/2.05; do not silently reinterpret the rubric's approximate 1.15/2.15 as new measured results.

来源性质：`source_conflict_resolution`

**paper.md**

```text
Table 2: FID for Inpainting Task. FID comparison between under two paradigms: a baseline, where $\rho_{0}$ is a Gaussian with independent coupling to $\rho_{1}$, and our datadependent coupling detailed in Section 4.1.
\begin{tabular}{ll}
\hline Model & FID-50k \\
\hline Uncoupled Interpolant (Baseline) & 1.35 \\
Dependent Coupling (Ours) & $\mathbf{1 . 1 3}$ \\
\hline
\end{tabular}
```

**paper.md**

```text
Table 3: FID-50k for Super-resolution, 64x64 to 256x256. FIDs for baselines taken from (Saharia et al., 2022; Ho et al., 2022a; Liu et al., 2023a).
\begin{tabular}{lll}
\hline Model & Train & Valid \\
\hline Improved DDPM (Nichol \& Dhariwal, 2021) & 12.26 & - \\
SR3 (Saharia et al., 2022) & 11.30 & 5.20 \\
ADM (Dhariwal \& Nichol, 2021) & 7.49 & 3.10 \\
Cascaded Diffusion (Ho et al., 2022a) & 4.88 & 4.63 \\
I $^{2}$ SB (Liu et al., 2023a) & - & 2.70 \\
Dependent Coupling (Ours) & $\mathbf{2 . 1 3}$ & $\mathbf{2 . 0 5}$ \\
\hline
\end{tabular}
```

## task_013 — README documentation and reproduce.sh submission packaging

Create README.md documenting the reproduction effort, describing what was implemented, how the codebase relates to various parts of the paper sections algorithms figures tables, and instructions for running the experiments. Include explanations of the theoretical framework implementation, the two main experiments inpainting and super-resolution, the coupling mechanisms, and the evaluation methodology. Also create reproduce.sh as the main execution script that can be invoked to run the full reproduction pipeline, including downloading ImageNet, configuring experiments, running training, sampling, and evaluation. Ensure the final submission directory contains all source code, configuration files, and these two required artifacts without exceeding the size limit.

### task_013_check_001

README must map implemented files and commands to the two image tasks, baseline, Algorithms 1/2, Appendix B architecture/optimization, and Tables 2/3. Explicitly document the forward-time inconsistency resolution and which crop/nearest/class-channel/sigma choices belong to this rubric-assisted diagnostic rather than being quoted paper facts.

来源性质：`paper_or_addendum`

**instructions.txt**

```text
Finally, please also include a README.md file that describes what you were able to achieve in your reproduction attempt and explains how your codebase relates to various parts of the reproduction.
```

**paper.md**

```text
Definition 3.1 (Stochastic interpolant with coupling). The stochastic interpolant $I_{t}$ is the process defined as ${ }^{1}$
\[
I_{t}=\alpha_{t} x_{0}+\beta_{t} x_{1}+\gamma_{t} z \quad t \in[0,1]
\]
where
- $\alpha_{t}, \beta_{t}$, and $\gamma_{t}^{2}$ are differentiable functions of time such that $\alpha_{0}=\beta_{1}=1, \alpha_{1}=\beta_{0}=\gamma_{0}=\gamma_{1}=0$, and $\alpha_{t}^{2}+\beta_{t}^{2}+\gamma_{t}^{2}>0$ for all $t \in[0,1]$.
- The pair $\left(x_{0}, x_{1}\right)$ is jointly drawn from a probability density $\rho\left(x_{0}, x_{1}\right)$ with finite second moments and such that
\[
\begin{aligned}
& \int_{\mathbb{R}^{d}} \rho\left(x_{0}, x_{1}\right) d x_{1}=\rho_{0}\left(x_{0}\right) \\
& \int_{\mathbb{R}^{d}} \rho\left(x_{0}, x_{1}\right) d x_{0}=\rho_{1}\left(x_{1}\right)
\end{aligned}
\]
- $z \sim \mathrm{~N}(0$, Id $)$, independent of $\left(x_{0}, x_{1}\right)$.
```

**paper.md**

```text
The model sees the mask; we note that we do not need to additionally input the partial image as extra conditioning because it is present, uncorrupted, in $x_{t}$ for each $t$ because the values are present in $x_{0}$ and $x_{1}$. In the interpolant (20), we set $\alpha_{t}=t$ and $\beta_{t}=1-t$. In this setup, the velocity field $b_{t}(x, \xi)$ is such that $b_{t}(x, \xi)=0$ except in the masked regions. This follows because $\xi \circ I_{t}=\xi \circ x_{1}$ for every $t$, i.e., the unmasked pixels in $I_{t}$ are always those of $x_{1}$ for which $\dot{I}_{t}=0$. To take this structural information into account, we can build this property into our neural network model, and mask the output of the approximate velocity field to enforce that the unmasked pixels remain fixed. We note that this method does not necessitate any inference time corrections, such as the replacement method or MCMC.
```

**paper.md**

```text
tion 4.1. We denote by $\mathcal{D}: \mathbb{R}^{C \times W \times H} \rightarrow \mathbb{R}^{C \times W_{\text {low }} \times H_{\text {low }}}$ and $\mathcal{U}: \mathbb{R}^{C \times W_{\text {low }} \times H_{\text {low }}} \rightarrow \mathbb{R}^{C \times W \times H}$ image downsampling and upsampling operations, where $W_{\text {low }}$ and $H_{\text {low }}$ denote the width and height of a low-resolution image. To define the base density, we then set $x_{0}=\mathcal{U}\left(\mathcal{D}\left(x_{1}\right)\right)+\sigma \zeta$ with $\zeta \in \mathbb{R}^{C \times W \times H}, \zeta \sim \mathrm{~N}(0, I d)$, and $\sigma>0$. Defining $x_{0}$ in this way frames the transport problem such that each starting pixel is proximal to its intended target. Notice in particular that, with $\sigma=0$, each $x_{0}$ would correspond to a lowerdimensional sample embedded in a higher-dimensional space, and the corresponding distribution would be concentrated on a lower-dimensional manifold. Working with $\sigma>0$ alleviates the associated singularities by adding a small amount of Gaussian noise to smooth the base density so it is well-defined over the entire higher-dimensional ambient space. In addition, we give the model access to the lowresolution image at all times; this problem setting then corresponds to using $\rho\left(x_{0}, x_{1} \mid \xi\right)=\rho_{1}\left(x_{1}\right) \rho_{0}\left(x_{0} \mid x_{1}, \xi\right)$ with $\xi=\mathcal{U}\left(\mathcal{D}\left(x_{1}\right)\right)$. In the experiments, we set $\rho_{1}$ to correspond to ImageNet ( 256 or 512 ), following prior work (Saharia et al., 2022; Ho et al., 2022a).
```

**paper.md**

```text
Architecture For the velocity model we use the U-net from (Ho et al., 2020b) as implemented in lucidrain's denoising-diffusion-pytorch repository; this variant of the architecture includes embeddings to condition on class labels. We use the following hyperparameters:
- Dim Mults: (1,1,2,3,4)
- Dim (channels): 256
- Resnet block groups: 8
- Leanred Sinusoidal Cond: True
- Learned Sinusoidal Dim: 32
- Attention Dim Head: 64
- Attention Heads: 4
- Random Fourier Features: False
```

### task_013_check_002

Provide a real reproduce.sh that connects data access, production configurations, training, checkpoint-based sampling and evaluation without embedding unrelated workspace paths. It must expose required data credentials/cache handling without containing secrets, and must not silently skip failed components or replace their results.

来源性质：`paper_or_addendum`

**instructions.txt**

```text
SUBMISSION
---
The output of your work must be a git repository containing all the necessary components to reproduce the results of the paper. Your repository should be located at `/home/submission/`, and the committed files should not exceed 1GB in size (i.e. you should only commit source code without any heavy artifacts). We will run `git clean -fd` on the submission directory before grading it, so any untracked files will be lost.
```

**instructions.txt**

```text
Finally, please also include a README.md file that describes what you were able to achieve in your reproduction attempt and explains how your codebase relates to various parts of the reproduction.
```

**addendum.md**

```text
You should download ImageNet using HuggingFace. Below is some example code of how you can do this. You should use use `trust_remote_code=True` if you want to avoid the code waiting for stdin:
```python
from datasets import load_dataset
dataset = load_dataset("imagenet-1k", trust_remote_code=True)
```
```

### task_013_check_003

Package all necessary source/configuration files and declared dependencies under the submission repository, with README and entry scripts included in the tracked source; keep committed content below 1GB and exclude downloaded ImageNet, large model artifacts and credentials. CodeDev grading does not require running the packaged full experiments.

来源性质：`paper_or_addendum`

**instructions.txt**

```text
SUBMISSION
---
The output of your work must be a git repository containing all the necessary components to reproduce the results of the paper. Your repository should be located at `/home/submission/`, and the committed files should not exceed 1GB in size (i.e. you should only commit source code without any heavy artifacts). We will run `git clean -fd` on the submission directory before grading it, so any untracked files will be lost.
```

**instructions.txt**

```text
Your submission will be graded based on whether the code necessary for reproducing the paper has been correctly implemented. The code will not be executed during grading.
```
