# All-in-one：人工编写、rubric 辅助的诊断清单

plan.json 是 agent 输入。保留普通 RA 的 17 个任务及顺序；description 的补充与来源冲突见 AUDIT.md。
目标为 Code-Dev 实现检查，不要求现在执行训练或达到论文指标。

## task_001 — Project scaffolding and configuration management

Set up the project structure under /home/submission/ with all necessary Python files, configuration management via Hydra for tracking hyperparameters, and requirements specification. Implement resource inspection utilities to verify available hardware. Create a reproduce.sh script that orchestrates all experiments. This subtask establishes the foundational infrastructure without implementing any domain-specific logic.

### task_001_check_001

Provide an installable environment with explicitly recorded versions for the chosen numerical backend, transformer implementation, sbi and scikit-learn. Imports and configuration paths must resolve from the submission, without depending on another experiment directory. Record available resources; setup must not implicitly launch training or download generated experimental results.

> **paper.md**: We used JAX (Bradbury et al., 2018) as backbone and hydra (Yadan, 2019) to track all configurations. Code to reproduce results is available at https://github.com/ mackelab/simformer. We use SBI (Tejero-Cantero et al., 2020) for reference implementations of baselines.

> **instructions.txt**: Your submission will be graded based on whether the code necessary for reproducing the paper has been correctly implemented. The code will not be executed during grading.

### task_001_check_002

Provide task-specific scientific configuration and output/checkpoint locations, with separate simulation budget, batch size, optimizer, attention variant, random seed and inference-step settings. The scientific defaults must remain full-scale; any small diagnostic preset must be explicitly separate. The initial reproduce.sh may be a scaffold, with full orchestration owned by task_017.

> **paper.md**: The employed transformer model features a token dimension of 50 and represents diffusion time through a 128 -dimensional random Gaussian Fourier embedding. It comprises 6 layers and 4 heads with an attention size of 10, and a widening factor of 3, implying that the feed-forward block expands to a hidden dimension of 150. For the Lotka-Volterra, SIR, and Hodgkin-Huxley tasks, we increased the number of layers to 8. Similar to the above, we used a training batch size of 1000 and an Adam optimizer.

> **paper.md**: We set $\sigma_{\max }=15, \sigma_{\min }=0.0001, \beta_{\min }=0.01$, and $\beta_{\max }=10$ for all experiments. Both for the time interval $[1 e-5,1$.$] .$
> For implementing Neural Posterior Estimation (NPE), Neural Ratio Estimation (NRE), and Neural Likelihood Estimation (NLE), we utilize the sbi library (Tejero-Cantero et al., 2020), adopting default parameters but opting for a more expressive neural spline flow for NPE and NLE. Each method was trained using the provided training loop with a batch size of 1000 and an Adam optimizer. Training ceased upon convergence, as indicated by early stopping based on validation loss.

> **paper.md**: For inference, we solved the reverse SDE using an Euler-Maruyama discretization. We use 500 steps by default; accuracy for different budgets is shown in Fig. A7.

> **instructions.txt**: Your submission will be graded based on whether the code necessary for reproducing the paper has been correctly implemented. The code will not be executed during grading.

## task_002 — Diffusion model core VESDE and VPSDE SDEs with score matching

Implement the core diffusion model machinery including Variance Exploding SDE and Variance Preserving SDE formulations. Implement the forward diffusion process that transforms data into noise according to the specified SDE drift and diffusion coefficients. Implement the reverse SDE solver using Euler-Maruyama discretization for sampling. Implement denoising score matching loss computation, including the time-dependent weighting function. Include Gaussian Fourier embedding for diffusion time encoding and linear projection added to each feed-forward block output as specified in the addendum. Diagnostic clarification: use the SDE conventions and consistent perturbation/score units specified in the checklist; time embedding is implemented by task_003. No training or full sampling run is required at this stage.

### task_002_check_001

Implement VE drift f(x,t)=0 and diffusion g(t)=sigma_min*(sigma_max/sigma_min)**t*sqrt(2*log(sigma_max/sigma_min)), with sigma_min=0.0001 and sigma_max=15. g is a diffusion coefficient, not the marginal variance or standard deviation; the actual sampling/training configuration must use these constants.

> **paper.md**: For VESDE:
> \[
> f_{\mathrm{VESDE}}(x, t)=0, \quad g_{\mathrm{VESDE}}(t)=\sigma_{\min } \cdot\left(\frac{\sigma_{\max }}{\sigma_{\min }}\right)^{t} \cdot \sqrt{2 \log \frac{\sigma_{\max }}{\sigma_{\min }}}
> \]

> **paper.md**: We set $\sigma_{\max }=15, \sigma_{\min }=0.0001, \beta_{\min }=0.01$, and $\beta_{\max }=10$ for all experiments. Both for the time interval $[1 e-5,1$.$] .$
> For implementing Neural Posterior Estimation (NPE), Neural Ratio Estimation (NRE), and Neural Likelihood Estimation (NLE), we utilize the sbi library (Tejero-Cantero et al., 2020), adopting default parameters but opting for a more expressive neural spline flow for NPE and NLE. Each method was trained using the provided training loop with a batch size of 1000 and an Adam optimizer. Training ceased upon convergence, as indicated by early stopping based on validation loss.

### task_002_check_002

For VE use mean x0 and marginal standard deviation sigma(t)=sigma_min*(sigma_max/sigma_min)**t; sample xt=x0+sigma(t)*epsilon and target score=-(xt-x0)/sigma(t)**2=-epsilon/sigma(t). Distinguish variance sigma(t)**2 from standard deviation at library calls. Initialize terminal unobserved variables with scale sigma(1)=15. Do not copy the malformed rubric kernel mean xt as a data-dependent forward mean.

> **paper.md**: For VESDE:
> \[
> f_{\mathrm{VESDE}}(x, t)=0, \quad g_{\mathrm{VESDE}}(t)=\sigma_{\min } \cdot\left(\frac{\sigma_{\max }}{\sigma_{\min }}\right)^{t} \cdot \sqrt{2 \log \frac{\sigma_{\max }}{\sigma_{\min }}}
> \]

> **paper.md**: Samples from the generative model are then generated by simulating the reverse diffusion process (Anderson, 1982)
> \[
> d \hat{\boldsymbol{x}}_{t}=\left[f\left(\hat{\boldsymbol{x}}_{t}, t\right)-g(t)^{2} s\left(\hat{\boldsymbol{x}}_{t}, t\right)\right] d t+g(t) d \tilde{\boldsymbol{w}}
> \]

> **paper.md**: We set $\sigma_{\max }=15, \sigma_{\min }=0.0001, \beta_{\min }=0.01$, and $\beta_{\max }=10$ for all experiments. Both for the time interval $[1 e-5,1$.$] .$
> For implementing Neural Posterior Estimation (NPE), Neural Ratio Estimation (NRE), and Neural Likelihood Estimation (NLE), we utilize the sbi library (Tejero-Cantero et al., 2020), adopting default parameters but opting for a more expressive neural spline flow for NPE and NLE. Each method was trained using the provided training loop with a batch size of 1000 and an Adam optimizer. Training ceased upon convergence, as indicated by early stopping based on validation loss.

### task_002_check_003

Sample a diffusion time uniformly in [1e-5,1] for each training sample and use the same bounds for inference; keep this time distinct from physical simulator observation times. Handle scalar and batched time broadcasting across variable axes explicitly.

> **paper.md**: We set $\sigma_{\max }=15, \sigma_{\min }=0.0001, \beta_{\min }=0.01$, and $\beta_{\max }=10$ for all experiments. Both for the time interval $[1 e-5,1$.$] .$
> For implementing Neural Posterior Estimation (NPE), Neural Ratio Estimation (NRE), and Neural Likelihood Estimation (NLE), we utilize the sbi library (Tejero-Cantero et al., 2020), adopting default parameters but opting for a more expressive neural spline flow for NPE and NLE. Each method was trained using the provided training loop with a batch size of 1000 and an Adam optimizer. Training ceased upon convergence, as indicated by early stopping based on validation loss.

> **paper.md**: \subsection*{3.3. Simformer training and sampling}
> 
> Having defined the tokenizer which processes every $(\boldsymbol{\theta}, \boldsymbol{x})$ pair and the attention mask to specify dependencies within the simulator, the Simformer can be trained using denoising score-matching (Hyvärinen \& Dayan, 2005; Song et al., 2021b): We sample the noise level $t$ for the diffusion model uniformly at random and generate a (partially) noisy sample $\hat{\mathbf{x}}_{t}^{M_{C}}=\left(1-M_{C}\right) \cdot \hat{\mathbf{x}}_{t}+M_{C} \cdot \hat{\mathbf{x}}_{0}$ i.e. variables that we want to condition on remain clean. The loss can then be defined as
> \[
> \begin{aligned}
> & \ell\left(\phi, M_{C}, t, \hat{\mathbf{x}}_{0}, \hat{\mathbf{x}}_{t}\right)= \\
> & \quad\left(1-M_{C}\right) \cdot\left(s_{\phi}^{M_{E}}\left(\hat{\mathbf{x}}_{t}^{M_{C}}, t\right)-\nabla_{\hat{\mathbf{x}}_{t}} \log p_{t}\left(\hat{\mathbf{x}}_{t} \mid \hat{\mathbf{x}}_{0}\right)\right),
> \end{aligned}
> \]
> where $s_{\phi}^{M_{E}}$ denotes the score model equipped with a specific attention mask $M_{E}$. In expectation across noise levels $t$ and the data, this results in
> \[
> \mathcal{L}(\phi)=\mathbb{E}_{M_{C}, t, \hat{\mathbf{x}}_{0}, \hat{\mathbf{x}}_{t}}\left[\left\|\ell\left(\phi, M_{C}, t, \hat{\mathbf{x}}_{0}, \hat{\mathbf{x}}_{t}\right)\right\|_{2}^{2}\right] .
> \]
> 
> We note that to simplify notation, $M_{E}$ remains fixed here, but as stated in Sec. 3.2, it might depend on the condition state or input.
> 
> After having trained the Simformer, it can straightforwardly sample arbitrary conditionals (Fig. 3). We draw samples from the noise distribution and run the reverse diffusion process on all unobserved variables, while keeping observed variables constant at their conditioning value (Weilbach et al., 2023). Having access to all conditional distributions also allows us to combine scores and thereby perform inference for simulators with i.i.d. datapoints (Geffner et al.,
> 2023). Similarly, we can use other score transformations to adapt to other prior or likelihood configurations post-hoc (see Appendix Sec. A1.4).

### task_002_check_004

Implement the planned VP option consistently with beta(t)=0.01+t*(10-0.01), drift=-0.5*beta(t)*x and diffusion=sqrt(beta(t)). The marginal mean coefficient is exp(-0.5*integral_0^t beta), with variance 1-exp(-integral_0^t beta). Document the missing x factor in the printed drift as a source correction. The main experiment configurations must select VE; extra VP benchmark runs are not required.

> **paper.md**: For VPSDE:
> \[
> f_{\mathrm{VPSDE}}(x, t)=-0.5 \cdot\left(\beta_{\min }+t \cdot\left(\beta_{\max }-\beta_{\min }\right)\right), \quad g_{\mathrm{VPSDE}}(t)=\sqrt{\beta_{\min }+t \cdot\left(\beta_{\max }-\beta_{\min }\right)}
> \]

> **paper.md**: We set $\sigma_{\max }=15, \sigma_{\min }=0.0001, \beta_{\min }=0.01$, and $\beta_{\max }=10$ for all experiments. Both for the time interval $[1 e-5,1$.$] .$
> For implementing Neural Posterior Estimation (NPE), Neural Ratio Estimation (NRE), and Neural Likelihood Estimation (NLE), we utilize the sbi library (Tejero-Cantero et al., 2020), adopting default parameters but opting for a more expressive neural spline flow for NPE and NLE. Each method was trained using the provided training loop with a batch size of 1000 and an Adam optimizer. Training ceased upon convergence, as indicated by early stopping based on validation loss.

> **paper.md**: \subsection*{4.1. Benchmark tasks}
> 
> We evaluated performance in approximating posterior distributions across four benchmark tasks (Lueckmann et al., 2021). For each task, samples for ten ground-truth posteriors are available (Appendix Sec. A2.2), and we assessed performance as classifier two-sample test (C2ST) accuracy to these samples. Here, a score of 0.5 signifies perfect alignment with the ground truth posterior, and 1.0 indicates that a classifier can completely distinguish between the approximation and the ground truth. All results are obtained using the Variance Exploding SDE (VESDE); additional results using the Variance Preserving SDE (VPSDE) can be found in Appendix Sec. A3. See Appendix Sec. A2 for details on the parameterization.
> ![](assets/asset_4.jpg)
> 
> Figure 4. Simformer performance on benchmark tasks. The suffices "undirected graph" and "directed graph" denote Simformer variants with structured attention based on the respective graphical models. (a) Classifier Two-Sample Test (C2ST) accuracy between Simformer- and ground-truth posteriors. (b) C2ST between arbitrary Simformer-conditional distributions and their ground truth.
> 
> Across all four benchmark tasks, the Simformer outperformed neural posterior estimation (NPE), even when the Simformer used a dense attention mask (Fig. 4a). The only exception was the Gaussian linear task with 10k simulations; we show an extended comparison with NRE and NLE in Appendix Fig. A5, results with VPSDE in Appendix Fig. A6). Incorporating domain knowledge into the attention mask of the transformer led to further improvements in the accuracy of the Simformer, particularly in tasks with sparser dependency structures, such as the Linear Gaussian (fully factorized) and SLCP (4 i.i.d. observations). Averaged across all benchmark tasks and observations, the Simformer required about 10 times fewer simulations than NPE, leading to a vast reduction of computational cost for amortized inference.
> 
> Next, we evaluated the ability of the Simformer to evaluate arbitrary conditionals. Arbitrary parameter and data conditions often vastly differ from the form of the posterior distribution, leading to a challenging inference task (Fig. 3). We performed inference on two of the benchmark tasks and established two new tasks with particularly interesting dependencies (Tree and HMM, details in Appendix Sec. A2.2). For each of the tasks, we generated ground truth posterior samples with Markov-Chain Monte-Carlo on 100 randomly selected conditional or full joint distributions. We found that, despite the complexity of these tasks, Simformer was able to accurately model all conditionals across all tasks
> (Fig. 4b). We note that training solely on the posterior mask does not enhance performance relative to learning all conditional distributions (Appendix Sec. A3). Further, Simformer is well calibrated (Appendix Fig. A9, Fig. A10, Fig. A11, Fig. A12) and, in most cases, also superior with respect to the loglikelihood (Appendix Fig. A8).

> **addendum.md**: In Section 4.1, the experiments related to the calibration and loglikelihood of Simformer are not required for replication.
> 
> In Section 4.3, the experiments related to the calibration of Simformer are not required for replication.
> 
> In Section 4.4, the experiments related to "Additional details and results on guidance are in Appendix Sec. A3.3" are not required for replication.

### task_002_check_005

Implement a per-example squared denoising-score error multiplied by a positive explicit time weight before batch reduction. For this diagnostic use lambda(t)=g(t)**2 (or a mathematically equivalent explicit weighting), and preserve the unobserved-coordinate mask. A helper defining lambda without its use in the training loss does not satisfy this criterion.

> **paper.md**: where $\lambda$ denotes a positive weighting function.

> **paper.md**: Having defined the tokenizer which processes every $(\boldsymbol{\theta}, \boldsymbol{x})$ pair and the attention mask to specify dependencies within the simulator, the Simformer can be trained using denoising score-matching (Hyvärinen \& Dayan, 2005; Song et al., 2021b): We sample the noise level $t$ for the diffusion model uniformly at random and generate a (partially) noisy sample $\hat{\mathbf{x}}_{t}^{M_{C}}=\left(1-M_{C}\right) \cdot \hat{\mathbf{x}}_{t}+M_{C} \cdot \hat{\mathbf{x}}_{0}$ i.e. variables that we want to condition on remain clean. The loss can then be defined as
> \[
> \begin{aligned}
> & \ell\left(\phi, M_{C}, t, \hat{\mathbf{x}}_{0}, \hat{\mathbf{x}}_{t}\right)= \\
> & \quad\left(1-M_{C}\right) \cdot\left(s_{\phi}^{M_{E}}\left(\hat{\mathbf{x}}_{t}^{M_{C}}, t\right)-\nabla_{\hat{\mathbf{x}}_{t}} \log p_{t}\left(\hat{\mathbf{x}}_{t} \mid \hat{\mathbf{x}}_{0}\right)\right),
> \end{aligned}
> \]

### task_002_check_006

For positive step h on a decreasing time grid, implement reverse Euler-Maruyama as x_next=x-(f(x,t)-g(t)**2*score(x,t))*h+g(t)*sqrt(h)*epsilon, with fresh standard Gaussian noise. An equivalent noise sign is valid. Use 500 reverse steps by default and actual endpoint differences; noise-free ODE integration or a DDPM update is not a replacement.

> **paper.md**: Samples from the generative model are then generated by simulating the reverse diffusion process (Anderson, 1982)
> \[
> d \hat{\boldsymbol{x}}_{t}=\left[f\left(\hat{\boldsymbol{x}}_{t}, t\right)-g(t)^{2} s\left(\hat{\boldsymbol{x}}_{t}, t\right)\right] d t+g(t) d \tilde{\boldsymbol{w}}
> \]

> **paper.md**: For inference, we solved the reverse SDE using an Euler-Maruyama discretization. We use 500 steps by default; accuracy for different budgets is shown in Fig. A7.

### task_002_check_007

Initialize only unobserved coordinates from the terminal noise distribution, pass the fixed observed values and condition state to the score model, and keep observed coordinates exactly constant after every reverse update. Support joint, posterior, likelihood and mixed parameter/data queries. Return samples in a documented variable order; implement all-observed and all-unobserved edge cases without NaNs.

> **paper.md**: We draw samples from the noise distribution and run the reverse diffusion process on all unobserved variables, while keeping observed variables constant at their conditioning value (Weilbach et al., 2023).

## task_003 — Transformer architecture with attention masks

Implement the transformer backbone for the Simformer. The transformer uses token dimension fifty, random Gaussian Fourier embedding for time with dimension one hundred twenty-eight, multiple layers, multiple attention heads, attention size ten, and widening factor three for the feed-forward hidden dimension. Implement multi-head self-attention with configurable attention masks M_E. Add linear projection to the output of each feed-forward block. Support both directed and undirected attention masks. Diagnostic dimensional convention: diffusion time uses 128 sampled frequencies and concatenated sine/cosine features of width 256 before each block projection. This is a disclosed rubric-assisted choice where the printed paper says 128-dimensional; metadata Fourier features are separately width 128 before projection. Token width is 50, while each of four attention heads has Q/K/V width 10.

### task_003_check_001

Use an encoder-only transformer with multi-head self-attention, feed-forward blocks, residual paths and layer normalization. Accept diffusion time, token sequence and attention mask, preserve batch/token identity, and finish with a shared single linear map from each token hidden state to one scalar score. Return shape [batch,variables] (or a documented trailing singleton); no token pooling, batch/token diagonal extraction, vector-valued score with later slicing, or nonlinear multi-layer output head.

> **paper.md**: Figure 2. Simformer architecture. All variables (parameters and data) are reduced to a token representation which includes the variables' identity, the variables' value (val) as well as the conditional state (latent ( L ) or conditioned (C)). This sequence of tokens is processed by a transformer model; the interaction of variables can be explicitly controlled through an attention mask. The transformer architecture returns a score that is used to generate samples from the score-based diffusion model and can be modified (e.g. to guide the diffusion process).

> **paper.md**: The employed transformer model features a token dimension of 50 and represents diffusion time through a 128 -dimensional random Gaussian Fourier embedding. It comprises 6 layers and 4 heads with an attention size of 10, and a widening factor of 3, implying that the feed-forward block expands to a hidden dimension of 150. For the Lotka-Volterra, SIR, and Hodgkin-Huxley tasks, we increased the number of layers to 8. Similar to the above, we used a training batch size of 1000 and an Adam optimizer.

### task_003_check_002

Use token/model width 50 and four attention heads, each with query/key/value width 10. Project 50-dimensional tokens to 4*10-dimensional Q/K/V and the concatenated head outputs back to 50. Do not silently change model width to 40/48/52/64 to accommodate a stock attention layer that requires divisibility by four.

> **paper.md**: The employed transformer model features a token dimension of 50 and represents diffusion time through a 128 -dimensional random Gaussian Fourier embedding. It comprises 6 layers and 4 heads with an attention size of 10, and a widening factor of 3, implying that the feed-forward block expands to a hidden dimension of 150. For the Lotka-Volterra, SIR, and Hodgkin-Huxley tasks, we increased the number of layers to 8. Similar to the above, we used a training batch size of 1000 and an Adam optimizer.

### task_003_check_003

Use feed-forward hidden width 150 (widening factor 3), six transformer layers for Section 4.1 benchmarks and eight for Lotka-Volterra, SIRD and Hodgkin-Huxley. Expose and actually use the layer count in each application constructor; a common constructor default alone does not establish the requested per-experiment depth.

> **paper.md**: The employed transformer model features a token dimension of 50 and represents diffusion time through a 128 -dimensional random Gaussian Fourier embedding. It comprises 6 layers and 4 heads with an attention size of 10, and a widening factor of 3, implying that the feed-forward block expands to a hidden dimension of 150. For the Lotka-Volterra, SIR, and Hodgkin-Huxley tasks, we increased the number of layers to 8. Similar to the above, we used a training batch size of 1000 and an Adam optimizer.

### task_003_check_004

Use random Gaussian Fourier features of diffusion time. For this rubric-assisted diagnostic choose 128 fixed sampled frequencies, concatenate sine and cosine into 256 features, and linearly project to width 50. This explicit diagnostic choice resolves the paper-versus-rubric width discrepancy for this input; it is not a claim that the paper explicitly states 256. Keep time features separate from metadata features.

> **paper.md**: The employed transformer model features a token dimension of 50 and represents diffusion time through a 128 -dimensional random Gaussian Fourier embedding. It comprises 6 layers and 4 heads with an attention size of 10, and a widening factor of 3, implying that the feed-forward block expands to a hidden dimension of 150. For the Lotka-Volterra, SIR, and Hodgkin-Huxley tasks, we increased the number of layers to 8. Similar to the above, we used a training batch size of 1000 and an Adam optimizer.

> **addendum.md**: Diffusion time is embedded as a random Gaussian Fourier embedding, and a linear projection is added to the output of each feed-forward block in the transformer

### task_003_check_005

Add a learned linear projection of diffusion-time features to the output of every transformer feed-forward block, broadcasting per-example time across tokens. Time injection solely into input tokens, only one block, or an unused projection does not satisfy the specified architecture.

> **addendum.md**: Diffusion time is embedded as a random Gaussian Fourier embedding, and a linear projection is added to the output of each feed-forward block in the transformer

> **paper.md**: The employed transformer model features a token dimension of 50 and represents diffusion time through a 128 -dimensional random Gaussian Fourier embedding. It comprises 6 layers and 4 heads with an attention size of 10, and a widening factor of 3, implying that the feed-forward block expands to a hidden dimension of 150. For the Lotka-Volterra, SIR, and Hodgkin-Huxley tasks, we increased the number of layers to 8. Similar to the above, we used a training batch size of 1000 and an Adam optimizer.

### task_003_check_006

Apply each sample's attention mask to query-key logits before softmax in every layer/head, with explicit allowed/blocked boolean semantics and self-attention on the diagonal. Support a per-batch-item mask without leaking masks across examples. Dense means every real token can attend to every other real token; padding, if used, remains excluded.

> **paper.md**: These constraints can be implemented as undirected (via a symmetric attention mask) or as directed dependencies (via a non-symmetric attention mask), that allow to enforce causal relations between parameters and observations. The latter, however, requires updating the mask if dependencies change i.e., due to conditioning (Webb et al., 2018) (Fig. 2, Appendix Sec. A1.1).

## task_004 — Tokenizer for simulation-based inference variables

Implement the tokenizer that converts parameters and data into transformer-compatible tokens. Each variable becomes a token comprising an identifier embedding unique to each variable, a value embedding created by repeating the scalar value to match the desired dimensionality, optional metadata embeddings such as random Fourier embeddings for function-valued parameters, and a condition state embedding where True values are projected via learnable embeddings and False values are projected to zeros. Concatenate these embeddings in order. Handle both parametric finite-dimensional and nonparametric function-valued variables. Support conditional mask resampling during training where M_C is uniformly sampled from joint, posterior, likelihood, rand_mask1 with Bernoulli probability point-three, and rand_mask2 with Bernoulli probability point-seven options.

### task_004_check_001

Assign integer identifiers by scientific variable identity, with learned embedding parameters. Multiple time points/observations of the same function or population share its identifier and are distinguished by metadata; independent scalar parameter identities remain distinct. Fixed one-hot vectors and a new learned ID for every sequence position are not equivalent.

> **paper.md**: The tokenizer represents each variable as an identifier that uniquely identifies the variable, a representation of the value of the variable, and a condition state (Fig. 2). The condition state is a binary variable and signifies whether the variable is conditioned on or not. It is resampled for every $(\boldsymbol{\theta}, \boldsymbol{x}) \in \mathbb{R}^{d}$ pair at every iteration of training. We denote the condition state of all variables as $M_{C} \in\{0,1\}^{d}$. Setting $M_{C}=(0, \ldots, 0)$ corresponds to an unconditional diffusion model (Song et al., 2021b), whereas adopting $M_{C}^{(i)}=1$ for data and $M_{C}^{(i)}=0$ for parameters corresponds to training a conditional diffusion model of the posterior distribution (Simons et al., 2023; Geffner et al., 2023).

> **paper.md**: The Simformer uses learnable vector embeddings for identifiers and condition states, as proposed in Weilbach et al. (2023). In cases where parameters or data are functions
> ![](assets/asset_2.jpg)
> 
> Figure 2. Simformer architecture. All variables (parameters and data) are reduced to a token representation which includes the variables' identity, the variables' value (val) as well as the conditional state (latent ( L ) or conditioned (C)). This sequence of tokens is processed by a transformer model; the interaction of variables can be explicitly controlled through an attention mask. The transformer architecture returns a score that is used to generate samples from the score-based diffusion model and can be modified (e.g. to guide the diffusion process).
> of space or time, the node identifier will comprise a shared embedding vector and a random Fourier embedding of the elements in the index set.

### task_004_check_002

Embed each scalar value by repeating that scalar along the configured value-embedding dimension. Preserve its continuous numeric value and gradient; do not replace repetition by a learned scalar projection, discretized token ID or positional feature.

> **addendum.md**: # Tokenization
> To correctly compute the embedding for values, the embedding for a single value is created by repeating the scalar value to match the desired dimensionality. For example, to embed the value 1 to desired dimensionality N, we would have a vector [1, 1, ...., 1] of length N.

### task_004_check_003

For function-valued parameters or observations, encode the actual index-set coordinates such as observation time using random Fourier features, followed by a learned linear projection to the metadata field width. For this diagnostic the Fourier feature vector before projection has 128 components (e.g. 64 sine/cosine pairs). Route this field into the token; a diffusion-time embedding does not provide observation-time metadata.

> **paper.md**: The Simformer uses learnable vector embeddings for identifiers and condition states, as proposed in Weilbach et al. (2023). In cases where parameters or data are functions
> ![](assets/asset_2.jpg)
> 
> Figure 2. Simformer architecture. All variables (parameters and data) are reduced to a token representation which includes the variables' identity, the variables' value (val) as well as the conditional state (latent ( L ) or conditioned (C)). This sequence of tokens is processed by a transformer model; the interaction of variables can be explicitly controlled through an attention mask. The transformer architecture returns a score that is used to generate samples from the score-based diffusion model and can be modified (e.g. to guide the diffusion process).
> of space or time, the node identifier will comprise a shared embedding vector and a random Fourier embedding of the elements in the index set.

> **paper.md**: The employed transformer model features a token dimension of 50 and represents diffusion time through a 128 -dimensional random Gaussian Fourier embedding. It comprises 6 layers and 4 heads with an attention size of 10, and a widening factor of 3, implying that the feed-forward block expands to a hidden dimension of 150. For the Lotka-Volterra, SIR, and Hodgkin-Huxley tasks, we increased the number of layers to 8. Similar to the above, we used a training batch size of 1000 and an Adam optimizer.

### task_004_check_004

Use a shared trainable condition vector for True/observed and an exact zero vector for False/unobserved. These flags must agree with the loss/sampler convention that True means held fixed. Two nonzero class embeddings or an inverted boolean convention are not equivalent.

> **addendum.md**: To embed the condition mask, the tokenizer uses learnable vector embeddings to embed each value - "True" values are projected using the learnable vector embeddings (of whatever dimensionality), whereas "False" values are projected to zeros (of the desired dimensionality).

> **addendum.md**: During training, for each element in a batch, the condition mask $M_C$ is sampled uniformly at random from the following options:
> - joint mask (all False)
> - posterior mask (all "parameter" variables are False, all "data" variables are True)
> - likelihood mask (all "data" variables are False, all "parameter" variables are True)
> - rand_mask1 ~ Ber(0.3) (resampled for every element)
> - rand_mask2 ~ Ber(0.7) (resampled for every element)

### task_004_check_005

Concatenate identifier, repeated value, optional metadata, and condition-state embeddings in that order. Their configured widths must yield token width 50, with a documented layout for tasks without metadata. Do not add the fields together. Input scalar values, integer IDs, condition mask and optional metadata must share the same sample and variable axes.

> **addendum.md**: For each input, the tokenizer concatenates the embeddings of the identifier, value, metadata (if used), and condition state in that order

> **paper.md**: The employed transformer model features a token dimension of 50 and represents diffusion time through a 128 -dimensional random Gaussian Fourier embedding. It comprises 6 layers and 4 heads with an attention size of 10, and a widening factor of 3, implying that the feed-forward block expands to a hidden dimension of 150. For the Lotka-Volterra, SIR, and Hodgkin-Huxley tasks, we increased the number of layers to 8. Similar to the above, we used a training batch size of 1000 and an Adam optimizer.

> **paper.md**: The tokenizer represents each variable as an identifier that uniquely identifies the variable, a representation of the value of the variable, and a condition state (Fig. 2). The condition state is a binary variable and signifies whether the variable is conditioned on or not. It is resampled for every $(\boldsymbol{\theta}, \boldsymbol{x}) \in \mathbb{R}^{d}$ pair at every iteration of training. We denote the condition state of all variables as $M_{C} \in\{0,1\}^{d}$. Setting $M_{C}=(0, \ldots, 0)$ corresponds to an unconditional diffusion model (Song et al., 2021b), whereas adopting $M_{C}^{(i)}=1$ for data and $M_{C}^{(i)}=0$ for parameters corresponds to training a conditional diffusion model of the posterior distribution (Simons et al., 2023; Geffner et al., 2023).

### task_004_check_006

For every element of every training batch independently choose uniformly among five mask families: joint all False; posterior parameters False/data True; likelihood parameters True/data False; independently resampled per-variable Bernoulli(0.3); independently resampled per-variable Bernoulli(0.7). Use 0.7, correcting the paper typo p=7. Choosing one family for an entire batch or reusing a fixed random mask changes the specified protocol.

> **addendum.md**: During training, for each element in a batch, the condition mask $M_C$ is sampled uniformly at random from the following options:
> - joint mask (all False)
> - posterior mask (all "parameter" variables are False, all "data" variables are True)
> - likelihood mask (all "data" variables are False, all "parameter" variables are True)
> - rand_mask1 ~ Ber(0.3) (resampled for every element)
> - rand_mask2 ~ Ber(0.7) (resampled for every element)

### task_004_check_007

Support variable observation counts and arbitrary requested index-set points using shared variable identifiers plus physical-time metadata. Preserve which tokens are parameters versus data when constructing likelihood/posterior masks. Function values at query times must be model variables, not post-hoc interpolation of a fixed set of output neurons.

> **paper.md**: The Simformer uses learnable vector embeddings for identifiers and condition states, as proposed in Weilbach et al. (2023). In cases where parameters or data are functions
> ![](assets/asset_2.jpg)
> 
> Figure 2. Simformer architecture. All variables (parameters and data) are reduced to a token representation which includes the variables' identity, the variables' value (val) as well as the conditional state (latent ( L ) or conditioned (C)). This sequence of tokens is processed by a transformer model; the interaction of variables can be explicitly controlled through an attention mask. The transformer architecture returns a score that is used to generate samples from the score-based diffusion model and can be modified (e.g. to guide the diffusion process).
> of space or time, the node identifier will comprise a shared embedding vector and a random Fourier embedding of the elements in the index set.

> **paper.md**: Figure 6. Inference of $\infty$-dim parameter space in the SIRD model. (a) Inferred posterior for global parameters (upper left) and timedependent local parameters (upper right) based on five observations (crosses) of infected (I), recovered (R), and death (D) population densities. The black dot and dashed line indicate the true parameter, bold lines indicate the mean, and shaded areas represent $99 \%$ quantiles. (b) Inference with parameter measurements and a single measurement of fatalities.

> **paper.md**: Figure 5. Inference with unstructured observations in the LotkaVolterra model. (a) Posterior predictive (left) and posterior distribution (right) based on four unstructured observations of the prey population density (green crosses), using Simformer with $10^{5}$ simulations. True parameters in dark blue. (b) Same as (a) with nine additional observations of the predator population density. (c) C2ST-performance in estimating arbitrary conditionals (right) or the posterior distribution (left) using the C2ST metric.

## task_005 — Graph inversion algorithm for attention mask adaptation

Implement the Graph Inversion algorithm by Webb et al. to dynamically adapt the directed attention mask M_E based on conditioning. Given a joint Bayesian network structure G as the base mask M_E and latent variables Z defined by condition mask M_C, the algorithm moralizes the graph, performs min-fill elimination ordering, and returns additional edges H that must be added to M_E to faithfully represent dependencies for the given conditioning. The final attention mask combines the base mask with the inverted edges. This enables the Simformer to correctly model arbitrary conditionals from a single trained model.

### task_005_check_001

Treat the base Bayesian graph as a DAG without self-loops for graph operations, and set Z to the unobserved nodes from M_C. Moralize by removing direction and connecting every pair of co-parents; set all vertices unmarked; initialize H with all variables and no edges. Self-attention loops belong to the final attention mask, not DAG parent tests.

> **addendum.md**: Algorithm Graph Inversion by Webb et al. 2018:
> 1. Input: Joint Bayesian net structure $G$ as mask $M_E$, latent variables $Z$ as given by $M_C$.
> 2. $J \leftarrow \text{MORALIZE}(G)$ # Make undirected + connect parents.
> 3. Set all vertices of $J$ to be unmarked.
> 4. $H \leftarrow \{ \text{VARIABLES}(G), \emptyset \}$, i.e. unconnected graph.
> 5. $S \leftarrow$ all latent variables without latent parent in $G$.
> 6. while $S \neq \emptyset$ do
> 7.    Select $v \in S$ according to min-fill criterion # Node that adds fewest edges below.
> 8.   Add edges in $J$ between unmarked neighbours of $v$.
> 9.   Make unmarked neighbours of $v$ in $J$, $v$'s parents in $H$.
> 10.   Mark $v$ and remove from $S$.
> 11.   for unmarked child latents $u$ of $v$ in $G$ do
> 12.    Add $u$ to $S$ if all its parent latents in $G$ are marked.
> 13.   end for
> 14. end while
> 15. return $H$.
> 
> To produce the final attention mask, the edges in $H$ are added to the base attention mask $M_E$.

### task_005_check_002

Initialize the eligible set S to latent nodes with no latent parent in the original G. At each iteration select an eligible node minimizing the number of missing edges between its currently unmarked J-neighbours; a documented deterministic tie-break is acceptable. Eliminating arbitrary nodes, ignoring the eligibility set or using degree as a substitute for min-fill is not equivalent.

> **addendum.md**: Algorithm Graph Inversion by Webb et al. 2018:
> 1. Input: Joint Bayesian net structure $G$ as mask $M_E$, latent variables $Z$ as given by $M_C$.
> 2. $J \leftarrow \text{MORALIZE}(G)$ # Make undirected + connect parents.
> 3. Set all vertices of $J$ to be unmarked.
> 4. $H \leftarrow \{ \text{VARIABLES}(G), \emptyset \}$, i.e. unconnected graph.
> 5. $S \leftarrow$ all latent variables without latent parent in $G$.
> 6. while $S \neq \emptyset$ do
> 7.    Select $v \in S$ according to min-fill criterion # Node that adds fewest edges below.
> 8.   Add edges in $J$ between unmarked neighbours of $v$.
> 9.   Make unmarked neighbours of $v$ in $J$, $v$'s parents in $H$.
> 10.   Mark $v$ and remove from $S$.
> 11.   for unmarked child latents $u$ of $v$ in $G$ do
> 12.    Add $u$ to $S$ if all its parent latents in $G$ are marked.
> 13.   end for
> 14. end while
> 15. return $H$.
> 
> To produce the final attention mask, the edges in $H$ are added to the base attention mask $M_E$.

### task_005_check_003

For selected v, add all fill edges between unmarked neighbours in J and make those neighbours parents of v in H. Mark v, remove it from S, and inspect its unmarked latent children in original G; add a child only when all its latent parents in G are marked. Continue until no eligible nodes remain. Keep the evolving J, original G and output H separate.

> **addendum.md**: Algorithm Graph Inversion by Webb et al. 2018:
> 1. Input: Joint Bayesian net structure $G$ as mask $M_E$, latent variables $Z$ as given by $M_C$.
> 2. $J \leftarrow \text{MORALIZE}(G)$ # Make undirected + connect parents.
> 3. Set all vertices of $J$ to be unmarked.
> 4. $H \leftarrow \{ \text{VARIABLES}(G), \emptyset \}$, i.e. unconnected graph.
> 5. $S \leftarrow$ all latent variables without latent parent in $G$.
> 6. while $S \neq \emptyset$ do
> 7.    Select $v \in S$ according to min-fill criterion # Node that adds fewest edges below.
> 8.   Add edges in $J$ between unmarked neighbours of $v$.
> 9.   Make unmarked neighbours of $v$ in $J$, $v$'s parents in $H$.
> 10.   Mark $v$ and remove from $S$.
> 11.   for unmarked child latents $u$ of $v$ in $G$ do
> 12.    Add $u$ to $S$ if all its parent latents in $G$ are marked.
> 13.   end for
> 14. end while
> 15. return $H$.
> 
> To produce the final attention mask, the edges in $H$ are added to the base attention mask $M_E$.

### task_005_check_004

Form the final directed attention mask as base M_E union edges(H), retaining its diagonal. Translate parent-to-child graph edges into the chosen query-row/key-column convention explicitly. Recompute or cache by the full condition mask and base graph, so two different conditionings may receive different valid masks in training and inference. A fixed triangular, symmetrized or dense fallback is not the directed variant.

> **addendum.md**: Algorithm Graph Inversion by Webb et al. 2018:
> 1. Input: Joint Bayesian net structure $G$ as mask $M_E$, latent variables $Z$ as given by $M_C$.
> 2. $J \leftarrow \text{MORALIZE}(G)$ # Make undirected + connect parents.
> 3. Set all vertices of $J$ to be unmarked.
> 4. $H \leftarrow \{ \text{VARIABLES}(G), \emptyset \}$, i.e. unconnected graph.
> 5. $S \leftarrow$ all latent variables without latent parent in $G$.
> 6. while $S \neq \emptyset$ do
> 7.    Select $v \in S$ according to min-fill criterion # Node that adds fewest edges below.
> 8.   Add edges in $J$ between unmarked neighbours of $v$.
> 9.   Make unmarked neighbours of $v$ in $J$, $v$'s parents in $H$.
> 10.   Mark $v$ and remove from $S$.
> 11.   for unmarked child latents $u$ of $v$ in $G$ do
> 12.    Add $u$ to $S$ if all its parent latents in $G$ are marked.
> 13.   end for
> 14. end while
> 15. return $H$.
> 
> To produce the final attention mask, the edges in $H$ are added to the base attention mask $M_E$.

> **paper.md**: These constraints can be implemented as undirected (via a symmetric attention mask) or as directed dependencies (via a non-symmetric attention mask), that allow to enforce causal relations between parameters and observations. The latter, however, requires updating the mask if dependencies change i.e., due to conditioning (Webb et al., 2018) (Fig. 2, Appendix Sec. A1.1).

### task_005_check_005

Construct task masks with actual simulator dimensions and a documented query-row/key-column orientation. Gaussian Linear uses independent theta_i-to-x_i edges; Two Moons/Gaussian Mixture use both parameter parents for each of their TWO data variables and a lower-triangular data block; SLCP uses FIVE parameter parents and four 2-by-2 lower-triangular observation blocks. Repair the addendum's illustrative 10-data/4-parameter sizes rather than adding phantom variables. Self-edges remain True.

> **addendum.md**: ## Task Dependencies
> For each task considered the attention mask representing a directed graphical model for the joint distributions is given. The undirected mask is obtained by making it undirected. Assuming the variables are in order $\theta_1, \theta_2,…, x_1, x_2,…$, these can be constructed as follows for the directed case:
> 
> ```python
> import numpy as np
> from scipy.linalg import block_diag
> # Gaussian Linear Task: Represents that data depends on parameters but is factorized across dimensions.
> M_θθ = np.eye(10)
> M_xx = np.eye(10)
> zeros = np.zeros((10, 10))
> M_θx = np.eye(10)
> M_E_gaussian = np.block([[M_θθ, zeros], [M_θx, M_xx]])
> # Two moons and Gaussian Mixture: Represents that each data variable depends on all parameters and the other data variables
> M_θθ = np.eye(2)
> M_xx = np.tril(np.ones((10, 10)))
> zeros = np.zeros((2, 10))
> M_θx = np.ones((10, 2))
> M_E_two_moons = np.block([[M_θθ, zeros], [M_θx, M_xx]])
> # SLCP: Dense parameter-data dependence
> M_θθ = np.eye(4)
> M_xx = block_diag(*[np.tril(np.ones((2, 2))) for _ in range(4)])
> zeros = np.zeros((4, 8))
> M_θx = np.ones((8, 4))
> M_E_slcp = np.block([[M_θθ, zeros], [M_θx, M_xx]])
> # Tree Structure: Diagonal is always true, follows tree dependencies
> M_E_tree = np.eye(10)
> M_E_tree[0, 1:3] = True  # θ₁, θ₂ depend on θ₀
> M_E_tree[1, 3:5] = True  # x₁, x₂ depend on θ₁
> M_E_tree[2, 5:7] = True  # x₃, x₄ depend on θ₂
> # HMM (Hidden Markov Model): Markov chain for parameters and factorized data
> M_θθ = np.eye(10) + np.diag(np.ones(9), k=-1)
> M_xx = np.eye(10)
> zeros = np.zeros((10, 10))
> M_θx = np.eye(10)
> M_E_hmm = np.block([[M_θθ, zeros], [M_θx, M_xx]])
> ```
> The mask for the Lotka Volterra task is metadata dependent, i.e., it depends on which T time points the predator or prey variables are observed. The prior hence M_θθ = I, the first two parameters are for the prey population and the last two for the predator M_θx = np.array([[1]*T + [0]*T, [1]*T + [0]*T, [0]*T + [1]*T, [0]*T + [1]*T]). The simulation is Markovian so M_x1x1 = M_x2x2 = np.eye(T) + np.diag(np.ones(T-1), k=-1). For the cross-data dependence a causal dependency is used, i.e., each prey variable depends additionally on all the past predator variables.

> **paper.md**: Gaussian Linear: The prior for the parameter $\boldsymbol{\theta}$ is a normal distribution $\mathcal{N}(0,0.1 \cdot \mathbf{I})$. The data $\boldsymbol{x}$ given $\boldsymbol{\theta}$ is generated by a Gaussian distribution $\mathcal{N}(\boldsymbol{x} ; \boldsymbol{\theta}, 0.1 \cdot \mathbf{I})$. Both $\boldsymbol{\theta}, \boldsymbol{x} \in \mathbb{R}^{10}$.

> **paper.md**: Gaussian Mixture This task, commonly referenced in Approximate Bayesian Computation (ABC) literature (Sisson et al., 2007; Beaumont et al., 2009), involves inferring the common mean of a mixture of two-dimensional Gaussian distributions with distinct covariances. The task is defined as follows. The prior for the parameters $\boldsymbol{\theta}$ is a uniform distribution, denoted as $\mathcal{U}(-10,10)$. The data $\boldsymbol{x}$ given $\boldsymbol{\theta}$ is modeled as a mixture of two Gaussian distributions:
> \[
> \boldsymbol{x} \mid \boldsymbol{\theta} \sim 0.5 \cdot \mathcal{N}(\boldsymbol{x} ; \boldsymbol{\theta}, \mathbf{I})+0.5 \cdot \mathcal{N}(x ; \boldsymbol{\theta}, 0.01 \cdot \mathbf{I})
> \]
> 
> The parameter space $\boldsymbol{\theta}$ and the data space $\boldsymbol{x}$ are both in $\mathbb{R}^{2}$.

> **paper.md**: Two Moons : The Two Moons task is designed to test inference algorithms in handling multimodal distributions (Greenberg et al., 2019). The prior is a Uniform distribution $U(\boldsymbol{\theta} ;-1,1)$. The data $\boldsymbol{x}$ is generated from $\boldsymbol{\theta}$ as:
> \[
> \boldsymbol{x} \left\lvert\, \boldsymbol{\theta}=\left[\begin{array}{c}
> r \cos (\alpha)+0.25 \\
> r \sin (\alpha)
> \end{array}\right]+\left[\begin{array}{c}
> -\left|\theta_{1}+\theta_{2}\right| / \sqrt{2} \\
> \left(-\theta_{1}+\theta_{2}\right) / \sqrt{2}
> \end{array}\right]\right.,
> \]
> ![](assets/asset_12.jpg)
> 
> Figure A4. Directed base masks for each of the tasks. The Lotka Volterra mask dynamically adapts to different input times, here just for randomly selected times.
> where $\alpha \sim \mathcal{U}(-\pi / 2, \pi / 2)$ and $r \sim \mathcal{N}(0.1,0.012)$. Leading to a dimensionality $\boldsymbol{\theta} \in \mathbb{R}^{2}, \boldsymbol{x} \in \mathbb{R}^{2}$.
> To obtain reference samples for all possible conditionals, we run the following procedure:
> - We initialized $N$ Markov chains with samples from the joint distribution.
> - We run 1000 steps of a random direction slice sampling algorithm.
> - We run an additional 3000 steps of MHMCMC with step size of 0.01.
> - Only the last samples of each chain were considered, yielding $N$ reference samples.
> 
> This procedure yielded samples in agreement with the reference posterior provided by Lueckmann et al. (2021) (C2ST $\sim 0.5$ ). Other conditionals did also look correct, but were not extensively investigated.

> **paper.md**: SLCP Task: The SLCP (Simple Likelihood Complex Posterior) task is a challenging inference task designed to generate a complex posterior distribution (Papamakarios et al., 2019; Greenberg et al., 2019; Hermans et al., 2020; Durkan et al., 2020). The setup is as follows. The prior over $\boldsymbol{\theta}$ is a uniform distribution $\mathcal{U}(-3,3)$. The data $\boldsymbol{x}$ given $\boldsymbol{\theta}$ is $\boldsymbol{x}=\left(\boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{4}\right)$, where each $\boldsymbol{x}_{i} \sim \mathcal{N}\left(\mu_{\theta}, \boldsymbol{\Sigma}_{\theta}\right)$ with:
> \[
> \begin{aligned}
> \mu_{\theta} & =\left[\begin{array}{l}
> \theta_{1} \\
> \theta_{2}
> \end{array}\right], \\
> \boldsymbol{\Sigma}_{\theta} & =\left[\begin{array}{cc}
> \theta_{3}^{2} & \tanh \left(\theta_{5}\right) \cdot \theta_{3}^{2} \cdot \theta_{4}^{2} \\
> \tanh \left(\theta_{5}\right) \cdot \theta_{3}^{2} \cdot \theta_{4}^{2} & \theta_{4}^{2}
> \end{array}\right] .
> \end{aligned}
> \]
> 
> Leading to a dimensionality of $\boldsymbol{\theta} \in \mathbb{R}^{5}, \boldsymbol{x} \in \mathbb{R}^{8}$.
> To obtain reference samples for all possible conditionals, we run the following procedure:
> - We initialized $N$ Markov chains with samples from the joint distribution.
> - We run 600 steps of a random direction slice sampling algorithm.
> - We run an additional 2000 steps of MHMCMC with a step size of 0.1.
> - Only the last samples of each chain was considered, yielding $N$ reference samples.
> 
> This procedure yielded samples in agreement with the reference posterior provided by Lueckmann et al. (2021) (C2ST $\sim 0.5$ ). Other conditionals did also look correct, but were not extensively investigated.

### task_005_check_006

For Tree use exactly 3 parameter and 4 data nodes: theta0 parents theta1/theta2, theta1 parents x0/x1, and theta2 parents x2/x3. For HMM use 10 parameter and 10 data nodes with theta_i parenting theta_(i+1) and x_i. The addendum Tree matrix has inconsistent size/orientation; use these generative dependencies with the same orientation as other tasks. Undirected variants are the boolean union of directed base adjacency and its transpose, including the diagonal.

> **addendum.md**: ## Task Dependencies
> For each task considered the attention mask representing a directed graphical model for the joint distributions is given. The undirected mask is obtained by making it undirected. Assuming the variables are in order $\theta_1, \theta_2,…, x_1, x_2,…$, these can be constructed as follows for the directed case:
> 
> ```python
> import numpy as np
> from scipy.linalg import block_diag
> # Gaussian Linear Task: Represents that data depends on parameters but is factorized across dimensions.
> M_θθ = np.eye(10)
> M_xx = np.eye(10)
> zeros = np.zeros((10, 10))
> M_θx = np.eye(10)
> M_E_gaussian = np.block([[M_θθ, zeros], [M_θx, M_xx]])
> # Two moons and Gaussian Mixture: Represents that each data variable depends on all parameters and the other data variables
> M_θθ = np.eye(2)
> M_xx = np.tril(np.ones((10, 10)))
> zeros = np.zeros((2, 10))
> M_θx = np.ones((10, 2))
> M_E_two_moons = np.block([[M_θθ, zeros], [M_θx, M_xx]])
> # SLCP: Dense parameter-data dependence
> M_θθ = np.eye(4)
> M_xx = block_diag(*[np.tril(np.ones((2, 2))) for _ in range(4)])
> zeros = np.zeros((4, 8))
> M_θx = np.ones((8, 4))
> M_E_slcp = np.block([[M_θθ, zeros], [M_θx, M_xx]])
> # Tree Structure: Diagonal is always true, follows tree dependencies
> M_E_tree = np.eye(10)
> M_E_tree[0, 1:3] = True  # θ₁, θ₂ depend on θ₀
> M_E_tree[1, 3:5] = True  # x₁, x₂ depend on θ₁
> M_E_tree[2, 5:7] = True  # x₃, x₄ depend on θ₂
> # HMM (Hidden Markov Model): Markov chain for parameters and factorized data
> M_θθ = np.eye(10) + np.diag(np.ones(9), k=-1)
> M_xx = np.eye(10)
> zeros = np.zeros((10, 10))
> M_θx = np.eye(10)
> M_E_hmm = np.block([[M_θθ, zeros], [M_θx, M_xx]])
> ```
> The mask for the Lotka Volterra task is metadata dependent, i.e., it depends on which T time points the predator or prey variables are observed. The prior hence M_θθ = I, the first two parameters are for the prey population and the last two for the predator M_θx = np.array([[1]*T + [0]*T, [1]*T + [0]*T, [0]*T + [1]*T, [0]*T + [1]*T]). The simulation is Markovian so M_x1x1 = M_x2x2 = np.eye(T) + np.diag(np.ones(T-1), k=-1). For the cross-data dependence a causal dependency is used, i.e., each prey variable depends additionally on all the past predator variables.

> **paper.md**: Tree: This is a nonlinear tree-shaped task:
> \[
> \theta_{0} \sim \mathcal{N}\left(\theta_{0} ; 0,1 .\right) \quad \theta_{1} \sim \mathcal{N}\left(\theta_{0} ; 1 .\right) \quad \theta_{2} \sim \mathcal{N}\left(\theta_{2} ; \theta_{0}, 1 .\right)
> \]
> 
> Observable data is obtained through
> \[
> x_{0} \sim \mathcal{N}\left(x_{1} ; \sin \left(\theta_{1}\right)^{2}, 0.2^{2}\right) \quad x_{1} \sim \mathcal{N}\left(0.1 \cdot \theta_{1}^{2}, 0.2^{2}\right) \quad x_{2} \sim \mathcal{N}\left(x_{2} ; 0.1 \cdot \theta_{2}^{2}, 0.6^{2}\right) \quad x_{3} \sim \mathcal{N}\left(x_{3} ; \cos \left(\theta_{2}\right)^{2} ; 0.1^{2}\right)
> \]
> which leads to a tree-like factorization with highly multimodal conditionals.
> To obtain reference samples for all possible conditionals, we run the following procedure:
> - We initialized $N$ Markov chains with samples from the joint distribution.
> - We run 5000 steps of a HMC sampler.
> - Only the last samples of each chain were considered, yielding $N$ reference samples.

> **paper.md**: HMM: This is a task in which the parameters have a Markovian factorization.
> \[
> \theta_{0} \sim \mathcal{N}\left(\theta_{0} ; 0 ., 0.5^{2}\right) \quad \theta_{i+1} \sim \mathcal{N}\left(\theta_{i+1} ; \theta_{i}, 0.5^{2}\right)
> \]
> for $i=0, \ldots, 9$. Observations are generated according to $x_{i}=\mathcal{N}\left(x_{i} ; \theta_{i}^{2}, 0.5^{2}\right)$, leading to a nonlinear hidden Markov model with bimodal correlated posterior and leading to a dimensionality of $\boldsymbol{\theta} \in \mathbb{R}^{10}, \boldsymbol{x} \in \mathbb{R}^{10}$.
> To obtain reference samples for all possible conditionals, we run the following procedure:
> - We initialized $N$ Markov chains with samples from the joint distribution.
> - We run 5000 steps of an HMC sampler.
> - Only the last samples of each chain were considered, yielding $N$ reference samples.

## task_006 — Guided diffusion for interval conditioning

Implement guided diffusion to enable conditioning on observation intervals rather than exact values. Use the general formulation where the guided score equals the base score plus the gradient of log sigmoid applied to scaling function times constraint function. The scaling function is inversely proportional to variance. The constraint function defines inequality constraints. Implement the self-recurrence predictor-corrector scheme with configurable recurrence steps, gradient computation of constraint function through sigmoid guidance, and proper handling of the reverse SDE integration. Support multiple simultaneous constraints.

### task_006_check_001

Expose interval conditioning as an optional sampling mode with configurable upper bound u and recurrence count r. Use c(x)=x-u for an upper bound, with feasible c<=0; support lower/two-sided or multiple constraints through the same interface. Distinguish this soft interval from exact observed-value clamping, and preserve the unconditional sampling path when guidance is disabled.

> **paper.md**: In that line, we use the following general formulation to guide the diffusion process:
> \[
> s_{\phi}\left(\hat{\boldsymbol{x}}_{t}, t \mid c\right) \approx s_{\phi}\left(\hat{\boldsymbol{x}}_{t}, t\right)+\nabla_{\hat{\boldsymbol{x}}_{t}} \log \sigma\left(-s(t) c\left(\hat{\boldsymbol{x}}_{t}\right)\right)
> \]
> 
> Here $\sigma$ denotes the sigmoid function, $s(t)$ is an appropriate scaling function satisfying $s(t) \rightarrow \infty$ as $t \rightarrow 0$, depending on the choice of SDE, and $c$ denotes a constraint function $c(\hat{\boldsymbol{x}}) \leq 0$. For example, to enforce an interval upper bound $u$, we use $c(\hat{\boldsymbol{x}})=\hat{\boldsymbol{x}}-u$.

### task_006_check_002

At the current reverse state estimate the clean variables by x0_hat=(xt+sigma(t)**2*s_phi(xt,t))/mu(t), using the chosen SDE's mean coefficient and marginal standard deviation. Evaluate the constraint on this denoised estimate. Calling the constraint solely on noisy xt or using a fixed trained-data energy statistic is not Algorithm 1.

> **paper.md**: The second strategy entails adjusting the estimated score with a general constraint function, which we evaluate on a denoised estimate of the variables (Bansal et al., 2023; Chung et al., 2023; Rozet \& Louppe, 2021). Overall, this is remarkable flexibility and supports almost any constraint to be incorporated. We provide a pseudo-code in Algorithm 1. In our experimental assessments, it proved to be sufficiently accurate.

> **paper.md**: Algorithm 1 General Guidance
> Require: Number of steps $T$, Min time $T_{\min }$, Max time $T_{\max }$, self-recurrence steps $r$, scaling function $s(t)$ and constraint
>     function $c(x)$, drift coefficient $f(x, t)$, diffusion coefficient $g(t)$, associated mean and standard deviation functions $\mu, \sigma$
>     such that $\hat{\boldsymbol{x}}_{t}=\mu(t) \hat{\boldsymbol{x}}_{0}+\sigma(t) \epsilon$.
>     Set time step $\Delta t=\frac{T_{\text {max }}-T_{\text {min }}}{T}$
>     Sample $\hat{\boldsymbol{x}}_{T} \sim \mathcal{N}\left(\mu_{T}, \sigma_{T} \mathbf{I}\right) \quad / /$ Initialize at terminal distribution
>     for $i=1$ downto $T$ do
>         $t_{i}=T_{\text {max }}-i \cdot \Delta t$
>         for $j=1$ to $r$ do
>             $\epsilon \sim \mathcal{N}(0, \mathbf{I})$
>             $s=s_{\phi}\left(\hat{\boldsymbol{x}}_{t_{i+1}}, t_{i}\right) \quad / /$ Marginal score estimate
>             $\hat{\boldsymbol{x}}_{\sim 0}=\frac{\hat{\boldsymbol{x}}_{t_{i+1}}+\sigma\left(t_{i+1}\right)^{2} \cdot s}{\mu\left(t_{i+1}\right)} \quad / /$ Denoise
>             $\tilde{s}=s+\nabla_{\hat{\boldsymbol{x}}} \log \sigma\left(s(t) c\left(\hat{\boldsymbol{x}}_{\sim 0}\right)\right) \quad / /$ Constraint score
>             $\hat{\boldsymbol{x}}_{t_{i}}=\hat{\boldsymbol{x}}_{t_{i+1}}-\left(f\left(\hat{\boldsymbol{x}}_{t_{i+1}}, t_{i}\right)-g\left(t_{i}\right)^{2} \cdot \tilde{s}\right) \Delta t-g\left(t_{i}\right) \sqrt{\Delta t} \cdot \epsilon$
>             if $r>0$ then
>                     // Resample future point using SDE equations
>                 $\epsilon \sim \mathcal{N}(0, \mathbf{I})$
>                 $\hat{\boldsymbol{x}}_{t_{i+1}}=\hat{\boldsymbol{x}}_{t_{i}}+f\left(\hat{\boldsymbol{x}}_{t_{i+1}}, t_{i}\right) \Delta t+g\left(t_{i}\right) \sqrt{\Delta t} \cdot \epsilon$
>             end if
>         end for
>     end for
>     return $\hat{\boldsymbol{x}}_{T_{\text {min }}}$
> ```

### task_006_check_003

For the c<=0 convention add grad_xt log(sigmoid(-s(t)*c(x0_hat))) to the base score, with s(t)=1/sigma(t)**2. Use a numerically stable log-sigmoid and maintain the derivative through the denoised estimate to xt. Resolve the printed Algorithm 1 sign discrepancy by this Section 3.4 convention (or use the exactly equivalent reversed definition of c). An upper-energy constraint must push energy downward, not upward.

> **paper.md**: In that line, we use the following general formulation to guide the diffusion process:
> \[
> s_{\phi}\left(\hat{\boldsymbol{x}}_{t}, t \mid c\right) \approx s_{\phi}\left(\hat{\boldsymbol{x}}_{t}, t\right)+\nabla_{\hat{\boldsymbol{x}}_{t}} \log \sigma\left(-s(t) c\left(\hat{\boldsymbol{x}}_{t}\right)\right)
> \]
> 
> Here $\sigma$ denotes the sigmoid function, $s(t)$ is an appropriate scaling function satisfying $s(t) \rightarrow \infty$ as $t \rightarrow 0$, depending on the choice of SDE, and $c$ denotes a constraint function $c(\hat{\boldsymbol{x}}) \leq 0$. For example, to enforce an interval upper bound $u$, we use $c(\hat{\boldsymbol{x}})=\hat{\boldsymbol{x}}-u$.

> **paper.md**: General guidance requires the specification of a scaling function, which up and down scales the constrained score at different diffusion times $t$. As the magnitude of the marginal score does depend on the SDE, this scaling function should also. In our experiment, we generally used a scaling function of the form $s(t)=\frac{1}{\sigma(t)^{2}}$, i.e., which is inversely proportional to the variance of the approximate marginal SDE scores.

> **paper.md**: Algorithm 1 General Guidance
> Require: Number of steps $T$, Min time $T_{\min }$, Max time $T_{\max }$, self-recurrence steps $r$, scaling function $s(t)$ and constraint
>     function $c(x)$, drift coefficient $f(x, t)$, diffusion coefficient $g(t)$, associated mean and standard deviation functions $\mu, \sigma$
>     such that $\hat{\boldsymbol{x}}_{t}=\mu(t) \hat{\boldsymbol{x}}_{0}+\sigma(t) \epsilon$.
>     Set time step $\Delta t=\frac{T_{\text {max }}-T_{\text {min }}}{T}$
>     Sample $\hat{\boldsymbol{x}}_{T} \sim \mathcal{N}\left(\mu_{T}, \sigma_{T} \mathbf{I}\right) \quad / /$ Initialize at terminal distribution
>     for $i=1$ downto $T$ do
>         $t_{i}=T_{\text {max }}-i \cdot \Delta t$
>         for $j=1$ to $r$ do
>             $\epsilon \sim \mathcal{N}(0, \mathbf{I})$
>             $s=s_{\phi}\left(\hat{\boldsymbol{x}}_{t_{i+1}}, t_{i}\right) \quad / /$ Marginal score estimate
>             $\hat{\boldsymbol{x}}_{\sim 0}=\frac{\hat{\boldsymbol{x}}_{t_{i+1}}+\sigma\left(t_{i+1}\right)^{2} \cdot s}{\mu\left(t_{i+1}\right)} \quad / /$ Denoise
>             $\tilde{s}=s+\nabla_{\hat{\boldsymbol{x}}} \log \sigma\left(s(t) c\left(\hat{\boldsymbol{x}}_{\sim 0}\right)\right) \quad / /$ Constraint score
>             $\hat{\boldsymbol{x}}_{t_{i}}=\hat{\boldsymbol{x}}_{t_{i+1}}-\left(f\left(\hat{\boldsymbol{x}}_{t_{i+1}}, t_{i}\right)-g\left(t_{i}\right)^{2} \cdot \tilde{s}\right) \Delta t-g\left(t_{i}\right) \sqrt{\Delta t} \cdot \epsilon$
>             if $r>0$ then
>                     // Resample future point using SDE equations
>                 $\epsilon \sim \mathcal{N}(0, \mathbf{I})$
>                 $\hat{\boldsymbol{x}}_{t_{i+1}}=\hat{\boldsymbol{x}}_{t_{i}}+f\left(\hat{\boldsymbol{x}}_{t_{i+1}}, t_{i}\right) \Delta t+g\left(t_{i}\right) \sqrt{\Delta t} \cdot \epsilon$
>             end if
>         end for
>     end for
>     return $\hat{\boldsymbol{x}}_{T_{\text {min }}}$
> ```

### task_006_check_004

Implement self-recurrence as a genuine reverse/forward cycle at each diffusion-time interval: integrate the guided reverse step to the next lower-noise state; before an additional recurrence, resample forward to the previous higher-noise level with the forward SDE and fresh Gaussian noise, then recompute score and guidance. Return the final LOWER-time state; do not end the outer interval on a re-noised sample or treat r as repeated deterministic gradient descent.

> **paper.md**: Algorithm 1 General Guidance
> Require: Number of steps $T$, Min time $T_{\min }$, Max time $T_{\max }$, self-recurrence steps $r$, scaling function $s(t)$ and constraint
>     function $c(x)$, drift coefficient $f(x, t)$, diffusion coefficient $g(t)$, associated mean and standard deviation functions $\mu, \sigma$
>     such that $\hat{\boldsymbol{x}}_{t}=\mu(t) \hat{\boldsymbol{x}}_{0}+\sigma(t) \epsilon$.
>     Set time step $\Delta t=\frac{T_{\text {max }}-T_{\text {min }}}{T}$
>     Sample $\hat{\boldsymbol{x}}_{T} \sim \mathcal{N}\left(\mu_{T}, \sigma_{T} \mathbf{I}\right) \quad / /$ Initialize at terminal distribution
>     for $i=1$ downto $T$ do
>         $t_{i}=T_{\text {max }}-i \cdot \Delta t$
>         for $j=1$ to $r$ do
>             $\epsilon \sim \mathcal{N}(0, \mathbf{I})$
>             $s=s_{\phi}\left(\hat{\boldsymbol{x}}_{t_{i+1}}, t_{i}\right) \quad / /$ Marginal score estimate
>             $\hat{\boldsymbol{x}}_{\sim 0}=\frac{\hat{\boldsymbol{x}}_{t_{i+1}}+\sigma\left(t_{i+1}\right)^{2} \cdot s}{\mu\left(t_{i+1}\right)} \quad / /$ Denoise
>             $\tilde{s}=s+\nabla_{\hat{\boldsymbol{x}}} \log \sigma\left(s(t) c\left(\hat{\boldsymbol{x}}_{\sim 0}\right)\right) \quad / /$ Constraint score
>             $\hat{\boldsymbol{x}}_{t_{i}}=\hat{\boldsymbol{x}}_{t_{i+1}}-\left(f\left(\hat{\boldsymbol{x}}_{t_{i+1}}, t_{i}\right)-g\left(t_{i}\right)^{2} \cdot \tilde{s}\right) \Delta t-g\left(t_{i}\right) \sqrt{\Delta t} \cdot \epsilon$
>             if $r>0$ then
>                     // Resample future point using SDE equations
>                 $\epsilon \sim \mathcal{N}(0, \mathbf{I})$
>                 $\hat{\boldsymbol{x}}_{t_{i+1}}=\hat{\boldsymbol{x}}_{t_{i}}+f\left(\hat{\boldsymbol{x}}_{t_{i+1}}, t_{i}\right) \Delta t+g\left(t_{i}\right) \sqrt{\Delta t} \cdot \epsilon$
>             end if
>         end for
>     end for
>     return $\hat{\boldsymbol{x}}_{T_{\text {min }}}$
> ```

### task_006_check_005

During both forward and reverse recurrence, update only latent coordinates and restore exact observations at every step. Define r unambiguously as either total guided reverse passes or extra cycles and handle zero/one consistently, so disabling extra recurrence still permits one reverse step. Connect the actual guided score to the reverse drift; no clipping/rejection-only implementation may replace guidance.

> **paper.md**: We draw samples from the noise distribution and run the reverse diffusion process on all unobserved variables, while keeping observed variables constant at their conditioning value (Weilbach et al., 2023).

> **paper.md**: Algorithm 1 General Guidance
> Require: Number of steps $T$, Min time $T_{\min }$, Max time $T_{\max }$, self-recurrence steps $r$, scaling function $s(t)$ and constraint
>     function $c(x)$, drift coefficient $f(x, t)$, diffusion coefficient $g(t)$, associated mean and standard deviation functions $\mu, \sigma$
>     such that $\hat{\boldsymbol{x}}_{t}=\mu(t) \hat{\boldsymbol{x}}_{0}+\sigma(t) \epsilon$.
>     Set time step $\Delta t=\frac{T_{\text {max }}-T_{\text {min }}}{T}$
>     Sample $\hat{\boldsymbol{x}}_{T} \sim \mathcal{N}\left(\mu_{T}, \sigma_{T} \mathbf{I}\right) \quad / /$ Initialize at terminal distribution
>     for $i=1$ downto $T$ do
>         $t_{i}=T_{\text {max }}-i \cdot \Delta t$
>         for $j=1$ to $r$ do
>             $\epsilon \sim \mathcal{N}(0, \mathbf{I})$
>             $s=s_{\phi}\left(\hat{\boldsymbol{x}}_{t_{i+1}}, t_{i}\right) \quad / /$ Marginal score estimate
>             $\hat{\boldsymbol{x}}_{\sim 0}=\frac{\hat{\boldsymbol{x}}_{t_{i+1}}+\sigma\left(t_{i+1}\right)^{2} \cdot s}{\mu\left(t_{i+1}\right)} \quad / /$ Denoise
>             $\tilde{s}=s+\nabla_{\hat{\boldsymbol{x}}} \log \sigma\left(s(t) c\left(\hat{\boldsymbol{x}}_{\sim 0}\right)\right) \quad / /$ Constraint score
>             $\hat{\boldsymbol{x}}_{t_{i}}=\hat{\boldsymbol{x}}_{t_{i+1}}-\left(f\left(\hat{\boldsymbol{x}}_{t_{i+1}}, t_{i}\right)-g\left(t_{i}\right)^{2} \cdot \tilde{s}\right) \Delta t-g\left(t_{i}\right) \sqrt{\Delta t} \cdot \epsilon$
>             if $r>0$ then
>                     // Resample future point using SDE equations
>                 $\epsilon \sim \mathcal{N}(0, \mathbf{I})$
>                 $\hat{\boldsymbol{x}}_{t_{i+1}}=\hat{\boldsymbol{x}}_{t_{i}}+f\left(\hat{\boldsymbol{x}}_{t_{i+1}}, t_{i}\right) \Delta t+g\left(t_{i}\right) \sqrt{\Delta t} \cdot \epsilon$
>             end if
>         end for
>     end for
>     return $\hat{\boldsymbol{x}}_{T_{\text {min }}}$
> ```

> **paper.md**: In that line, we use the following general formulation to guide the diffusion process:
> \[
> s_{\phi}\left(\hat{\boldsymbol{x}}_{t}, t \mid c\right) \approx s_{\phi}\left(\hat{\boldsymbol{x}}_{t}, t\right)+\nabla_{\hat{\boldsymbol{x}}_{t}} \log \sigma\left(-s(t) c\left(\hat{\boldsymbol{x}}_{t}\right)\right)
> \]
> 
> Here $\sigma$ denotes the sigmoid function, $s(t)$ is an appropriate scaling function satisfying $s(t) \rightarrow \infty$ as $t \rightarrow 0$, depending on the choice of SDE, and $c$ denotes a constraint function $c(\hat{\boldsymbol{x}}) \leq 0$. For example, to enforce an interval upper bound $u$, we use $c(\hat{\boldsymbol{x}})=\hat{\boldsymbol{x}}-u$.

## task_007 — Simulation-based inference benchmark tasks

Implement all five benchmark simulators from Lueckmann et al.: Gaussian Linear with prior and data distributions in R^10; Two Moons with uniform prior and data generated from polar coordinates producing bimodal posteriors; Gaussian Mixture with uniform prior and data from mixture of two Gaussians with different covariances; SLCP with uniform prior, data from bivariate normals with theta-dependent means and covariances in R^5 and R^8; Tree with hierarchical normal priors and nonlinear observation functions producing multimodal posteriors. Implement the reference posterior generation procedures using slice sampling and MCMC for each task. Scope completion: include HMM as a sixth benchmark, exact joint log-density/conditional interfaces, and the distinct N-chain reference protocols for Two Moons, SLCP, Tree and HMM. Use the dimensionally consistent task/mask definitions and disclosed source-error resolutions in the checklist. Implement the samplers; do not generate large reference datasets during Code-Dev approval.

### task_007_check_001

Implement theta~N(0,0.1 I_10), x|theta~N(theta,0.1 I_10), with the second argument interpreted as covariance. Keep theta and data both 10-dimensional; a Normal API taking standard deviation must receive sqrt(0.1). The prior sampler, simulator and joint density must use the same convention.

> **paper.md**: Gaussian Linear: The prior for the parameter $\boldsymbol{\theta}$ is a normal distribution $\mathcal{N}(0,0.1 \cdot \mathbf{I})$. The data $\boldsymbol{x}$ given $\boldsymbol{\theta}$ is generated by a Gaussian distribution $\mathcal{N}(\boldsymbol{x} ; \boldsymbol{\theta}, 0.1 \cdot \mathbf{I})$. Both $\boldsymbol{\theta}, \boldsymbol{x} \in \mathbb{R}^{10}$.

### task_007_check_002

Implement theta~Uniform([-10,10]^2) and x|theta as equal weights of N(theta,I_2) and N(theta,0.01 I_2). The component indicator is sampled per observation and both coordinates use that component; log density uses the mixture log-sum-exp, not an average Gaussian or independent per-coordinate mixture choices.

> **paper.md**: Gaussian Mixture This task, commonly referenced in Approximate Bayesian Computation (ABC) literature (Sisson et al., 2007; Beaumont et al., 2009), involves inferring the common mean of a mixture of two-dimensional Gaussian distributions with distinct covariances. The task is defined as follows. The prior for the parameters $\boldsymbol{\theta}$ is a uniform distribution, denoted as $\mathcal{U}(-10,10)$. The data $\boldsymbol{x}$ given $\boldsymbol{\theta}$ is modeled as a mixture of two Gaussian distributions:
> \[
> \boldsymbol{x} \mid \boldsymbol{\theta} \sim 0.5 \cdot \mathcal{N}(\boldsymbol{x} ; \boldsymbol{\theta}, \mathbf{I})+0.5 \cdot \mathcal{N}(x ; \boldsymbol{\theta}, 0.01 \cdot \mathbf{I})
> \]
> 
> The parameter space $\boldsymbol{\theta}$ and the data space $\boldsymbol{x}$ are both in $\mathbb{R}^{2}$.

### task_007_check_003

Implement Two Moons with theta in [-1,1]^2, alpha uniform on [-pi/2,pi/2], r of mean 0.1 and standard deviation 0.01, and x=(r*cos(alpha)+0.25-|theta1+theta2|/sqrt(2), r*sin(alpha)+(-theta1+theta2)/sqrt(2)). The printed 0.012 denotes 0.01 squared. Preserve the absolute value, rotation, offset and support; generic two-cluster Gaussians are not this simulator.

> **paper.md**: Two Moons : The Two Moons task is designed to test inference algorithms in handling multimodal distributions (Greenberg et al., 2019). The prior is a Uniform distribution $U(\boldsymbol{\theta} ;-1,1)$. The data $\boldsymbol{x}$ is generated from $\boldsymbol{\theta}$ as:
> \[
> \boldsymbol{x} \left\lvert\, \boldsymbol{\theta}=\left[\begin{array}{c}
> r \cos (\alpha)+0.25 \\
> r \sin (\alpha)
> \end{array}\right]+\left[\begin{array}{c}
> -\left|\theta_{1}+\theta_{2}\right| / \sqrt{2} \\
> \left(-\theta_{1}+\theta_{2}\right) / \sqrt{2}
> \end{array}\right]\right.,
> \]
> ![](assets/asset_12.jpg)
> 
> Figure A4. Directed base masks for each of the tasks. The Lotka Volterra mask dynamically adapts to different input times, here just for randomly selected times.
> where $\alpha \sim \mathcal{U}(-\pi / 2, \pi / 2)$ and $r \sim \mathcal{N}(0.1,0.012)$. Leading to a dimensionality $\boldsymbol{\theta} \in \mathbb{R}^{2}, \boldsymbol{x} \in \mathbb{R}^{2}$.
> To obtain reference samples for all possible conditionals, we run the following procedure:
> - We initialized $N$ Markov chains with samples from the joint distribution.
> - We run 1000 steps of a random direction slice sampling algorithm.
> - We run an additional 3000 steps of MHMCMC with step size of 0.01.
> - Only the last samples of each chain were considered, yielding $N$ reference samples.
> 
> This procedure yielded samples in agreement with the reference posterior provided by Lueckmann et al. (2021) (C2ST $\sim 0.5$ ). Other conditionals did also look correct, but were not extensively investigated.

### task_007_check_004

Implement SLCP with five Uniform(-3,3) parameters and four conditionally i.i.d. bivariate observations flattened to eight data coordinates. Use the cited benchmark's valid covariance: s1=theta3**2, s2=theta4**2, rho=tanh(theta5), diagonal=(s1**2,s2**2), off-diagonal=rho*s1*s2, mean=(theta1,theta2). Document the printed diagonal typo; do not implement its generally indefinite covariance or change rho/clamp eigenvalues to conceal it. Tiny declared numerical jitter is acceptable.

> **paper.md**: SLCP Task: The SLCP (Simple Likelihood Complex Posterior) task is a challenging inference task designed to generate a complex posterior distribution (Papamakarios et al., 2019; Greenberg et al., 2019; Hermans et al., 2020; Durkan et al., 2020). The setup is as follows. The prior over $\boldsymbol{\theta}$ is a uniform distribution $\mathcal{U}(-3,3)$. The data $\boldsymbol{x}$ given $\boldsymbol{\theta}$ is $\boldsymbol{x}=\left(\boldsymbol{x}_{1}, \ldots, \boldsymbol{x}_{4}\right)$, where each $\boldsymbol{x}_{i} \sim \mathcal{N}\left(\mu_{\theta}, \boldsymbol{\Sigma}_{\theta}\right)$ with:
> \[
> \begin{aligned}
> \mu_{\theta} & =\left[\begin{array}{l}
> \theta_{1} \\
> \theta_{2}
> \end{array}\right], \\
> \boldsymbol{\Sigma}_{\theta} & =\left[\begin{array}{cc}
> \theta_{3}^{2} & \tanh \left(\theta_{5}\right) \cdot \theta_{3}^{2} \cdot \theta_{4}^{2} \\
> \tanh \left(\theta_{5}\right) \cdot \theta_{3}^{2} \cdot \theta_{4}^{2} & \theta_{4}^{2}
> \end{array}\right] .
> \end{aligned}
> \]
> 
> Leading to a dimensionality of $\boldsymbol{\theta} \in \mathbb{R}^{5}, \boldsymbol{x} \in \mathbb{R}^{8}$.
> To obtain reference samples for all possible conditionals, we run the following procedure:
> - We initialized $N$ Markov chains with samples from the joint distribution.
> - We run 600 steps of a random direction slice sampling algorithm.
> - We run an additional 2000 steps of MHMCMC with a step size of 0.1.
> - Only the last samples of each chain was considered, yielding $N$ reference samples.
> 
> This procedure yielded samples in agreement with the reference posterior provided by Lueckmann et al. (2021) (C2ST $\sim 0.5$ ). Other conditionals did also look correct, but were not extensively investigated.

### task_007_check_005

Implement theta0~N(0,1), theta1/theta2 independently conditional on theta0 with means theta0 and standard deviations 1. Observations x0,x1,x2,x3 have respective means sin(theta1)**2,0.1*theta1**2,0.1*theta2**2,cos(theta2)**2 and standard deviations 0.2,0.2,0.6,0.1. Preserve the 3+4 dimensions and factorization in both sampling and log-joint evaluation.

> **paper.md**: Tree: This is a nonlinear tree-shaped task:
> \[
> \theta_{0} \sim \mathcal{N}\left(\theta_{0} ; 0,1 .\right) \quad \theta_{1} \sim \mathcal{N}\left(\theta_{0} ; 1 .\right) \quad \theta_{2} \sim \mathcal{N}\left(\theta_{2} ; \theta_{0}, 1 .\right)
> \]
> 
> Observable data is obtained through
> \[
> x_{0} \sim \mathcal{N}\left(x_{1} ; \sin \left(\theta_{1}\right)^{2}, 0.2^{2}\right) \quad x_{1} \sim \mathcal{N}\left(0.1 \cdot \theta_{1}^{2}, 0.2^{2}\right) \quad x_{2} \sim \mathcal{N}\left(x_{2} ; 0.1 \cdot \theta_{2}^{2}, 0.6^{2}\right) \quad x_{3} \sim \mathcal{N}\left(x_{3} ; \cos \left(\theta_{2}\right)^{2} ; 0.1^{2}\right)
> \]
> which leads to a tree-like factorization with highly multimodal conditionals.
> To obtain reference samples for all possible conditionals, we run the following procedure:
> - We initialized $N$ Markov chains with samples from the joint distribution.
> - We run 5000 steps of a HMC sampler.
> - Only the last samples of each chain were considered, yielding $N$ reference samples.

### task_007_check_006

Implement ten latent states theta0~N(0,0.5**2) and theta_(i+1)|theta_i~N(theta_i,0.5**2) for i=0,...,8; ten observations x_i|theta_i~N(theta_i**2,0.5**2). Use standard deviation 0.5 in Normal APIs; do not create an eleventh state from the printed indexing typo or replace squared emissions with linear observations.

> **paper.md**: HMM: This is a task in which the parameters have a Markovian factorization.
> \[
> \theta_{0} \sim \mathcal{N}\left(\theta_{0} ; 0 ., 0.5^{2}\right) \quad \theta_{i+1} \sim \mathcal{N}\left(\theta_{i+1} ; \theta_{i}, 0.5^{2}\right)
> \]
> for $i=0, \ldots, 9$. Observations are generated according to $x_{i}=\mathcal{N}\left(x_{i} ; \theta_{i}^{2}, 0.5^{2}\right)$, leading to a nonlinear hidden Markov model with bimodal correlated posterior and leading to a dimensionality of $\boldsymbol{\theta} \in \mathbb{R}^{10}, \boldsymbol{x} \in \mathbb{R}^{10}$.
> To obtain reference samples for all possible conditionals, we run the following procedure:
> - We initialized $N$ Markov chains with samples from the joint distribution.
> - We run 5000 steps of an HMC sampler.
> - Only the last samples of each chain were considered, yielding $N$ reference samples.

### task_007_check_007

Provide a differentiable joint log density consistent with each simulator, including prior support, covariance determinants and dependencies. For arbitrary conditioning, reconstruct the full joint state from fixed observed and free coordinates and evaluate that density; it is not sufficient to implement only p(theta|all x). For transformed free variables include the log-Jacobian. Do not treat a learned Simformer density or an ad hoc distance kernel as ground truth.

> **paper.md**: Next, we evaluated the ability of the Simformer to evaluate arbitrary conditionals. Arbitrary parameter and data conditions often vastly differ from the form of the posterior distribution, leading to a challenging inference task (Fig. 3). We performed inference on two of the benchmark tasks and established two new tasks with particularly interesting dependencies (Tree and HMM, details in Appendix Sec. A2.2). For each of the tasks, we generated ground truth posterior samples with Markov-Chain Monte-Carlo on 100 randomly selected conditional or full joint distributions. We found that, despite the complexity of these tasks, Simformer was able to accurately model all conditionals across all tasks

> **paper.md**: To obtain reference samples for all possible conditionals, we run the following procedure:
> - We initialized $N$ Markov chains with samples from the joint distribution.
> - We run 1000 steps of a random direction slice sampling algorithm.
> - We run an additional 3000 steps of MHMCMC with step size of 0.01.
> - Only the last samples of each chain were considered, yielding $N$ reference samples.

> **paper.md**: To obtain reference samples for all possible conditionals, we run the following procedure:
> - We initialized $N$ Markov chains with samples from the joint distribution.
> - We run 600 steps of a random direction slice sampling algorithm.
> - We run an additional 2000 steps of MHMCMC with a step size of 0.1.
> - Only the last samples of each chain was considered, yielding $N$ reference samples.

> **paper.md**: To obtain reference samples for all possible conditionals, we run the following procedure:
> - We initialized $N$ Markov chains with samples from the joint distribution.
> - We run 5000 steps of a HMC sampler.
> - Only the last samples of each chain were considered, yielding $N$ reference samples.

> **paper.md**: To obtain reference samples for all possible conditionals, we run the following procedure:
> - We initialized $N$ Markov chains with samples from the joint distribution.
> - We run 5000 steps of an HMC sampler.
> - Only the last samples of each chain were considered, yielding $N$ reference samples.

### task_007_check_008

For each requested conditional in Two Moons, SLCP, Tree and HMM initialize N independent chains from N draws of the full joint (sample parameters then simulate associated data), retain chain identity, then impose the fixed observed coordinates. If clamping creates an out-of-support state, repair/reinitialize transparently with a valid target state. A single chain, repeated common initial point, or prior-only initialization without the joint draw is not the prescribed protocol.

> **paper.md**: To obtain reference samples for all possible conditionals, we run the following procedure:
> - We initialized $N$ Markov chains with samples from the joint distribution.
> - We run 1000 steps of a random direction slice sampling algorithm.
> - We run an additional 3000 steps of MHMCMC with step size of 0.01.
> - Only the last samples of each chain were considered, yielding $N$ reference samples.

> **paper.md**: To obtain reference samples for all possible conditionals, we run the following procedure:
> - We initialized $N$ Markov chains with samples from the joint distribution.
> - We run 600 steps of a random direction slice sampling algorithm.
> - We run an additional 2000 steps of MHMCMC with a step size of 0.1.
> - Only the last samples of each chain was considered, yielding $N$ reference samples.

> **paper.md**: To obtain reference samples for all possible conditionals, we run the following procedure:
> - We initialized $N$ Markov chains with samples from the joint distribution.
> - We run 5000 steps of a HMC sampler.
> - Only the last samples of each chain were considered, yielding $N$ reference samples.

> **paper.md**: To obtain reference samples for all possible conditionals, we run the following procedure:
> - We initialized $N$ Markov chains with samples from the joint distribution.
> - We run 5000 steps of an HMC sampler.
> - Only the last samples of each chain were considered, yielding $N$ reference samples.

### task_007_check_009

Implement genuine random-direction slice transitions over the free coordinates, with a slice-height draw, a bracket search and shrinkage/acceptance inside the slice; reject invalid-support proposals. A function named slice that merely applies Gaussian random-walk MH is not equivalent. Keep all conditioned coordinates fixed.

> **paper.md**: To obtain reference samples for all possible conditionals, we run the following procedure:
> - We initialized $N$ Markov chains with samples from the joint distribution.
> - We run 1000 steps of a random direction slice sampling algorithm.
> - We run an additional 3000 steps of MHMCMC with step size of 0.01.
> - Only the last samples of each chain were considered, yielding $N$ reference samples.

> **paper.md**: To obtain reference samples for all possible conditionals, we run the following procedure:
> - We initialized $N$ Markov chains with samples from the joint distribution.
> - We run 600 steps of a random direction slice sampling algorithm.
> - We run an additional 2000 steps of MHMCMC with a step size of 0.1.
> - Only the last samples of each chain was considered, yielding $N$ reference samples.

### task_007_check_010

Run 1,000 random-direction slice transitions followed by 3,000 Metropolis-Hastings transitions per Two Moons chain, with MH proposal step size 0.01. The stages act on the same chain states and target conditional; counts and proposal scale must reach the executed code, not just a configuration description.

> **paper.md**: To obtain reference samples for all possible conditionals, we run the following procedure:
> - We initialized $N$ Markov chains with samples from the joint distribution.
> - We run 1000 steps of a random direction slice sampling algorithm.
> - We run an additional 3000 steps of MHMCMC with step size of 0.01.
> - Only the last samples of each chain were considered, yielding $N$ reference samples.

### task_007_check_011

Run 600 random-direction slice transitions followed by 2,000 Metropolis-Hastings transitions per SLCP chain, with MH proposal step size 0.1. Include the correct prior/likelihood in acceptance and reject out-of-support proposals rather than clipping them into the parameter box.

> **paper.md**: To obtain reference samples for all possible conditionals, we run the following procedure:
> - We initialized $N$ Markov chains with samples from the joint distribution.
> - We run 600 steps of a random direction slice sampling algorithm.
> - We run an additional 2000 steps of MHMCMC with a step size of 0.1.
> - Only the last samples of each chain was considered, yielding $N$ reference samples.

### task_007_check_012

For Tree and HMM implement 5,000 HMC transitions per chain: sample momentum, use log-target gradients in leapfrog integration, compute Hamiltonian energy change and perform Metropolis acceptance/rejection. Document mass/step-size/leapfrog settings. Random-walk MH, unadjusted Langevin and a function only named HMC are not acceptable substitutes.

> **paper.md**: To obtain reference samples for all possible conditionals, we run the following procedure:
> - We initialized $N$ Markov chains with samples from the joint distribution.
> - We run 5000 steps of a HMC sampler.
> - Only the last samples of each chain were considered, yielding $N$ reference samples.

> **paper.md**: To obtain reference samples for all possible conditionals, we run the following procedure:
> - We initialized $N$ Markov chains with samples from the joint distribution.
> - We run 5000 steps of an HMC sampler.
> - Only the last samples of each chain were considered, yielding $N$ reference samples.

### task_007_check_013

After the complete task-specific transition protocol, return exactly the last state of each of the N chains, yielding N reference samples of the requested free variables (or full states with fixed coordinates identified). Keep the chain axis separate from transition history. Concatenating one chain's retained trajectory, thinning intermediate states, or replicating chain endpoints to reach N is not equivalent.

> **paper.md**: To obtain reference samples for all possible conditionals, we run the following procedure:
> - We initialized $N$ Markov chains with samples from the joint distribution.
> - We run 1000 steps of a random direction slice sampling algorithm.
> - We run an additional 3000 steps of MHMCMC with step size of 0.01.
> - Only the last samples of each chain were considered, yielding $N$ reference samples.

> **paper.md**: To obtain reference samples for all possible conditionals, we run the following procedure:
> - We initialized $N$ Markov chains with samples from the joint distribution.
> - We run 600 steps of a random direction slice sampling algorithm.
> - We run an additional 2000 steps of MHMCMC with a step size of 0.1.
> - Only the last samples of each chain was considered, yielding $N$ reference samples.

> **paper.md**: To obtain reference samples for all possible conditionals, we run the following procedure:
> - We initialized $N$ Markov chains with samples from the joint distribution.
> - We run 5000 steps of a HMC sampler.
> - Only the last samples of each chain were considered, yielding $N$ reference samples.

> **paper.md**: To obtain reference samples for all possible conditionals, we run the following procedure:
> - We initialized $N$ Markov chains with samples from the joint distribution.
> - We run 5000 steps of an HMC sampler.
> - Only the last samples of each chain were considered, yielding $N$ reference samples.

## task_008 — Lotka-Volterra simulator with unstructured observations

Implement the Lotka-Volterra predator-prey ODE simulator. Parameters alpha, beta, gamma, delta are sigmoid-transformed Normal distributed in range one to three. Solve coupled ODEs with Gaussian observation noise. Support unstructured and irregularly sampled observations at user-specified time points. Generate synthetic observations with varying numbers of prey and predator measurements placed irregularly in time. For the application experiment, infer posterior distributions on a uniform grid. Implement the dynamic attention mask that depends on which time points are observed for prey versus predator populations, with Markovian structure within each population type and causal cross-dependency from past predator to current prey.

### task_008_check_001

Implement four rate parameters alpha,beta,gamma,delta from a sigmoid-transformed Normal prior scaled by 1+2*sigmoid(z), giving range (1,3). Document the Normal location/scale and initial populations because their numerical values are not fully specified in the supplied text. Use dx/dt=alpha*x-beta*x*y and dy/dt=delta*x*y-gamma*y; a uniform prior or analytic sine-wave surrogate is not equivalent.

> **paper.md**: Lotka Volterra The Lotka-Volterra equations, a foundational model in population dynamics, describe the interactions between predator and prey species (Volterra, 1926; Lotka, 1925). This model is parameterized as follows: the prior is chosen to be a sigmoid-transformed Normal distribution, scaled to a range from one to three. Data then evolves according to the following differential equations:
> \[
> \begin{aligned}
> & \frac{d x}{d t}=\alpha x-\beta x y \\
> & \frac{d y}{d t}=\delta x y-\gamma y
> \end{aligned}
> \]
> 
> Here, $x$ and $y$ represent the population sizes of the prey and predator species, respectively. The parameters $\alpha, \beta, \gamma$, and $\delta$ are positive real numbers that describe the two species' interaction rates and survival rates. To each simulation, we add Gaussian observation noise with $\sigma=0.1$.

### task_008_check_002

Numerically solve the coupled predator/prey dynamics and sample observations with additive independent Gaussian noise of standard deviation 0.1. Accept species-specific irregular time arrays and different observation counts; align each returned scalar with its true species and time. Inference inputs are full time-series values, not summary statistics or a common-grid interpolation substituted for observations.

> **paper.md**: Lotka Volterra The Lotka-Volterra equations, a foundational model in population dynamics, describe the interactions between predator and prey species (Volterra, 1926; Lotka, 1925). This model is parameterized as follows: the prior is chosen to be a sigmoid-transformed Normal distribution, scaled to a range from one to three. Data then evolves according to the following differential equations:
> \[
> \begin{aligned}
> & \frac{d x}{d t}=\alpha x-\beta x y \\
> & \frac{d y}{d t}=\delta x y-\gamma y
> \end{aligned}
> \]
> 
> Here, $x$ and $y$ represent the population sizes of the prey and predator species, respectively. The parameters $\alpha, \beta, \gamma$, and $\delta$ are positive real numbers that describe the two species' interaction rates and survival rates. To each simulation, we add Gaussian observation noise with $\sigma=0.1$.

> **paper.md**: Figure 5. Inference with unstructured observations in the LotkaVolterra model. (a) Posterior predictive (left) and posterior distribution (right) based on four unstructured observations of the prey population density (green crosses), using Simformer with $10^{5}$ simulations. True parameters in dark blue. (b) Same as (a) with nine additional observations of the predator population density. (c) C2ST-performance in estimating arbitrary conditionals (right) or the posterior distribution (left) using the C2ST metric.

### task_008_check_003

Build the Lotka-Volterra base mask from actual time metadata: independent rate-prior nodes; prey receives its alpha/beta parents and predator receives gamma/delta; consecutive times within each population form Markov edges; each prey node additionally receives edges from all earlier predator nodes. Use true sorted physical times and restore caller order. Never infer temporal precedence from raw token position or assume equal counts/times for both species.

> **addendum.md**: ## Task Dependencies
> For each task considered the attention mask representing a directed graphical model for the joint distributions is given. The undirected mask is obtained by making it undirected. Assuming the variables are in order $\theta_1, \theta_2,…, x_1, x_2,…$, these can be constructed as follows for the directed case:
> 
> ```python
> import numpy as np
> from scipy.linalg import block_diag
> # Gaussian Linear Task: Represents that data depends on parameters but is factorized across dimensions.
> M_θθ = np.eye(10)
> M_xx = np.eye(10)
> zeros = np.zeros((10, 10))
> M_θx = np.eye(10)
> M_E_gaussian = np.block([[M_θθ, zeros], [M_θx, M_xx]])
> # Two moons and Gaussian Mixture: Represents that each data variable depends on all parameters and the other data variables
> M_θθ = np.eye(2)
> M_xx = np.tril(np.ones((10, 10)))
> zeros = np.zeros((2, 10))
> M_θx = np.ones((10, 2))
> M_E_two_moons = np.block([[M_θθ, zeros], [M_θx, M_xx]])
> # SLCP: Dense parameter-data dependence
> M_θθ = np.eye(4)
> M_xx = block_diag(*[np.tril(np.ones((2, 2))) for _ in range(4)])
> zeros = np.zeros((4, 8))
> M_θx = np.ones((8, 4))
> M_E_slcp = np.block([[M_θθ, zeros], [M_θx, M_xx]])
> # Tree Structure: Diagonal is always true, follows tree dependencies
> M_E_tree = np.eye(10)
> M_E_tree[0, 1:3] = True  # θ₁, θ₂ depend on θ₀
> M_E_tree[1, 3:5] = True  # x₁, x₂ depend on θ₁
> M_E_tree[2, 5:7] = True  # x₃, x₄ depend on θ₂
> # HMM (Hidden Markov Model): Markov chain for parameters and factorized data
> M_θθ = np.eye(10) + np.diag(np.ones(9), k=-1)
> M_xx = np.eye(10)
> zeros = np.zeros((10, 10))
> M_θx = np.eye(10)
> M_E_hmm = np.block([[M_θθ, zeros], [M_θx, M_xx]])
> ```
> The mask for the Lotka Volterra task is metadata dependent, i.e., it depends on which T time points the predator or prey variables are observed. The prior hence M_θθ = I, the first two parameters are for the prey population and the last two for the predator M_θx = np.array([[1]*T + [0]*T, [1]*T + [0]*T, [0]*T + [1]*T, [0]*T + [1]*T]). The simulation is Markovian so M_x1x1 = M_x2x2 = np.eye(T) + np.diag(np.ones(T-1), k=-1). For the cross-data dependence a causal dependency is used, i.e., each prey variable depends additionally on all the past predator variables.

> **paper.md**: Lotka Volterra The Lotka-Volterra equations, a foundational model in population dynamics, describe the interactions between predator and prey species (Volterra, 1926; Lotka, 1925). This model is parameterized as follows: the prior is chosen to be a sigmoid-transformed Normal distribution, scaled to a range from one to three. Data then evolves according to the following differential equations:
> \[
> \begin{aligned}
> & \frac{d x}{d t}=\alpha x-\beta x y \\
> & \frac{d y}{d t}=\delta x y-\gamma y
> \end{aligned}
> \]
> 
> Here, $x$ and $y$ represent the population sizes of the prey and predator species, respectively. The parameters $\alpha, \beta, \gamma$, and $\delta$ are positive real numbers that describe the two species' interaction rates and survival rates. To each simulation, we add Gaussian observation noise with $\sigma=0.1$.

### task_008_check_004

Provide an independent likelihood/prior-based posterior or arbitrary-conditional reference route for Lotka-Volterra, using the same ODE/noise model and exact observed time/species tuples. References must not be Simformer samples relabeled as truth. Expose reproducible initial conditions and solver tolerances; numerical failures must not silently emit zero trajectories.

> **paper.md**: We then used the ability of Simformer to sample from arbitrary conditional distribution to simultaneously generate posterior and posterior predictive samples without additional runs of the simulator. The posterior predictives of Simformer capture data and uncertainty in a realistic manner (Fig. 5a).

> **paper.md**: Lotka Volterra The Lotka-Volterra equations, a foundational model in population dynamics, describe the interactions between predator and prey species (Volterra, 1926; Lotka, 1925). This model is parameterized as follows: the prior is chosen to be a sigmoid-transformed Normal distribution, scaled to a range from one to three. Data then evolves according to the following differential equations:
> \[
> \begin{aligned}
> & \frac{d x}{d t}=\alpha x-\beta x y \\
> & \frac{d y}{d t}=\delta x y-\gamma y
> \end{aligned}
> \]
> 
> Here, $x$ and $y$ represent the population sizes of the prey and predator species, respectively. The parameters $\alpha, \beta, \gamma$, and $\delta$ are positive real numbers that describe the two species' interaction rates and survival rates. To each simulation, we add Gaussian observation noise with $\sigma=0.1$.

## task_009 — SIRD epidemiological model with time-dependent contact rate

Implement the SIRD Susceptible-Infected-Recovered-Deceased epidemiological simulator with time-varying contact rate. Global parameters gamma recovery and mu mortality follow Uniform priors. Time-dependent contact rate beta(t) is sampled from a Gaussian process with RBF kernel transformed via sigmoid to ensure beta(t) in zero to one. Solve the coupled ODEs for S, I, R, D compartments. Add log-normal observation noise. Support functional infinite-dimensional parameter inference for the time-dependent contact rate using subsampling at random time points. Demonstrate parameter-conditioned posterior inference with partial parameter measurements.

### task_009_check_001

Implement dS/dt=-beta(t)*S*I, dI/dt=beta(t)*S*I-gamma*I-mu*I, dR/dt=gamma*I, dD/dt=mu*I, with all four compartments and documented population normalization/initial conditions. gamma and mu are constant within a trajectory; only beta varies with time. The paper later calls mortality delta: map that alias to the same mu, not to an extra parameter.

> **paper.md**: SIRD Model with Time-Dependent Contact Rate The SIRD (Susceptible, Infected, Recovered, Deceased) model extends the classical SIR framework by incorporating a Deceased $(D)$ compartment. Similar models were explored by Chen et al. (2020); Schmidt et al. (2021). This addition is crucial for modeling diseases with significant mortality rates. The dynamics of the SIRD model, considering a time-dependent contact rate, are governed by the following set of differential equations:
> \[
> \begin{aligned}
> \frac{d S}{d t} & =-\beta(t) S I \\
> \frac{d I}{d t} & =\beta(t) S I-\gamma I-\mu I \\
> \frac{d R}{d t} & =\gamma I \\
> \frac{d D}{d t} & =\mu I
> \end{aligned}
> \]
> 
> Here, $S, I, R$, and $D$ denote the number of susceptible, infected, recovered, and deceased individuals, respectively. The term $\beta(t)$ represents the time-varying contact rate, while $\gamma$ and $\mu$ signify the recovery and mortality rates among the infected population, respectively.
> 
> Incorporating a time-dependent contact rate $\beta(t)$ is pivotal for capturing the effects of public health interventions and societal behavioral changes over time. This feature is essential for accurately simulating the real-world dynamics of a disease's spread, particularly in the context of varying public health policies and community responses.
> 
> We impose a Uniform prior on the global variables, $\gamma$ and $\delta$, denoted as $\gamma, \delta \sim \operatorname{Unif}(0,0.5)$. For the time-dependent contact rate, we first sample $\hat{\beta} \sim \mathcal{G}(0, k)$ from a Gaussian process prior, with $k$ representing an RBF kernel defined as $k\left(t_{1}, t_{2}\right)=2.5^{2} \exp \left(-\frac{1}{2} \frac{\left\|t_{1}-t_{2}\right\|^{2}}{7^{2}}\right)$. This is further transformed via a sigmoid function to ensure $\beta(t) \in[0,1]$ for all $t$. Observational data is modeled with log-normal noise, characterized by a mean of $S(t)$ and a standard deviation of $\sigma=0.05$.

### task_009_check_002

Sample independent global recovery and mortality rates uniformly from (0,0.5), and use the same parameters in simulation, joint data construction and inference. Preserve the gamma/mu (or explicitly mapped gamma/delta) names and shapes; no Normal or fixed-rate replacement.

> **paper.md**: SIRD Model with Time-Dependent Contact Rate The SIRD (Susceptible, Infected, Recovered, Deceased) model extends the classical SIR framework by incorporating a Deceased $(D)$ compartment. Similar models were explored by Chen et al. (2020); Schmidt et al. (2021). This addition is crucial for modeling diseases with significant mortality rates. The dynamics of the SIRD model, considering a time-dependent contact rate, are governed by the following set of differential equations:
> \[
> \begin{aligned}
> \frac{d S}{d t} & =-\beta(t) S I \\
> \frac{d I}{d t} & =\beta(t) S I-\gamma I-\mu I \\
> \frac{d R}{d t} & =\gamma I \\
> \frac{d D}{d t} & =\mu I
> \end{aligned}
> \]
> 
> Here, $S, I, R$, and $D$ denote the number of susceptible, infected, recovered, and deceased individuals, respectively. The term $\beta(t)$ represents the time-varying contact rate, while $\gamma$ and $\mu$ signify the recovery and mortality rates among the infected population, respectively.
> 
> Incorporating a time-dependent contact rate $\beta(t)$ is pivotal for capturing the effects of public health interventions and societal behavioral changes over time. This feature is essential for accurately simulating the real-world dynamics of a disease's spread, particularly in the context of varying public health policies and community responses.
> 
> We impose a Uniform prior on the global variables, $\gamma$ and $\delta$, denoted as $\gamma, \delta \sim \operatorname{Unif}(0,0.5)$. For the time-dependent contact rate, we first sample $\hat{\beta} \sim \mathcal{G}(0, k)$ from a Gaussian process prior, with $k$ representing an RBF kernel defined as $k\left(t_{1}, t_{2}\right)=2.5^{2} \exp \left(-\frac{1}{2} \frac{\left\|t_{1}-t_{2}\right\|^{2}}{7^{2}}\right)$. This is further transformed via a sigmoid function to ensure $\beta(t) \in[0,1]$ for all $t$. Observational data is modeled with log-normal noise, characterized by a mean of $S(t)$ and a standard deviation of $\sigma=0.05$.

### task_009_check_003

Sample a zero-mean correlated Gaussian process on actual time points with RBF covariance K_ij=2.5**2*exp(-0.5*(t_i-t_j)**2/7**2), then apply sigmoid to obtain beta(t) in [0,1]. A single constant beta, independent per-time Gaussian draws or a hand-chosen sinusoid is not this prior. Use coherent GP paths when solving/querying between sampled points and disclose the numerical representation.

> **paper.md**: SIRD Model with Time-Dependent Contact Rate The SIRD (Susceptible, Infected, Recovered, Deceased) model extends the classical SIR framework by incorporating a Deceased $(D)$ compartment. Similar models were explored by Chen et al. (2020); Schmidt et al. (2021). This addition is crucial for modeling diseases with significant mortality rates. The dynamics of the SIRD model, considering a time-dependent contact rate, are governed by the following set of differential equations:
> \[
> \begin{aligned}
> \frac{d S}{d t} & =-\beta(t) S I \\
> \frac{d I}{d t} & =\beta(t) S I-\gamma I-\mu I \\
> \frac{d R}{d t} & =\gamma I \\
> \frac{d D}{d t} & =\mu I
> \end{aligned}
> \]
> 
> Here, $S, I, R$, and $D$ denote the number of susceptible, infected, recovered, and deceased individuals, respectively. The term $\beta(t)$ represents the time-varying contact rate, while $\gamma$ and $\mu$ signify the recovery and mortality rates among the infected population, respectively.
> 
> Incorporating a time-dependent contact rate $\beta(t)$ is pivotal for capturing the effects of public health interventions and societal behavioral changes over time. This feature is essential for accurately simulating the real-world dynamics of a disease's spread, particularly in the context of varying public health policies and community responses.
> 
> We impose a Uniform prior on the global variables, $\gamma$ and $\delta$, denoted as $\gamma, \delta \sim \operatorname{Unif}(0,0.5)$. For the time-dependent contact rate, we first sample $\hat{\beta} \sim \mathcal{G}(0, k)$ from a Gaussian process prior, with $k$ representing an RBF kernel defined as $k\left(t_{1}, t_{2}\right)=2.5^{2} \exp \left(-\frac{1}{2} \frac{\left\|t_{1}-t_{2}\right\|^{2}}{7^{2}}\right)$. This is further transformed via a sigmoid function to ensure $\beta(t) \in[0,1]$ for all $t$. Observational data is modeled with log-normal noise, characterized by a mean of $S(t)$ and a standard deviation of $\sigma=0.05$.

### task_009_check_004

Implement log-normal observation noise around the selected S/I/R/D compartment trajectory with the stated noise parameter 0.05. Document whether 0.05 is a data-space standard deviation or log-space scale and map physical means to log-normal parameters consistently; for an arithmetic mean m/std 0.05, use log variance log(1+(0.05/m)**2) and log location log(m)-log_variance/2. Handle zero populations explicitly. Never pass a raw positive state as the log-location (which exponentiates the state), or replace the model by additive Gaussian noise.

> **paper.md**: SIRD Model with Time-Dependent Contact Rate The SIRD (Susceptible, Infected, Recovered, Deceased) model extends the classical SIR framework by incorporating a Deceased $(D)$ compartment. Similar models were explored by Chen et al. (2020); Schmidt et al. (2021). This addition is crucial for modeling diseases with significant mortality rates. The dynamics of the SIRD model, considering a time-dependent contact rate, are governed by the following set of differential equations:
> \[
> \begin{aligned}
> \frac{d S}{d t} & =-\beta(t) S I \\
> \frac{d I}{d t} & =\beta(t) S I-\gamma I-\mu I \\
> \frac{d R}{d t} & =\gamma I \\
> \frac{d D}{d t} & =\mu I
> \end{aligned}
> \]
> 
> Here, $S, I, R$, and $D$ denote the number of susceptible, infected, recovered, and deceased individuals, respectively. The term $\beta(t)$ represents the time-varying contact rate, while $\gamma$ and $\mu$ signify the recovery and mortality rates among the infected population, respectively.
> 
> Incorporating a time-dependent contact rate $\beta(t)$ is pivotal for capturing the effects of public health interventions and societal behavioral changes over time. This feature is essential for accurately simulating the real-world dynamics of a disease's spread, particularly in the context of varying public health policies and community responses.
> 
> We impose a Uniform prior on the global variables, $\gamma$ and $\delta$, denoted as $\gamma, \delta \sim \operatorname{Unif}(0,0.5)$. For the time-dependent contact rate, we first sample $\hat{\beta} \sim \mathcal{G}(0, k)$ from a Gaussian process prior, with $k$ representing an RBF kernel defined as $k\left(t_{1}, t_{2}\right)=2.5^{2} \exp \left(-\frac{1}{2} \frac{\left\|t_{1}-t_{2}\right\|^{2}}{7^{2}}\right)$. This is further transformed via a sigmoid function to ensure $\beta(t) \in[0,1]$ for all $t$. Observational data is modeled with log-normal noise, characterized by a mean of $S(t)$ and a standard deviation of $\sigma=0.05$.

> **paper.md**: We generated synthetic measurements from infected, recovered, and deceased individuals at irregularly spaced time points and applied the Simformer to estimate the posterior distribution of parameters. The Simformer estimated realistic death and recovery rates and successfully recovers a time-dependent contact rate that aligns with ground truth observations (Fig. 6a). Indeed, as measurements of infections tend towards zero (around timestamp 25, Fig. 6a, orange), the Simformer-posterior for the contact rate increases its uncertainty. This is expected, as we cannot obtain conclusive insights about the contact rate in scenarios with negligible infections.

### task_009_check_005

Construct training examples by subsampling the same latent GP path and simulated trajectories at varying physical times, with beta tokens sharing a beta identifier and time metadata. At inference accept requested times beyond the training subsample, condition on observed beta values where supplied, and sample remaining beta values jointly with global rates/data. A fixed output vector followed only by interpolation does not implement the stated functional inference interface.

> **paper.md**: The Simformer uses learnable vector embeddings for identifiers and condition states, as proposed in Weilbach et al. (2023). In cases where parameters or data are functions
> ![](assets/asset_2.jpg)
> 
> Figure 2. Simformer architecture. All variables (parameters and data) are reduced to a token representation which includes the variables' identity, the variables' value (val) as well as the conditional state (latent ( L ) or conditioned (C)). This sequence of tokens is processed by a transformer model; the interaction of variables can be explicitly controlled through an attention mask. The transformer architecture returns a score that is used to generate samples from the score-based diffusion model and can be modified (e.g. to guide the diffusion process).
> of space or time, the node identifier will comprise a shared embedding vector and a random Fourier embedding of the elements in the index set.

> **paper.md**: Figure 6. Inference of $\infty$-dim parameter space in the SIRD model. (a) Inferred posterior for global parameters (upper left) and timedependent local parameters (upper right) based on five observations (crosses) of infected (I), recovered (R), and death (D) population densities. The black dot and dashed line indicate the true parameter, bold lines indicate the mean, and shaded areas represent $99 \%$ quantiles. (b) Inference with parameter measurements and a single measurement of fatalities.

> **paper.md**: SIRD Model with Time-Dependent Contact Rate The SIRD (Susceptible, Infected, Recovered, Deceased) model extends the classical SIR framework by incorporating a Deceased $(D)$ compartment. Similar models were explored by Chen et al. (2020); Schmidt et al. (2021). This addition is crucial for modeling diseases with significant mortality rates. The dynamics of the SIRD model, considering a time-dependent contact rate, are governed by the following set of differential equations:
> \[
> \begin{aligned}
> \frac{d S}{d t} & =-\beta(t) S I \\
> \frac{d I}{d t} & =\beta(t) S I-\gamma I-\mu I \\
> \frac{d R}{d t} & =\gamma I \\
> \frac{d D}{d t} & =\mu I
> \end{aligned}
> \]
> 
> Here, $S, I, R$, and $D$ denote the number of susceptible, infected, recovered, and deceased individuals, respectively. The term $\beta(t)$ represents the time-varying contact rate, while $\gamma$ and $\mu$ signify the recovery and mortality rates among the infected population, respectively.
> 
> Incorporating a time-dependent contact rate $\beta(t)$ is pivotal for capturing the effects of public health interventions and societal behavioral changes over time. This feature is essential for accurately simulating the real-world dynamics of a disease's spread, particularly in the context of varying public health policies and community responses.
> 
> We impose a Uniform prior on the global variables, $\gamma$ and $\delta$, denoted as $\gamma, \delta \sim \operatorname{Unif}(0,0.5)$. For the time-dependent contact rate, we first sample $\hat{\beta} \sim \mathcal{G}(0, k)$ from a Gaussian process prior, with $k$ representing an RBF kernel defined as $k\left(t_{1}, t_{2}\right)=2.5^{2} \exp \left(-\frac{1}{2} \frac{\left\|t_{1}-t_{2}\right\|^{2}}{7^{2}}\right)$. This is further transformed via a sigmoid function to ensure $\beta(t) \in[0,1]$ for all $t$. Observational data is modeled with log-normal noise, characterized by a mean of $S(t)$ and a standard deviation of $\sigma=0.05$.

## task_010 — Hodgkin-Huxley neural simulator with summary statistics and energy

Implement the Hodgkin-Huxley neuron model following Pospischil et al. Initial membrane voltage negative sixty-five mV, simulate over two hundred ms with input current four mA applied between fifty ms and one hundred fifty ms. Implement the rate functions for m, h, n gates. Solve the full HH differential equations for V, m, h, n gates plus energy accumulation. Compute sodium charge-based energy consumption using the exact formula from the addendum with all conversion factors including mS to S, mV to V, membrane area, elementary charge, valence, three Na per ATP, ATP energy, divided by point-two for J/s, multiplied by one million for uJ/s. Extract seven summary statistics: spike count, mean resting potential, std resting potential, mean spiking potential, second central moment variance, third central moment, fourth central moment of voltage during the spiking domain. Add energy consumption as an additional statistic.

### task_010_check_001

Implement the seven-parameter Hodgkin-Huxley simulator described in the paper, with an explicit parameter/prior schema and units including the plotted g_Na,g_K,C_m,g_L. Document choices not numerically specified by the paper rather than claiming invented prior ranges are paper values. Start V at -65 mV, integrate 200 ms, and apply a 4 mA current only in the 50-to-150 ms stimulus window. Keep physical-unit conversions consistent with the membrane-area convention.

> **paper.md**: Hodgkin-Huxley Model: In our study, we adhere to the implementation guidelines set forth by Pospischil et al. (2008) for the Hodgkin-Huxley model. The initial membrane voltage is established at $V_{0}=-65.0 \mathrm{mV}$. Simulations are conducted over a duration of 200 ms , during which an input current of 4 mA is applied in the interval between 50 ms and 150 ms
> 
> The rate functions are defined by the following equations:
> \[
> \begin{aligned}
> \alpha_{m}(V) & =0.32 \times \frac{\operatorname{efun}\left(-0.25\left(V-V_{0}-13.0\right)\right)}{0.25} \\
> \beta_{m}(V) & =0.28 \times \frac{\operatorname{efun}\left(0.2\left(V-V_{0}-40.0\right)\right)}{0.2} \\
> \alpha_{h}(V) & =0.128 \times \exp \left(-\frac{\left(V-V_{0}-17.0\right)}{18.0}\right) \\
> \beta_{h}(V) & =\frac{4.0}{1.0+\exp \left(-\frac{\left(V-V_{0}-40.0\right)}{5.0}\right)} \\
> \alpha_{n}(V) & =0.032 \times \frac{\operatorname{efun}\left(-0.2\left(V-V_{0}-15.0\right)\right)}{0.2} \\
> \beta_{n}(V) & =0.5 \times \exp \left(-\frac{\left(V-V_{0}-10.0\right)}{40.0}\right)
> \end{aligned}
> \]
> where efun $(x)=\left\{\begin{array}{ll}1-\frac{x}{2} & \text { if } x<1 e-4 \\ \frac{x}{\exp (x)-1.0} & \text { otherwise }\end{array}\right.$.
> 
> This formulation leads to the comprehensive Hodgkin-Huxley differential equations:
> \[
> \begin{aligned}
> \frac{d V}{d t} & =\frac{I_{\mathrm{inj}}(t)-g_{\mathrm{Na}} m^{3} h\left(V-E_{\mathrm{Na}}\right)-g_{\mathrm{K}} n^{4}\left(V-E_{\mathrm{K}}\right)-g_{\mathrm{L}}\left(V-E_{\mathrm{L}}\right)}{C_{m}}+0.05 d w_{t}, \\
> \frac{d m}{d t} & =\alpha_{m}(V)(1-m)-\beta_{m}(V) m, \\
> \frac{d h}{d t} & =\alpha_{h}(V)(1-h)-\beta_{h}(V) h, \\
> \frac{d n}{d t} & =\alpha_{n}(V)(1-n)-\beta_{n}(V) n, \\
> \frac{d H}{d t} & =g_{\mathrm{Na}} m^{3} h\left(V-E_{\mathrm{Na}}\right) .
> \end{aligned}
> \]
> 
> Notably, there exist multiple methodologies for estimating energy consumption in neuronal models, as discussed by Deistler et al. (2022b). In our approach, we opt to calculate energy consumption based on sodium charge, which can be converted into $\mu J / s$ as detailed by Deistler et al. (2022b). For observational data, we employ summary features consistent with those used by Gonçalves et al. (2020).

> **paper.md**: Figure 7. Inference in the Hodgkin-Huxley model. (a) Model schematic, observed voltage trace, and associated energy consumption. (b) Marginals of inferred posterior for four parameters. (c) Posterior predictive energy consumption from Simformer (blue) and from simulator outputs (green). (d) Posterior predictive samples from the posterior in (c) using the simulator. (e) Marginals of inferred energy constrained posterior for four parameters. (f) Posterior predictive energy consumption from Simformer (blue) and from simulator outputs (green). Energy constraint as red line. (g) Posterior predictive samples from posterior in (e) using the simulator.

### task_010_check_002

Implement all six Appendix A2.2 gating rates with its coefficients and shifts: alpha_m=0.32*efun(-0.25*(V-V0-13))/0.25; beta_m=0.28*efun(0.2*(V-V0-40))/0.2; alpha_h=0.128*exp(-(V-V0-17)/18); beta_h=4/(1+exp(-(V-V0-40)/5)); alpha_n=0.032*efun(-0.2*(V-V0-15))/0.2; beta_n=0.5*exp(-(V-V0-10)/40). Use efun(z)=z/expm1(z), with the series 1-z/2 near ZERO (absolute smallness), not for every negative z. This resolves the printed one-sided test without changing the rate law.

> **paper.md**: Hodgkin-Huxley Model: In our study, we adhere to the implementation guidelines set forth by Pospischil et al. (2008) for the Hodgkin-Huxley model. The initial membrane voltage is established at $V_{0}=-65.0 \mathrm{mV}$. Simulations are conducted over a duration of 200 ms , during which an input current of 4 mA is applied in the interval between 50 ms and 150 ms
> 
> The rate functions are defined by the following equations:
> \[
> \begin{aligned}
> \alpha_{m}(V) & =0.32 \times \frac{\operatorname{efun}\left(-0.25\left(V-V_{0}-13.0\right)\right)}{0.25} \\
> \beta_{m}(V) & =0.28 \times \frac{\operatorname{efun}\left(0.2\left(V-V_{0}-40.0\right)\right)}{0.2} \\
> \alpha_{h}(V) & =0.128 \times \exp \left(-\frac{\left(V-V_{0}-17.0\right)}{18.0}\right) \\
> \beta_{h}(V) & =\frac{4.0}{1.0+\exp \left(-\frac{\left(V-V_{0}-40.0\right)}{5.0}\right)} \\
> \alpha_{n}(V) & =0.032 \times \frac{\operatorname{efun}\left(-0.2\left(V-V_{0}-15.0\right)\right)}{0.2} \\
> \beta_{n}(V) & =0.5 \times \exp \left(-\frac{\left(V-V_{0}-10.0\right)}{40.0}\right)
> \end{aligned}
> \]
> where efun $(x)=\left\{\begin{array}{ll}1-\frac{x}{2} & \text { if } x<1 e-4 \\ \frac{x}{\exp (x)-1.0} & \text { otherwise }\end{array}\right.$.
> 
> This formulation leads to the comprehensive Hodgkin-Huxley differential equations:
> \[
> \begin{aligned}
> \frac{d V}{d t} & =\frac{I_{\mathrm{inj}}(t)-g_{\mathrm{Na}} m^{3} h\left(V-E_{\mathrm{Na}}\right)-g_{\mathrm{K}} n^{4}\left(V-E_{\mathrm{K}}\right)-g_{\mathrm{L}}\left(V-E_{\mathrm{L}}\right)}{C_{m}}+0.05 d w_{t}, \\
> \frac{d m}{d t} & =\alpha_{m}(V)(1-m)-\beta_{m}(V) m, \\
> \frac{d h}{d t} & =\alpha_{h}(V)(1-h)-\beta_{h}(V) h, \\
> \frac{d n}{d t} & =\alpha_{n}(V)(1-n)-\beta_{n}(V) n, \\
> \frac{d H}{d t} & =g_{\mathrm{Na}} m^{3} h\left(V-E_{\mathrm{Na}}\right) .
> \end{aligned}
> \]
> 
> Notably, there exist multiple methodologies for estimating energy consumption in neuronal models, as discussed by Deistler et al. (2022b). In our approach, we opt to calculate energy consumption based on sodium charge, which can be converted into $\mu J / s$ as detailed by Deistler et al. (2022b). For observational data, we employ summary features consistent with those used by Gonçalves et al. (2020).

### task_010_check_003

Integrate coupled V,m,h,n and cumulative sodium charge H: dV=[Iinj-gNa*m**3*h*(V-ENa)-gK*n**4*(V-EK)-gL*(V-EL)]/Cm plus the stated 0.05 Wiener noise; dgating=alpha(V)*(1-gating)-beta(V)*gating; dH=gNa*m**3*h*(V-ENa). Scale stochastic increments with sqrt(dt), use the evolving gates/voltage and a numerically adequate step/solver. A spike-count model, synthetic voltage trace or energy independent of the sodium current is not equivalent.

> **paper.md**: Hodgkin-Huxley Model: In our study, we adhere to the implementation guidelines set forth by Pospischil et al. (2008) for the Hodgkin-Huxley model. The initial membrane voltage is established at $V_{0}=-65.0 \mathrm{mV}$. Simulations are conducted over a duration of 200 ms , during which an input current of 4 mA is applied in the interval between 50 ms and 150 ms
> 
> The rate functions are defined by the following equations:
> \[
> \begin{aligned}
> \alpha_{m}(V) & =0.32 \times \frac{\operatorname{efun}\left(-0.25\left(V-V_{0}-13.0\right)\right)}{0.25} \\
> \beta_{m}(V) & =0.28 \times \frac{\operatorname{efun}\left(0.2\left(V-V_{0}-40.0\right)\right)}{0.2} \\
> \alpha_{h}(V) & =0.128 \times \exp \left(-\frac{\left(V-V_{0}-17.0\right)}{18.0}\right) \\
> \beta_{h}(V) & =\frac{4.0}{1.0+\exp \left(-\frac{\left(V-V_{0}-40.0\right)}{5.0}\right)} \\
> \alpha_{n}(V) & =0.032 \times \frac{\operatorname{efun}\left(-0.2\left(V-V_{0}-15.0\right)\right)}{0.2} \\
> \beta_{n}(V) & =0.5 \times \exp \left(-\frac{\left(V-V_{0}-10.0\right)}{40.0}\right)
> \end{aligned}
> \]
> where efun $(x)=\left\{\begin{array}{ll}1-\frac{x}{2} & \text { if } x<1 e-4 \\ \frac{x}{\exp (x)-1.0} & \text { otherwise }\end{array}\right.$.
> 
> This formulation leads to the comprehensive Hodgkin-Huxley differential equations:
> \[
> \begin{aligned}
> \frac{d V}{d t} & =\frac{I_{\mathrm{inj}}(t)-g_{\mathrm{Na}} m^{3} h\left(V-E_{\mathrm{Na}}\right)-g_{\mathrm{K}} n^{4}\left(V-E_{\mathrm{K}}\right)-g_{\mathrm{L}}\left(V-E_{\mathrm{L}}\right)}{C_{m}}+0.05 d w_{t}, \\
> \frac{d m}{d t} & =\alpha_{m}(V)(1-m)-\beta_{m}(V) m, \\
> \frac{d h}{d t} & =\alpha_{h}(V)(1-h)-\beta_{h}(V) h, \\
> \frac{d n}{d t} & =\alpha_{n}(V)(1-n)-\beta_{n}(V) n, \\
> \frac{d H}{d t} & =g_{\mathrm{Na}} m^{3} h\left(V-E_{\mathrm{Na}}\right) .
> \end{aligned}
> \]
> 
> Notably, there exist multiple methodologies for estimating energy consumption in neuronal models, as discussed by Deistler et al. (2022b). In our approach, we opt to calculate energy consumption based on sodium charge, which can be converted into $\mu J / s$ as detailed by Deistler et al. (2022b). For observational data, we employ summary features consistent with those used by Gonçalves et al. (2020).

### task_010_check_004

Extract exactly seven voltage features from each actual noisy simulated trace: spike count; pre-stimulus/resting mean and standard deviation; stimulus/spiking-window mean; and second, third and fourth CENTRAL moments in that window. Define windows and spike detection explicitly in physical time and count events rather than above-threshold samples. Follow the addendum's raw central moments; optional standardized moments must be separately named, not substituted.

> **addendum.md**: The summary statistics computed for the Hodgkin-Huxley task are:
> 1. Spike count
> 2. Mean of the resting potential
> 3. Standard deviation of the resting potential
> 4. Mean potential in the spiking domain
> 5. Second central moment (variance) of the voltage during the spiking domain
> 6. Third central moment of the voltage during the spiking domain
> 7. Fourth central moment of the voltage during the spiking domain

> **paper.md**: Hodgkin-Huxley Model: In our study, we adhere to the implementation guidelines set forth by Pospischil et al. (2008) for the Hodgkin-Huxley model. The initial membrane voltage is established at $V_{0}=-65.0 \mathrm{mV}$. Simulations are conducted over a duration of 200 ms , during which an input current of 4 mA is applied in the interval between 50 ms and 150 ms
> 
> The rate functions are defined by the following equations:
> \[
> \begin{aligned}
> \alpha_{m}(V) & =0.32 \times \frac{\operatorname{efun}\left(-0.25\left(V-V_{0}-13.0\right)\right)}{0.25} \\
> \beta_{m}(V) & =0.28 \times \frac{\operatorname{efun}\left(0.2\left(V-V_{0}-40.0\right)\right)}{0.2} \\
> \alpha_{h}(V) & =0.128 \times \exp \left(-\frac{\left(V-V_{0}-17.0\right)}{18.0}\right) \\
> \beta_{h}(V) & =\frac{4.0}{1.0+\exp \left(-\frac{\left(V-V_{0}-40.0\right)}{5.0}\right)} \\
> \alpha_{n}(V) & =0.032 \times \frac{\operatorname{efun}\left(-0.2\left(V-V_{0}-15.0\right)\right)}{0.2} \\
> \beta_{n}(V) & =0.5 \times \exp \left(-\frac{\left(V-V_{0}-10.0\right)}{40.0}\right)
> \end{aligned}
> \]
> where efun $(x)=\left\{\begin{array}{ll}1-\frac{x}{2} & \text { if } x<1 e-4 \\ \frac{x}{\exp (x)-1.0} & \text { otherwise }\end{array}\right.$.
> 
> This formulation leads to the comprehensive Hodgkin-Huxley differential equations:
> \[
> \begin{aligned}
> \frac{d V}{d t} & =\frac{I_{\mathrm{inj}}(t)-g_{\mathrm{Na}} m^{3} h\left(V-E_{\mathrm{Na}}\right)-g_{\mathrm{K}} n^{4}\left(V-E_{\mathrm{K}}\right)-g_{\mathrm{L}}\left(V-E_{\mathrm{L}}\right)}{C_{m}}+0.05 d w_{t}, \\
> \frac{d m}{d t} & =\alpha_{m}(V)(1-m)-\beta_{m}(V) m, \\
> \frac{d h}{d t} & =\alpha_{h}(V)(1-h)-\beta_{h}(V) h, \\
> \frac{d n}{d t} & =\alpha_{n}(V)(1-n)-\beta_{n}(V) n, \\
> \frac{d H}{d t} & =g_{\mathrm{Na}} m^{3} h\left(V-E_{\mathrm{Na}}\right) .
> \end{aligned}
> \]
> 
> Notably, there exist multiple methodologies for estimating energy consumption in neuronal models, as discussed by Deistler et al. (2022b). In our approach, we opt to calculate energy consumption based on sodium charge, which can be converted into $\mu J / s$ as detailed by Deistler et al. (2022b). For observational data, we employ summary features consistent with those used by Gonçalves et al. (2020).

### task_010_check_005

For the cumulative sodium-charge trajectory H, implement the addendum trace conversion: first difference to obtain per-step noncumulative charge, then convolve with the length-5 uniform kernel in same mode, then call the charge-to-power conversion. These are charge increments, so do not multiply by dt a second time or treat the cumulative trajectory as per-step energy.

> **addendum.md**: In the Hodgkin-Huxley task, the energy consumption is computed based on sodium charge using the following formula:
> ```python
> def convert_charge_to_energy(E):
>     E = E # Energy is negative
>     E = jnp.diff(E) # Non cumulative energy
>     E = jnp.convolve(E, 1/5*jnp.ones(5), mode="same") # Smooth due to numerical differentiation error
>     return convert_total_energy(E)
> 
> def convert_total_energy(E):
>     E = -E # Energy is negative
>     E = E/1000 # mS to S
>     E = E / 1000 # mV to V
>     E = E * 0.628e-3 # Area of the membrane
>     e = 1.602176634e-19 # Elementary charge
>     N_Na = E / e # Number of elementary charges
>     valence_Na = 1 # Valence of sodium
>     number_of_transports = 3 # Number of Na out per ATP 
>     ATP_Na = N_Na / (valence_Na * number_of_transports) # Number of ATP
>     ATP_energy = 10e-19 # Energy by ATP hydrolysis
>     E = ATP_Na * ATP_energy # Energy in Joules
>     E = E / 0.2 # Energy in J/s
>     return E * 1e+6 # Energy in uJ/s
> ```

> **paper.md**: Hodgkin-Huxley Model: In our study, we adhere to the implementation guidelines set forth by Pospischil et al. (2008) for the Hodgkin-Huxley model. The initial membrane voltage is established at $V_{0}=-65.0 \mathrm{mV}$. Simulations are conducted over a duration of 200 ms , during which an input current of 4 mA is applied in the interval between 50 ms and 150 ms
> 
> The rate functions are defined by the following equations:
> \[
> \begin{aligned}
> \alpha_{m}(V) & =0.32 \times \frac{\operatorname{efun}\left(-0.25\left(V-V_{0}-13.0\right)\right)}{0.25} \\
> \beta_{m}(V) & =0.28 \times \frac{\operatorname{efun}\left(0.2\left(V-V_{0}-40.0\right)\right)}{0.2} \\
> \alpha_{h}(V) & =0.128 \times \exp \left(-\frac{\left(V-V_{0}-17.0\right)}{18.0}\right) \\
> \beta_{h}(V) & =\frac{4.0}{1.0+\exp \left(-\frac{\left(V-V_{0}-40.0\right)}{5.0}\right)} \\
> \alpha_{n}(V) & =0.032 \times \frac{\operatorname{efun}\left(-0.2\left(V-V_{0}-15.0\right)\right)}{0.2} \\
> \beta_{n}(V) & =0.5 \times \exp \left(-\frac{\left(V-V_{0}-10.0\right)}{40.0}\right)
> \end{aligned}
> \]
> where efun $(x)=\left\{\begin{array}{ll}1-\frac{x}{2} & \text { if } x<1 e-4 \\ \frac{x}{\exp (x)-1.0} & \text { otherwise }\end{array}\right.$.
> 
> This formulation leads to the comprehensive Hodgkin-Huxley differential equations:
> \[
> \begin{aligned}
> \frac{d V}{d t} & =\frac{I_{\mathrm{inj}}(t)-g_{\mathrm{Na}} m^{3} h\left(V-E_{\mathrm{Na}}\right)-g_{\mathrm{K}} n^{4}\left(V-E_{\mathrm{K}}\right)-g_{\mathrm{L}}\left(V-E_{\mathrm{L}}\right)}{C_{m}}+0.05 d w_{t}, \\
> \frac{d m}{d t} & =\alpha_{m}(V)(1-m)-\beta_{m}(V) m, \\
> \frac{d h}{d t} & =\alpha_{h}(V)(1-h)-\beta_{h}(V) h, \\
> \frac{d n}{d t} & =\alpha_{n}(V)(1-n)-\beta_{n}(V) n, \\
> \frac{d H}{d t} & =g_{\mathrm{Na}} m^{3} h\left(V-E_{\mathrm{Na}}\right) .
> \end{aligned}
> \]
> 
> Notably, there exist multiple methodologies for estimating energy consumption in neuronal models, as discussed by Deistler et al. (2022b). In our approach, we opt to calculate energy consumption based on sodium charge, which can be converted into $\mu J / s$ as detailed by Deistler et al. (2022b). For observational data, we employ summary features consistent with those used by Gonçalves et al. (2020).

### task_010_check_006

Implement charge-to-power with the addendum factors in order: negate inward sodium charge; divide by 1000 for mS-to-S and by 1000 for mV-to-V; multiply membrane area 0.628e-3; divide by elementary charge 1.602176634e-19, Na valence 1 and 3 Na transported per ATP; multiply ATP energy 10e-19 J; divide by 0.2 s; multiply by 1e6 to obtain microJ/s. Equivalent algebra is acceptable, but omitted area, missing sign/ATP conversion or arbitrary rescaling is not.

> **addendum.md**: In the Hodgkin-Huxley task, the energy consumption is computed based on sodium charge using the following formula:
> ```python
> def convert_charge_to_energy(E):
>     E = E # Energy is negative
>     E = jnp.diff(E) # Non cumulative energy
>     E = jnp.convolve(E, 1/5*jnp.ones(5), mode="same") # Smooth due to numerical differentiation error
>     return convert_total_energy(E)
> 
> def convert_total_energy(E):
>     E = -E # Energy is negative
>     E = E/1000 # mS to S
>     E = E / 1000 # mV to V
>     E = E * 0.628e-3 # Area of the membrane
>     e = 1.602176634e-19 # Elementary charge
>     N_Na = E / e # Number of elementary charges
>     valence_Na = 1 # Valence of sodium
>     number_of_transports = 3 # Number of Na out per ATP 
>     ATP_Na = N_Na / (valence_Na * number_of_transports) # Number of ATP
>     ATP_energy = 10e-19 # Energy by ATP hydrolysis
>     E = ATP_Na * ATP_energy # Energy in Joules
>     E = E / 0.2 # Energy in J/s
>     return E * 1e+6 # Energy in uJ/s
> ```

### task_010_check_007

Produce a scalar whole-stimulus-duration energy/power statistic consistently from integrated charge H[-1]-H[0] via convert_total_energy; retain convert_charge_to_energy for the time-resolved trace. Do not average the already duration-normalized incremental trace as a substitute for total power. Append the energy scalar as a distinct eighth data statistic after the seven voltage statistics; keep its index/units available to conditional masks and guidance.

> **addendum.md**: In the Hodgkin-Huxley task, the energy consumption is computed based on sodium charge using the following formula:
> ```python
> def convert_charge_to_energy(E):
>     E = E # Energy is negative
>     E = jnp.diff(E) # Non cumulative energy
>     E = jnp.convolve(E, 1/5*jnp.ones(5), mode="same") # Smooth due to numerical differentiation error
>     return convert_total_energy(E)
> 
> def convert_total_energy(E):
>     E = -E # Energy is negative
>     E = E/1000 # mS to S
>     E = E / 1000 # mV to V
>     E = E * 0.628e-3 # Area of the membrane
>     e = 1.602176634e-19 # Elementary charge
>     N_Na = E / e # Number of elementary charges
>     valence_Na = 1 # Valence of sodium
>     number_of_transports = 3 # Number of Na out per ATP 
>     ATP_Na = N_Na / (valence_Na * number_of_transports) # Number of ATP
>     ATP_energy = 10e-19 # Energy by ATP hydrolysis
>     E = ATP_Na * ATP_energy # Energy in Joules
>     E = E / 0.2 # Energy in J/s
>     return E * 1e+6 # Energy in uJ/s
> ```

> **addendum.md**: The summary statistics computed for the Hodgkin-Huxley task are:
> 1. Spike count
> 2. Mean of the resting potential
> 3. Standard deviation of the resting potential
> 4. Mean potential in the spiking domain
> 5. Second central moment (variance) of the voltage during the spiking domain
> 6. Third central moment of the voltage during the spiking domain
> 7. Fourth central moment of the voltage during the spiking domain

> **paper.md**: We first inferred the posterior distribution given only the summary statistics of the voltage (not the energy) with the Simformer, and we found that, consistent with prior work (Gonçalves et al., 2020), the posterior distribution has wide marginals for some parameters and narrow marginals for others (Fig. 7b). We then used Simformer's ability to sample arbitrary conditionals and generate posterior predictives for energy consumption (Fig. 7c). The posterior predictive distribution of Simformer closely matched the posterior predictive distribution obtained by running the simulator (Fig. 7cd), and the energy cost of different posterior samples varied significantly (Deistler et al., 2022b).

## task_011 — Baseline implementations NPE NRE NLE via SBI library

Implement baseline methods Neural Posterior Estimation NPE, Neural Ratio Estimation NRE, and Neural Likelihood Estimation NLE using the sbi Python library as required by the addendum. Use default parameters from sbi but configure NPE and NLE to use neural spline flow instead of the default density estimator. Train each baseline with batch size one thousand and Adam optimizer, stopping on convergence via early stopping based on validation loss. These baselines serve as comparison targets for the Simformer on all benchmark tasks.

### task_011_check_001

Instantiate NPE, NRE and NLE through the sbi Python library. NPE/NLE must select neural spline flows (NSF) rather than the default estimator; NRE must use the library ratio estimator. Feed actual matched theta/x simulations from the same benchmark definitions. Handwritten MLP regression, Gaussian surrogates or a custom non-sbi implementation cannot replace these baselines.

> **paper.md**: We set $\sigma_{\max }=15, \sigma_{\min }=0.0001, \beta_{\min }=0.01$, and $\beta_{\max }=10$ for all experiments. Both for the time interval $[1 e-5,1$.$] .$
> For implementing Neural Posterior Estimation (NPE), Neural Ratio Estimation (NRE), and Neural Likelihood Estimation (NLE), we utilize the sbi library (Tejero-Cantero et al., 2020), adopting default parameters but opting for a more expressive neural spline flow for NPE and NLE. Each method was trained using the provided training loop with a batch size of 1000 and an Adam optimizer. Training ceased upon convergence, as indicated by early stopping based on validation loss.

### task_011_check_002

Train through each selected sbi method's provided training API, with training_batch_size=1000 or the equivalent supported setting. Wire Adam into the optimizer path; when Adam is the pinned library's internal default, verify and document that implementation rather than inventing an unsupported optimizer keyword. Save the trained estimator and prior/transform information required for its actual posterior construction.

> **paper.md**: We set $\sigma_{\max }=15, \sigma_{\min }=0.0001, \beta_{\min }=0.01$, and $\beta_{\max }=10$ for all experiments. Both for the time interval $[1 e-5,1$.$] .$
> For implementing Neural Posterior Estimation (NPE), Neural Ratio Estimation (NRE), and Neural Likelihood Estimation (NLE), we utilize the sbi library (Tejero-Cantero et al., 2020), adopting default parameters but opting for a more expressive neural spline flow for NPE and NLE. Each method was trained using the provided training loop with a batch size of 1000 and an Adam optimizer. Training ceased upon convergence, as indicated by early stopping based on validation loss.

### task_011_check_003

Use held-out validation loss and the sbi-supported validation split/early-stopping controls to stop training on convergence. Verify argument names against the pinned version's signatures; validation_fraction or other supported mechanisms are valid, and a nonexistent validate_data parameter must not be required. Do not substitute training-loss convergence or a tiny fixed epoch cap for the scientific early-stopping protocol.

> **paper.md**: We set $\sigma_{\max }=15, \sigma_{\min }=0.0001, \beta_{\min }=0.01$, and $\beta_{\max }=10$ for all experiments. Both for the time interval $[1 e-5,1$.$] .$
> For implementing Neural Posterior Estimation (NPE), Neural Ratio Estimation (NRE), and Neural Likelihood Estimation (NLE), we utilize the sbi library (Tejero-Cantero et al., 2020), adopting default parameters but opting for a more expressive neural spline flow for NPE and NLE. Each method was trained using the provided training loop with a batch size of 1000 and an Adam optimizer. Training ceased upon convergence, as indicated by early stopping based on validation loss.

### task_011_check_004

Expose a common posterior-sampling interface conditioned on actual observations. NPE uses the learned posterior; NLE/NRE combine their learned likelihood/ratio with the prior through the appropriate library posterior/MCMC machinery. Preserve sample dimensions and physical parameter units so the benchmark C2ST compares the same target distributions. Do not sample estimator weights, regress to a point or compare likelihood samples to posterior references.

> **paper.md**: We set $\sigma_{\max }=15, \sigma_{\min }=0.0001, \beta_{\min }=0.01$, and $\beta_{\max }=10$ for all experiments. Both for the time interval $[1 e-5,1$.$] .$
> For implementing Neural Posterior Estimation (NPE), Neural Ratio Estimation (NRE), and Neural Likelihood Estimation (NLE), we utilize the sbi library (Tejero-Cantero et al., 2020), adopting default parameters but opting for a more expressive neural spline flow for NPE and NLE. Each method was trained using the provided training loop with a batch size of 1000 and an Adam optimizer. Training ceased upon convergence, as indicated by early stopping based on validation loss.

> **paper.md**: We evaluated performance in approximating posterior distributions across four benchmark tasks (Lueckmann et al., 2021). For each task, samples for ten ground-truth posteriors are available (Appendix Sec. A2.2), and we assessed performance as classifier two-sample test (C2ST) accuracy to these samples. Here, a score of 0.5 signifies perfect alignment with the ground truth posterior, and 1.0 indicates that a classifier can completely distinguish between the approximation and the ground truth. All results are obtained using the Variance Exploding SDE (VESDE); additional results using the Variance Preserving SDE (VPSDE) can be found in Appendix Sec. A3. See Appendix Sec. A2 for details on the parameterization.

## task_012 — Training loop and evaluation pipeline

Implement the complete training loop for the Simformer. For each training batch: sample parameters and generate simulated data via the appropriate simulator; construct the joint sequence of parameters and data; randomly sample a condition mask M_C from the five specified options; apply graph inversion if needed to compute the adapted attention mask M_E; tokenize the data; add noise at a random diffusion time; compute the denoising score matching loss masked by one minus M_C; backpropagate and update. Training batch size one thousand, Adam optimizer. Implement the evaluation pipeline: generate test simulations; run the Simformer to sample from target conditionals using the reverse SDE with five hundred Euler-Maruyama steps default; compute C2ST accuracy using a random forest classifier with one hundred trees to compare Simformer samples against ground-truth reference posteriors.

### task_012_check_001

For each batch use matched prior/simulator joint samples and independently resample the five-family condition mask per example. Compute the graph adaptation for each directed-mask example and pass its condition mask, metadata, variable IDs and attention mask through the same tokenizer/model call. Do not train with one mask distribution and silently infer with another fixed dependency mask.

> **paper.md**: Having defined the tokenizer which processes every $(\boldsymbol{\theta}, \boldsymbol{x})$ pair and the attention mask to specify dependencies within the simulator, the Simformer can be trained using denoising score-matching (Hyvärinen \& Dayan, 2005; Song et al., 2021b): We sample the noise level $t$ for the diffusion model uniformly at random and generate a (partially) noisy sample $\hat{\mathbf{x}}_{t}^{M_{C}}=\left(1-M_{C}\right) \cdot \hat{\mathbf{x}}_{t}+M_{C} \cdot \hat{\mathbf{x}}_{0}$ i.e. variables that we want to condition on remain clean. The loss can then be defined as
> \[
> \begin{aligned}
> & \ell\left(\phi, M_{C}, t, \hat{\mathbf{x}}_{0}, \hat{\mathbf{x}}_{t}\right)= \\
> & \quad\left(1-M_{C}\right) \cdot\left(s_{\phi}^{M_{E}}\left(\hat{\mathbf{x}}_{t}^{M_{C}}, t\right)-\nabla_{\hat{\mathbf{x}}_{t}} \log p_{t}\left(\hat{\mathbf{x}}_{t} \mid \hat{\mathbf{x}}_{0}\right)\right),
> \end{aligned}
> \]

> **addendum.md**: During training, for each element in a batch, the condition mask $M_C$ is sampled uniformly at random from the following options:
> - joint mask (all False)
> - posterior mask (all "parameter" variables are False, all "data" variables are True)
> - likelihood mask (all "data" variables are False, all "parameter" variables are True)
> - rand_mask1 ~ Ber(0.3) (resampled for every element)
> - rand_mask2 ~ Ber(0.7) (resampled for every element)

> **addendum.md**: Algorithm Graph Inversion by Webb et al. 2018:
> 1. Input: Joint Bayesian net structure $G$ as mask $M_E$, latent variables $Z$ as given by $M_C$.
> 2. $J \leftarrow \text{MORALIZE}(G)$ # Make undirected + connect parents.
> 3. Set all vertices of $J$ to be unmarked.
> 4. $H \leftarrow \{ \text{VARIABLES}(G), \emptyset \}$, i.e. unconnected graph.
> 5. $S \leftarrow$ all latent variables without latent parent in $G$.
> 6. while $S \neq \emptyset$ do
> 7.    Select $v \in S$ according to min-fill criterion # Node that adds fewest edges below.
> 8.   Add edges in $J$ between unmarked neighbours of $v$.
> 9.   Make unmarked neighbours of $v$ in $J$, $v$'s parents in $H$.
> 10.   Mark $v$ and remove from $S$.
> 11.   for unmarked child latents $u$ of $v$ in $G$ do
> 12.    Add $u$ to $S$ if all its parent latents in $G$ are marked.
> 13.   end for
> 14. end while
> 15. return $H$.
> 
> To produce the final attention mask, the edges in $H$ are added to the base attention mask $M_E$.

### task_012_check_002

Generate xt from the chosen SDE, replace observed positions with clean x0, and compute the scalar score error only at unobserved positions. Sum the squared coordinate errors per example, apply the explicit positive lambda(t), then reduce the batch. Conditioned and padded entries must contribute zero; an all-observed example must not cause division by zero. Assert or otherwise enforce matching [batch,variables] output/target shapes so unintended broadcasting cannot train a different objective.

> **paper.md**: Having defined the tokenizer which processes every $(\boldsymbol{\theta}, \boldsymbol{x})$ pair and the attention mask to specify dependencies within the simulator, the Simformer can be trained using denoising score-matching (Hyvärinen \& Dayan, 2005; Song et al., 2021b): We sample the noise level $t$ for the diffusion model uniformly at random and generate a (partially) noisy sample $\hat{\mathbf{x}}_{t}^{M_{C}}=\left(1-M_{C}\right) \cdot \hat{\mathbf{x}}_{t}+M_{C} \cdot \hat{\mathbf{x}}_{0}$ i.e. variables that we want to condition on remain clean. The loss can then be defined as
> \[
> \begin{aligned}
> & \ell\left(\phi, M_{C}, t, \hat{\mathbf{x}}_{0}, \hat{\mathbf{x}}_{t}\right)= \\
> & \quad\left(1-M_{C}\right) \cdot\left(s_{\phi}^{M_{E}}\left(\hat{\mathbf{x}}_{t}^{M_{C}}, t\right)-\nabla_{\hat{\mathbf{x}}_{t}} \log p_{t}\left(\hat{\mathbf{x}}_{t} \mid \hat{\mathbf{x}}_{0}\right)\right),
> \end{aligned}
> \]

> **paper.md**: where $\lambda$ denotes a positive weighting function.

### task_012_check_003

Use Adam and batch size 1000 for all Simformer scientific configurations. Select VE in the main experiments; honor the declared simulation budget as the number of simulated training pairs instead of silently simulating fresh unlimited pairs each epoch. A last smaller batch is acceptable, and an explicit memory-equivalent accumulation scheme must preserve the intended update. Save actual trained weights and data/model configuration for inference.

> **paper.md**: The employed transformer model features a token dimension of 50 and represents diffusion time through a 128 -dimensional random Gaussian Fourier embedding. It comprises 6 layers and 4 heads with an attention size of 10, and a widening factor of 3, implying that the feed-forward block expands to a hidden dimension of 150. For the Lotka-Volterra, SIR, and Hodgkin-Huxley tasks, we increased the number of layers to 8. Similar to the above, we used a training batch size of 1000 and an Adam optimizer.

> **paper.md**: We evaluated performance in approximating posterior distributions across four benchmark tasks (Lueckmann et al., 2021). For each task, samples for ten ground-truth posteriors are available (Appendix Sec. A2.2), and we assessed performance as classifier two-sample test (C2ST) accuracy to these samples. Here, a score of 0.5 signifies perfect alignment with the ground truth posterior, and 1.0 indicates that a classifier can completely distinguish between the approximation and the ground truth. All results are obtained using the Variance Exploding SDE (VESDE); additional results using the Variance Preserving SDE (VPSDE) can be found in Appendix Sec. A3. See Appendix Sec. A2 for details on the parameterization.

> **paper.md**: We set $\sigma_{\max }=15, \sigma_{\min }=0.0001, \beta_{\min }=0.01$, and $\beta_{\max }=10$ for all experiments. Both for the time interval $[1 e-5,1$.$] .$
> For implementing Neural Posterior Estimation (NPE), Neural Ratio Estimation (NRE), and Neural Likelihood Estimation (NLE), we utilize the sbi library (Tejero-Cantero et al., 2020), adopting default parameters but opting for a more expressive neural spline flow for NPE and NLE. Each method was trained using the provided training loop with a batch size of 1000 and an Adam optimizer. Training ceased upon convergence, as indicated by early stopping based on validation loss.

### task_012_check_004

Load the correct task/budget/attention-variant checkpoint, reconstruct its tokenizer and score model, and execute the 500-step reverse sampler on unobserved variables while clamping observed values throughout. Guidance, when selected, must call task_006 with the same scaling and variable ordering. Preserve transforms/units for reported physical samples; initializing a new random model at evaluation is not acceptable.

> **paper.md**: We draw samples from the noise distribution and run the reverse diffusion process on all unobserved variables, while keeping observed variables constant at their conditioning value (Weilbach et al., 2023).

> **paper.md**: For inference, we solved the reverse SDE using an Euler-Maruyama discretization. We use 500 steps by default; accuracy for different budgets is shown in Fig. A7.

> **paper.md**: In that line, we use the following general formulation to guide the diffusion process:
> \[
> s_{\phi}\left(\hat{\boldsymbol{x}}_{t}, t \mid c\right) \approx s_{\phi}\left(\hat{\boldsymbol{x}}_{t}, t\right)+\nabla_{\hat{\boldsymbol{x}}_{t}} \log \sigma\left(-s(t) c\left(\hat{\boldsymbol{x}}_{t}\right)\right)
> \]
> 
> Here $\sigma$ denotes the sigmoid function, $s(t)$ is an appropriate scaling function satisfying $s(t) \rightarrow \infty$ as $t \rightarrow 0$, depending on the choice of SDE, and $c$ denotes a constraint function $c(\hat{\boldsymbol{x}}) \leq 0$. For example, to enforce an interval upper bound $u$, we use $c(\hat{\boldsymbol{x}})=\hat{\boldsymbol{x}}-u$.

### task_012_check_005

Implement C2ST with a RandomForestClassifier of 100 trees, labels identifying two independent sample sets, and mean held-out accuracy over FIVE cross-validation folds. Use balanced sample counts or an explicitly balanced evaluation, shuffle/stratify reproducibly, and fit any learned preprocessing within each training fold. Do not report training accuracy, a single holdout split, a neural classifier or a fixed 0.5.

> **addendum.md**: C2ST is implemented using a random forest classifier with 100 trees.

> **paper.md**: We evaluated performance in approximating posterior distributions across four benchmark tasks (Lueckmann et al., 2021). For each task, samples for ten ground-truth posteriors are available (Appendix Sec. A2.2), and we assessed performance as classifier two-sample test (C2ST) accuracy to these samples. Here, a score of 0.5 signifies perfect alignment with the ground truth posterior, and 1.0 indicates that a classifier can completely distinguish between the approximation and the ground truth. All results are obtained using the Variance Exploding SDE (VESDE); additional results using the Variance Preserving SDE (VPSDE) can be found in Appendix Sec. A3. See Appendix Sec. A2 for details on the parameterization.

### task_012_check_006

Compare model and independent reference samples for the identical task, observation, free-coordinate set and units. Report raw classifier accuracy: near 0.5 means indistinguishability and near 1 means separation. Do not invert the score, force it into a desired range, include labels/chain IDs as features or reuse model samples as references. Save sample counts, condition masks and observation identifiers with metrics.

> **addendum.md**: C2ST is implemented using a random forest classifier with 100 trees.

> **paper.md**: We evaluated performance in approximating posterior distributions across four benchmark tasks (Lueckmann et al., 2021). For each task, samples for ten ground-truth posteriors are available (Appendix Sec. A2.2), and we assessed performance as classifier two-sample test (C2ST) accuracy to these samples. Here, a score of 0.5 signifies perfect alignment with the ground truth posterior, and 1.0 indicates that a classifier can completely distinguish between the approximation and the ground truth. All results are obtained using the Variance Exploding SDE (VESDE); additional results using the Variance Preserving SDE (VPSDE) can be found in Appendix Sec. A3. See Appendix Sec. A2 for details on the parameterization.

## task_013 — Benchmark experiment orchestration and result collection

Orchestrate and execute the main benchmark experiments described in Section four point one. For each of the four standard benchmark tasks Gaussian Linear, Two Moons, Gaussian Mixture, SLCP, train the Simformer with varying simulation budgets one thousand, ten thousand, one hundred thousand. Evaluate C2ST performance against ground-truth posteriors for ten reference posteriors per task. Compare Simformer variants dense attention, undirected structured attention, directed structured attention against NPE baseline. Also evaluate on the Tree and HMM tasks for arbitrary conditional distributions. Collect and organize all quantitative results. Code-Dev scope: implement runnable experiment orchestration and result collection, without requiring experiments to execute now. Figure 4(b) includes Two Moons, SLCP, Tree and HMM and 100 random conditional/joint queries per task. Figures 4(a,b) use 1,000/10,000/100,000 simulations. Execution success and reported numeric results are not approval conditions.

### task_013_check_001

Implement the Figure 4(a) experiment matrix: Gaussian Linear, Gaussian Mixture, Two Moons and SLCP; simulation budgets 10**3,10**4,10**5; six-layer VE Simformer with dense, undirected and directed masks plus NPE comparison. Route variant choice into actual masks/models and keep budget/task/model outputs distinct. NRE/NLE APIs remain implemented, but reproducing the appendix-only extended comparison is not an approval requirement.

> **paper.md**: We evaluated performance in approximating posterior distributions across four benchmark tasks (Lueckmann et al., 2021). For each task, samples for ten ground-truth posteriors are available (Appendix Sec. A2.2), and we assessed performance as classifier two-sample test (C2ST) accuracy to these samples. Here, a score of 0.5 signifies perfect alignment with the ground truth posterior, and 1.0 indicates that a classifier can completely distinguish between the approximation and the ground truth. All results are obtained using the Variance Exploding SDE (VESDE); additional results using the Variance Preserving SDE (VPSDE) can be found in Appendix Sec. A3. See Appendix Sec. A2 for details on the parameterization.

> **paper.md**: Across all four benchmark tasks, the Simformer outperformed neural posterior estimation (NPE), even when the Simformer used a dense attention mask (Fig. 4a). The only exception was the Gaussian linear task with 10k simulations; we show an extended comparison with NRE and NLE in Appendix Fig. A5, results with VPSDE in Appendix Fig. A6). Incorporating domain knowledge into the attention mask of the transformer led to further improvements in the accuracy of the Simformer, particularly in tasks with sparser dependency structures, such as the Linear Gaussian (fully factorized) and SLCP (4 i.i.d. observations).

> **paper.md**: The employed transformer model features a token dimension of 50 and represents diffusion time through a 128 -dimensional random Gaussian Fourier embedding. It comprises 6 layers and 4 heads with an attention size of 10, and a widening factor of 3, implying that the feed-forward block expands to a hidden dimension of 150. For the Lotka-Volterra, SIR, and Hodgkin-Huxley tasks, we increased the number of layers to 8. Similar to the above, we used a training batch size of 1000 and an Adam optimizer.

> **addendum.md**: In Section 4.1, the experiments related to the calibration and loglikelihood of Simformer are not required for replication.
> 
> In Section 4.3, the experiments related to the calibration of Simformer are not required for replication.
> 
> In Section 4.4, the experiments related to "Additional details and results on guidance are in Appendix Sec. A3.3" are not required for replication.

### task_013_check_002

For each of the four standard posterior tasks provide ten fixed reference observations and their matching reference posterior samples, reusing them across model variants/budgets. Use official benchmark reference data where available or the specified validated target sampler; infer N model samples for N reference samples and compute/aggregate C2ST per observation. Preserve all ten observation identities, not ten repeats of one observation.

> **paper.md**: We evaluated performance in approximating posterior distributions across four benchmark tasks (Lueckmann et al., 2021). For each task, samples for ten ground-truth posteriors are available (Appendix Sec. A2.2), and we assessed performance as classifier two-sample test (C2ST) accuracy to these samples. Here, a score of 0.5 signifies perfect alignment with the ground truth posterior, and 1.0 indicates that a classifier can completely distinguish between the approximation and the ground truth. All results are obtained using the Variance Exploding SDE (VESDE); additional results using the Variance Preserving SDE (VPSDE) can be found in Appendix Sec. A3. See Appendix Sec. A2 for details on the parameterization.

> **paper.md**: To obtain reference samples for all possible conditionals, we run the following procedure:
> - We initialized $N$ Markov chains with samples from the joint distribution.
> - We run 1000 steps of a random direction slice sampling algorithm.
> - We run an additional 3000 steps of MHMCMC with step size of 0.01.
> - Only the last samples of each chain were considered, yielding $N$ reference samples.

> **paper.md**: To obtain reference samples for all possible conditionals, we run the following procedure:
> - We initialized $N$ Markov chains with samples from the joint distribution.
> - We run 600 steps of a random direction slice sampling algorithm.
> - We run an additional 2000 steps of MHMCMC with a step size of 0.1.
> - Only the last samples of each chain was considered, yielding $N$ reference samples.

### task_013_check_003

Implement Figure 4(b) for Two Moons, SLCP, Tree and HMM, using 100 reproducibly selected mixed conditional or full-joint queries per task. Generate reference samples with that task's N-chain protocol and model samples for the identical free variables and conditions. Evaluate all three attention variants over the same 10**3/10**4/10**5 budget grid; do not replace arbitrary conditionals by posterior-only queries.

> **paper.md**: Next, we evaluated the ability of the Simformer to evaluate arbitrary conditionals. Arbitrary parameter and data conditions often vastly differ from the form of the posterior distribution, leading to a challenging inference task (Fig. 3). We performed inference on two of the benchmark tasks and established two new tasks with particularly interesting dependencies (Tree and HMM, details in Appendix Sec. A2.2). For each of the tasks, we generated ground truth posterior samples with Markov-Chain Monte-Carlo on 100 randomly selected conditional or full joint distributions. We found that, despite the complexity of these tasks, Simformer was able to accurately model all conditionals across all tasks

### task_013_check_004

Record actual configurations, checkpoint identities, query definitions, sample counts and per-query/aggregated C2ST values in machine-readable outputs consumed by Figure 4 plotting. Missing computations must be reported as missing, not fabricated curves or constants. Provide runnable code and full scientific defaults; observed superiority over NPE, target C2ST thresholds, calibration and log-likelihood experiments are not Code-Dev approval requirements.

> **paper.md**: Across all four benchmark tasks, the Simformer outperformed neural posterior estimation (NPE), even when the Simformer used a dense attention mask (Fig. 4a). The only exception was the Gaussian linear task with 10k simulations; we show an extended comparison with NRE and NLE in Appendix Fig. A5, results with VPSDE in Appendix Fig. A6). Incorporating domain knowledge into the attention mask of the transformer led to further improvements in the accuracy of the Simformer, particularly in tasks with sparser dependency structures, such as the Linear Gaussian (fully factorized) and SLCP (4 i.i.d. observations).

> **addendum.md**: In Section 4.1, the experiments related to the calibration and loglikelihood of Simformer are not required for replication.
> 
> In Section 4.3, the experiments related to the calibration of Simformer are not required for replication.
> 
> In Section 4.4, the experiments related to "Additional details and results on guidance are in Appendix Sec. A3.3" are not required for replication.

> **instructions.txt**: Your submission will be graded based on whether the code necessary for reproducing the paper has been correctly implemented. The code will not be executed during grading.

## task_014 — Application experiment Lotka-Volterra unstructured observations

Execute the application experiment described in Section four point two. Train the Simformer on the Lotka-Volterra simulator with one hundred thousand simulations. Generate synthetic unstructured observations: first scenario with four irregularly-spaced prey population measurements; second scenario adding nine additional irregularly-spaced predator measurements. Infer posterior distributions for both scenarios. Generate posterior predictive samples using the Simformer ability to sample arbitrary conditionals. Compute C2ST performance for both the posterior distribution and arbitrary conditionals. Visualize posterior predictive distributions and posterior marginals compared to ground truth. Code-Dev scope: implement the full experiment without running it now. The 100,000-simulation model supplies the two illustrative cases; the quantitative Figure 5(c) sweep additionally uses 1,000/10,000/100,000 simulations and dense/undirected/directed masks. Predictions include a uniform 0-to-15 time grid. Dense attention is used for the illustrative cases.

### task_014_check_001

Configure eight-layer VE Simformers; implement a 10**3/10**4/10**5 budget sweep with dense/undirected/directed attention for Figure 5(c), and use the 10**5 dense model for Figure 5(a,b) illustrative observations. Pass actual metadata-derived structured masks for structured variants; store separate artifacts for each configuration.

> **paper.md**: The employed transformer model features a token dimension of 50 and represents diffusion time through a 128 -dimensional random Gaussian Fourier embedding. It comprises 6 layers and 4 heads with an attention size of 10, and a widening factor of 3, implying that the feed-forward block expands to a hidden dimension of 150. For the Lotka-Volterra, SIR, and Hodgkin-Huxley tasks, we increased the number of layers to 8. Similar to the above, we used a training batch size of 1000 and an Adam optimizer.

> **paper.md**: Figure 5. Inference with unstructured observations in the LotkaVolterra model. (a) Posterior predictive (left) and posterior distribution (right) based on four unstructured observations of the prey population density (green crosses), using Simformer with $10^{5}$ simulations. True parameters in dark blue. (b) Same as (a) with nine additional observations of the predator population density. (c) C2ST-performance in estimating arbitrary conditionals (right) or the posterior distribution (left) using the C2ST metric.

### task_014_check_002

Generate one synthetic underlying Lotka-Volterra system and first condition on four irregularly spaced prey observations. The second case retains those same four observations and adds nine irregular predator observations from that same system. Reuse the trained model, expose variable counts/times through the tokenizer, and retain true parameters for comparison; do not resimulate an unrelated truth for the second panel.

> **paper.md**: Figure 5. Inference with unstructured observations in the LotkaVolterra model. (a) Posterior predictive (left) and posterior distribution (right) based on four unstructured observations of the prey population density (green crosses), using Simformer with $10^{5}$ simulations. True parameters in dark blue. (b) Same as (a) with nine additional observations of the predator population density. (c) C2ST-performance in estimating arbitrary conditionals (right) or the posterior distribution (left) using the C2ST metric.

### task_014_check_003

Infer the joint conditional of unknown global parameters and unobserved prey/predator values on a UNIFORM grid from t=0 to t=15, retaining off-grid observed tokens with their exact times. Sample the posterior and predictive trajectories directly from Simformer without extra simulator runs in this predictive route. Interpolating the four observations or only re-running the ODE at estimated parameters does not reproduce that capability.

> **paper.md**: We then used the ability of Simformer to sample from arbitrary conditional distribution to simultaneously generate posterior and posterior predictive samples without additional runs of the simulator. The posterior predictives of Simformer capture data and uncertainty in a realistic manner (Fig. 5a).

> **addendum.md**: In Section 4.2, infer the posterior distributions on a uniform grid between t=0 and t=15.

### task_014_check_004

Compute posterior and arbitrary-conditional C2ST with independent MCMC/target references using the same observation/time tuple and free coordinates. Save posterior marginals, predictive means/uncertainty and ground-truth/observation overlays for both illustrative cases and the quantitative budget sweep. Code must compute these quantities; narrower uncertainty or a published numerical score is not required during static approval.

> **paper.md**: Figure 5. Inference with unstructured observations in the LotkaVolterra model. (a) Posterior predictive (left) and posterior distribution (right) based on four unstructured observations of the prey population density (green crosses), using Simformer with $10^{5}$ simulations. True parameters in dark blue. (b) Same as (a) with nine additional observations of the predator population density. (c) C2ST-performance in estimating arbitrary conditionals (right) or the posterior distribution (left) using the C2ST metric.

> **paper.md**: We then used the ability of Simformer to sample from arbitrary conditional distribution to simultaneously generate posterior and posterior predictive samples without additional runs of the simulator. The posterior predictives of Simformer capture data and uncertainty in a realistic manner (Fig. 5a).

> **addendum.md**: C2ST is implemented using a random forest classifier with 100 trees.

> **instructions.txt**: Your submission will be graded based on whether the code necessary for reproducing the paper has been correctly implemented. The code will not be executed during grading.

## task_015 — Application experiment SIRD infinite-dimensional parameter inference

Execute the application experiment described in Section four point three. Train the Simformer on the SIRD simulator with time-dependent contact rate beta(t) modeled as a Gaussian process. Use subsampling at random time points to handle the infinite-dimensional parameter space. Generate synthetic observations of infected, recovered, and deceased populations at irregular time points. Infer the posterior distribution for global parameters gamma, mu and the time-dependent contact rate. Demonstrate parameter-conditioned posterior inference with partial parameter measurements. Generate posterior predictive samples. Code-Dev scope: implement these workflows without running training or inference now. Use dense attention and eight layers. Implement both Figure 6 conditions; use four beta(t) observations plus one deceased-population observation for panel (b), following the figure caption rather than the conflicting infected-population sentence. Grid queries span 0 to 40; expose irregular query times too.

### task_015_check_001

Use an eight-layer VE Simformer with a fully DENSE attention mask in both Section 4.3 scenarios. Confirm the executed application constructor and sampler actually receive all-real-token attention, rather than inheriting a structured mask default from another task.

> **paper.md**: The employed transformer model features a token dimension of 50 and represents diffusion time through a 128 -dimensional random Gaussian Fourier embedding. It comprises 6 layers and 4 heads with an attention size of 10, and a widening factor of 3, implying that the feed-forward block expands to a hidden dimension of 150. For the Lotka-Volterra, SIR, and Hodgkin-Huxley tasks, we increased the number of layers to 8. Similar to the above, we used a training batch size of 1000 and an Adam optimizer.

> **paper.md**: Figure 6. Inference of $\infty$-dim parameter space in the SIRD model. (a) Inferred posterior for global parameters (upper left) and timedependent local parameters (upper right) based on five observations (crosses) of infected (I), recovered (R), and death (D) population densities. The black dot and dashed line indicate the true parameter, bold lines indicate the mean, and shaded areas represent $99 \%$ quantiles. (b) Inference with parameter measurements and a single measurement of fatalities.

### task_015_check_002

Implement Figure 6(a) using five irregular time-point observations of infected/recovered/deceased populations, with explicit compartment/time identity; support different observation masks/counts through the same interface. Infer global recovery/mortality and the time-varying contact rate and predict missing I/R/D values on the regular 0-to-40 grid. Preserve physical-time metadata instead of treating token position as time.

> **paper.md**: Figure 6. Inference of $\infty$-dim parameter space in the SIRD model. (a) Inferred posterior for global parameters (upper left) and timedependent local parameters (upper right) based on five observations (crosses) of infected (I), recovered (R), and death (D) population densities. The black dot and dashed line indicate the true parameter, bold lines indicate the mean, and shaded areas represent $99 \%$ quantiles. (b) Inference with parameter measurements and a single measurement of fatalities.

> **paper.md**: We generated synthetic measurements from infected, recovered, and deceased individuals at irregularly spaced time points and applied the Simformer to estimate the posterior distribution of parameters. The Simformer estimated realistic death and recovery rates and successfully recovers a time-dependent contact rate that aligns with ground truth observations (Fig. 6a). Indeed, as measurements of infections tend towards zero (around timestamp 25, Fig. 6a, orange), the Simformer-posterior for the contact rate increases its uncertainty. This is expected, as we cannot obtain conclusive insights about the contact rate in scenarios with negligible infections.

> **paper.md**: The Simformer uses learnable vector embeddings for identifiers and condition states, as proposed in Weilbach et al. (2023). In cases where parameters or data are functions
> ![](assets/asset_2.jpg)
> 
> Figure 2. Simformer architecture. All variables (parameters and data) are reduced to a token representation which includes the variables' identity, the variables' value (val) as well as the conditional state (latent ( L ) or conditioned (C)). This sequence of tokens is processed by a transformer model; the interaction of variables can be explicitly controlled through an attention mask. The transformer architecture returns a score that is used to generate samples from the score-based diffusion model and can be modified (e.g. to guide the diffusion process).
> of space or time, the node identifier will comprise a shared embedding vector and a random Fourier embedding of the elements in the index set.

### task_015_check_003

Implement Figure 6(b) by conditioning on FOUR measurements of beta(t) and ONE deceased-population measurement, sampling unobserved beta values, global parameters and data jointly. This diagnostic follows the Figure 6(b) fatalities caption and rubric; the main-body sentence says infected and is inconsistent. Exact observed parameter values must remain fixed during sampling; do not use them merely as initialization.

> **paper.md**: Figure 6. Inference of $\infty$-dim parameter space in the SIRD model. (a) Inferred posterior for global parameters (upper left) and timedependent local parameters (upper right) based on five observations (crosses) of infected (I), recovered (R), and death (D) population densities. The black dot and dashed line indicate the true parameter, bold lines indicate the mean, and shaded areas represent $99 \%$ quantiles. (b) Inference with parameter measurements and a single measurement of fatalities.

> **paper.md**: We draw samples from the noise distribution and run the reverse diffusion process on all unobserved variables, while keeping observed variables constant at their conditioning value (Weilbach et al., 2023).

### task_015_check_004

Generate the predictive I/R/D trajectories directly through Simformer conditional sampling; save means and 99% uncertainty quantiles for trajectories and beta(t), with true curves and actual measurement points for both scenarios. Plot global-parameter posterior marginals from generated samples. Do not substitute only simulator rollouts or interpolation for Simformer predictions, force uncertainty to a chosen trend, or require the out-of-scope calibration experiment.

> **paper.md**: Figure 6. Inference of $\infty$-dim parameter space in the SIRD model. (a) Inferred posterior for global parameters (upper left) and timedependent local parameters (upper right) based on five observations (crosses) of infected (I), recovered (R), and death (D) population densities. The black dot and dashed line indicate the true parameter, bold lines indicate the mean, and shaded areas represent $99 \%$ quantiles. (b) Inference with parameter measurements and a single measurement of fatalities.

> **paper.md**: Additionally, as we already demonstrated on the Lotka-Volterra task, the ability of the Simformer to sample any conditional distribution allows us to generate posterior predictive samples without running the simulator. These samples closely match the observed data, further demonstrating the accuracy of the Simformer.

> **addendum.md**: In Section 4.1, the experiments related to the calibration and loglikelihood of Simformer are not required for replication.
> 
> In Section 4.3, the experiments related to the calibration of Simformer are not required for replication.
> 
> In Section 4.4, the experiments related to "Additional details and results on guidance are in Appendix Sec. A3.3" are not required for replication.

> **instructions.txt**: Your submission will be graded based on whether the code necessary for reproducing the paper has been correctly implemented. The code will not be executed during grading.

## task_016 — Application experiment Hodgkin-Huxley with guided diffusion

Execute the application experiment described in Section four point four. Train the Simformer on the Hodgkin-Huxley simulator with summary statistics and energy consumption. First, infer the posterior distribution given only voltage summary statistics. Generate posterior predictive samples for energy consumption. Then, define an observation interval constraint requiring energy to be within the lowest ten percent quantile of posterior predictives. Use guided diffusion to infer the constrained posterior. Generate posterior predictive samples from the constrained posterior and verify energy consumption falls below the threshold. Verify voltage traces remain consistent with observations. Code-Dev scope: implement both unconstrained and energy-guided inference and both predictive routes, without running experiments now or requiring their reported outcomes. Use dense attention and eight layers. Guidance must use the full denoised-score/self-recurrence procedure. The energy threshold is derived from the unconstrained energy posterior predictive and is also explicitly configurable.

### task_016_check_001

Use an eight-layer VE Simformer with fully DENSE attention for Section 4.4 and preserve the seven voltage-statistic coordinates plus the distinct energy coordinate. Model construction, training and both unguided/guided sampling must use the same token/variable convention.

> **paper.md**: The employed transformer model features a token dimension of 50 and represents diffusion time through a 128 -dimensional random Gaussian Fourier embedding. It comprises 6 layers and 4 heads with an attention size of 10, and a widening factor of 3, implying that the feed-forward block expands to a hidden dimension of 150. For the Lotka-Volterra, SIR, and Hodgkin-Huxley tasks, we increased the number of layers to 8. Similar to the above, we used a training batch size of 1000 and an Adam optimizer.

> **paper.md**: Figure 7. Inference in the Hodgkin-Huxley model. (a) Model schematic, observed voltage trace, and associated energy consumption. (b) Marginals of inferred posterior for four parameters. (c) Posterior predictive energy consumption from Simformer (blue) and from simulator outputs (green). (d) Posterior predictive samples from the posterior in (c) using the simulator. (e) Marginals of inferred energy constrained posterior for four parameters. (f) Posterior predictive energy consumption from Simformer (blue) and from simulator outputs (green). Energy constraint as red line. (g) Posterior predictive samples from posterior in (e) using the simulator.

> **addendum.md**: The summary statistics computed for the Hodgkin-Huxley task are:
> 1. Spike count
> 2. Mean of the resting potential
> 3. Standard deviation of the resting potential
> 4. Mean potential in the spiking domain
> 5. Second central moment (variance) of the voltage during the spiking domain
> 6. Third central moment of the voltage during the spiking domain
> 7. Fourth central moment of the voltage during the spiking domain

### task_016_check_002

First condition only on the observed voltage summary statistics, leaving energy and parameters unobserved. Obtain parameter posterior and energy posterior predictive jointly with Simformer. Also run the implemented HH simulator on corresponding posterior parameter samples through a separately callable evaluation path to compute independent voltage traces and energy predictions for comparison. Both routes must be implemented, not executed during approval.

> **paper.md**: We first inferred the posterior distribution given only the summary statistics of the voltage (not the energy) with the Simformer, and we found that, consistent with prior work (Gonçalves et al., 2020), the posterior distribution has wide marginals for some parameters and narrow marginals for others (Fig. 7b). We then used Simformer's ability to sample arbitrary conditionals and generate posterior predictives for energy consumption (Fig. 7c). The posterior predictive distribution of Simformer closely matched the posterior predictive distribution obtained by running the simulator (Fig. 7cd), and the energy cost of different posterior samples varied significantly (Deistler et al., 2022b).

> **instructions.txt**: Your submission will be graded based on whether the code necessary for reproducing the paper has been correctly implemented. The code will not be executed during grading.

### task_016_check_003

Compute the energy upper bound u as the empirical 0.10 quantile of the UNCONSTRAINED posterior-predictive energy distribution for the same voltage observation. Persist this threshold and expose an explicit configurable override. Do not use a tenth of the mean/max, the prior-energy distribution, an already filtered/guided sample or a hardcoded paper number.

> **paper.md**: To identify energy-efficient parameter sets, we then defined an observation interval for the energy consumption (energy must be within the lowest $10 \%$ quantile of posterior predictives), and we used Simformer with guided diffusion to infer the posterior given voltage summary statistics and this constraint on energy consumption. The additional constraint on energy consumption significantly constrained the parameters posterior, in particular the maximal sodium and potassium conductances (Fig. 7e). We generated posterior predictive samples from this new posterior (via Simformer and by running the simulation) and found that their energy consumption indeed lies below the desired threshold (Fig. 7f). Furthermore, the corresponding predictive voltage trace is still in agreement with observations (Fig. 7g).

> **paper.md**: In that line, we use the following general formulation to guide the diffusion process:
> \[
> s_{\phi}\left(\hat{\boldsymbol{x}}_{t}, t \mid c\right) \approx s_{\phi}\left(\hat{\boldsymbol{x}}_{t}, t\right)+\nabla_{\hat{\boldsymbol{x}}_{t}} \log \sigma\left(-s(t) c\left(\hat{\boldsymbol{x}}_{t}\right)\right)
> \]
> 
> Here $\sigma$ denotes the sigmoid function, $s(t)$ is an appropriate scaling function satisfying $s(t) \rightarrow \infty$ as $t \rightarrow 0$, depending on the choice of SDE, and $c$ denotes a constraint function $c(\hat{\boldsymbol{x}}) \leq 0$. For example, to enforce an interval upper bound $u$, we use $c(\hat{\boldsymbol{x}})=\hat{\boldsymbol{x}}-u$.

### task_016_check_004

Run the actual task_006 denoised-score/self-recurrence guidance for c(x)=energy(x)-u while exactly holding the voltage summaries fixed. Energy remains a latent variable constrained by an inequality, not an observed value clamped to u. Compute guidance in consistent physical energy units, including any inverse normalization inside the differentiable constraint; simple sample rejection or clipping cannot replace the guided posterior.

> **paper.md**: To identify energy-efficient parameter sets, we then defined an observation interval for the energy consumption (energy must be within the lowest $10 \%$ quantile of posterior predictives), and we used Simformer with guided diffusion to infer the posterior given voltage summary statistics and this constraint on energy consumption. The additional constraint on energy consumption significantly constrained the parameters posterior, in particular the maximal sodium and potassium conductances (Fig. 7e). We generated posterior predictive samples from this new posterior (via Simformer and by running the simulation) and found that their energy consumption indeed lies below the desired threshold (Fig. 7f). Furthermore, the corresponding predictive voltage trace is still in agreement with observations (Fig. 7g).

> **paper.md**: In that line, we use the following general formulation to guide the diffusion process:
> \[
> s_{\phi}\left(\hat{\boldsymbol{x}}_{t}, t \mid c\right) \approx s_{\phi}\left(\hat{\boldsymbol{x}}_{t}, t\right)+\nabla_{\hat{\boldsymbol{x}}_{t}} \log \sigma\left(-s(t) c\left(\hat{\boldsymbol{x}}_{t}\right)\right)
> \]
> 
> Here $\sigma$ denotes the sigmoid function, $s(t)$ is an appropriate scaling function satisfying $s(t) \rightarrow \infty$ as $t \rightarrow 0$, depending on the choice of SDE, and $c$ denotes a constraint function $c(\hat{\boldsymbol{x}}) \leq 0$. For example, to enforce an interval upper bound $u$, we use $c(\hat{\boldsymbol{x}})=\hat{\boldsymbol{x}}-u$.

> **paper.md**: Algorithm 1 General Guidance
> Require: Number of steps $T$, Min time $T_{\min }$, Max time $T_{\max }$, self-recurrence steps $r$, scaling function $s(t)$ and constraint
>     function $c(x)$, drift coefficient $f(x, t)$, diffusion coefficient $g(t)$, associated mean and standard deviation functions $\mu, \sigma$
>     such that $\hat{\boldsymbol{x}}_{t}=\mu(t) \hat{\boldsymbol{x}}_{0}+\sigma(t) \epsilon$.
>     Set time step $\Delta t=\frac{T_{\text {max }}-T_{\text {min }}}{T}$
>     Sample $\hat{\boldsymbol{x}}_{T} \sim \mathcal{N}\left(\mu_{T}, \sigma_{T} \mathbf{I}\right) \quad / /$ Initialize at terminal distribution
>     for $i=1$ downto $T$ do
>         $t_{i}=T_{\text {max }}-i \cdot \Delta t$
>         for $j=1$ to $r$ do
>             $\epsilon \sim \mathcal{N}(0, \mathbf{I})$
>             $s=s_{\phi}\left(\hat{\boldsymbol{x}}_{t_{i+1}}, t_{i}\right) \quad / /$ Marginal score estimate
>             $\hat{\boldsymbol{x}}_{\sim 0}=\frac{\hat{\boldsymbol{x}}_{t_{i+1}}+\sigma\left(t_{i+1}\right)^{2} \cdot s}{\mu\left(t_{i+1}\right)} \quad / /$ Denoise
>             $\tilde{s}=s+\nabla_{\hat{\boldsymbol{x}}} \log \sigma\left(s(t) c\left(\hat{\boldsymbol{x}}_{\sim 0}\right)\right) \quad / /$ Constraint score
>             $\hat{\boldsymbol{x}}_{t_{i}}=\hat{\boldsymbol{x}}_{t_{i+1}}-\left(f\left(\hat{\boldsymbol{x}}_{t_{i+1}}, t_{i}\right)-g\left(t_{i}\right)^{2} \cdot \tilde{s}\right) \Delta t-g\left(t_{i}\right) \sqrt{\Delta t} \cdot \epsilon$
>             if $r>0$ then
>                     // Resample future point using SDE equations
>                 $\epsilon \sim \mathcal{N}(0, \mathbf{I})$
>                 $\hat{\boldsymbol{x}}_{t_{i+1}}=\hat{\boldsymbol{x}}_{t_{i}}+f\left(\hat{\boldsymbol{x}}_{t_{i+1}}, t_{i}\right) \Delta t+g\left(t_{i}\right) \sqrt{\Delta t} \cdot \epsilon$
>             end if
>         end for
>     end for
>     return $\hat{\boldsymbol{x}}_{T_{\text {min }}}$
> ```

### task_016_check_005

From the guided posterior implement both direct Simformer energy prediction and simulator-based voltage/energy prediction. Save the original threshold, energy distributions and below-threshold fraction, parameter marginals including gNa,gK,Cm,gL, and comparison traces for the unguided/guided cases. Do not fabricate agreement or discard constraint violations solely to claim perfect guidance; the empirical success level is not a static code gate.

> **paper.md**: To identify energy-efficient parameter sets, we then defined an observation interval for the energy consumption (energy must be within the lowest $10 \%$ quantile of posterior predictives), and we used Simformer with guided diffusion to infer the posterior given voltage summary statistics and this constraint on energy consumption. The additional constraint on energy consumption significantly constrained the parameters posterior, in particular the maximal sodium and potassium conductances (Fig. 7e). We generated posterior predictive samples from this new posterior (via Simformer and by running the simulation) and found that their energy consumption indeed lies below the desired threshold (Fig. 7f). Furthermore, the corresponding predictive voltage trace is still in agreement with observations (Fig. 7g).

> **paper.md**: Figure 7. Inference in the Hodgkin-Huxley model. (a) Model schematic, observed voltage trace, and associated energy consumption. (b) Marginals of inferred posterior for four parameters. (c) Posterior predictive energy consumption from Simformer (blue) and from simulator outputs (green). (d) Posterior predictive samples from the posterior in (c) using the simulator. (e) Marginals of inferred energy constrained posterior for four parameters. (f) Posterior predictive energy consumption from Simformer (blue) and from simulator outputs (green). Energy constraint as red line. (g) Posterior predictive samples from posterior in (e) using the simulator.

> **instructions.txt**: Your submission will be graded based on whether the code necessary for reproducing the paper has been correctly implemented. The code will not be executed during grading.

## task_017 — Visualization documentation and final submission packaging

Create comprehensive visualizations for all experiments: C2ST bar charts comparing Simformer variants and baselines on benchmark tasks, posterior distribution plots for Lotka-Volterra, SIRD posterior for global and time-dependent parameters, Hodgkin-Huxley posterior and energy-constrained results. Write README.md documenting the reproduction status, what was implemented, how to run each experiment, and how the codebase relates to paper sections. Create reproduce.sh that orchestrates all experiments end-to-end. Perform lightweight syntax import startup validation checks. Package everything for submission under /home/submission/. This is the final subtask that creates README.md and reproduce.sh, performs final validation, and completes submission packaging. Code-Dev scope: inspect the complete executable workflow and scientific configurations; training, large simulations, MCMC and generation of final figures are not required during approval. Inspect saved-result plotting code and data provenance rather than demanding empirical agreement with the paper.

### task_017_check_001

Connect saved real experiment outputs to Figures 4-7: benchmark C2ST across budgets/variants, both Lotka-Volterra conditions, both SIRD conditions and unguided/guided HH parameter-energy-trace comparisons. Preserve labels, physical units and uncertainty definitions. Plotting only synthetic example arrays or copied paper numbers does not implement the result workflow.

> **paper.md**: Across all four benchmark tasks, the Simformer outperformed neural posterior estimation (NPE), even when the Simformer used a dense attention mask (Fig. 4a). The only exception was the Gaussian linear task with 10k simulations; we show an extended comparison with NRE and NLE in Appendix Fig. A5, results with VPSDE in Appendix Fig. A6). Incorporating domain knowledge into the attention mask of the transformer led to further improvements in the accuracy of the Simformer, particularly in tasks with sparser dependency structures, such as the Linear Gaussian (fully factorized) and SLCP (4 i.i.d. observations).

> **paper.md**: Figure 5. Inference with unstructured observations in the LotkaVolterra model. (a) Posterior predictive (left) and posterior distribution (right) based on four unstructured observations of the prey population density (green crosses), using Simformer with $10^{5}$ simulations. True parameters in dark blue. (b) Same as (a) with nine additional observations of the predator population density. (c) C2ST-performance in estimating arbitrary conditionals (right) or the posterior distribution (left) using the C2ST metric.

> **paper.md**: Figure 6. Inference of $\infty$-dim parameter space in the SIRD model. (a) Inferred posterior for global parameters (upper left) and timedependent local parameters (upper right) based on five observations (crosses) of infected (I), recovered (R), and death (D) population densities. The black dot and dashed line indicate the true parameter, bold lines indicate the mean, and shaded areas represent $99 \%$ quantiles. (b) Inference with parameter measurements and a single measurement of fatalities.

> **paper.md**: Figure 7. Inference in the Hodgkin-Huxley model. (a) Model schematic, observed voltage trace, and associated energy consumption. (b) Marginals of inferred posterior for four parameters. (c) Posterior predictive energy consumption from Simformer (blue) and from simulator outputs (green). (d) Posterior predictive samples from the posterior in (c) using the simulator. (e) Marginals of inferred energy constrained posterior for four parameters. (f) Posterior predictive energy consumption from Simformer (blue) and from simulator outputs (green). Energy constraint as red line. (g) Posterior predictive samples from posterior in (e) using the simulator.

### task_017_check_002

Provide a complete runnable chain from task/budget configuration to simulator data, tokenizer/masks, training, saved checkpoint, conditional sampler, independent references, C2ST and plotting. Check actual call sites and parameter flow, including scalar score shapes, time-versus-metadata features, task-specific N-chain reference protocol, dense application masks and guidance threshold provenance. Correct helpers that are unused while old placeholder paths remain active do not satisfy the workflow.

> **paper.md**: Having defined the tokenizer which processes every $(\boldsymbol{\theta}, \boldsymbol{x})$ pair and the attention mask to specify dependencies within the simulator, the Simformer can be trained using denoising score-matching (Hyvärinen \& Dayan, 2005; Song et al., 2021b): We sample the noise level $t$ for the diffusion model uniformly at random and generate a (partially) noisy sample $\hat{\mathbf{x}}_{t}^{M_{C}}=\left(1-M_{C}\right) \cdot \hat{\mathbf{x}}_{t}+M_{C} \cdot \hat{\mathbf{x}}_{0}$ i.e. variables that we want to condition on remain clean. The loss can then be defined as
> \[
> \begin{aligned}
> & \ell\left(\phi, M_{C}, t, \hat{\mathbf{x}}_{0}, \hat{\mathbf{x}}_{t}\right)= \\
> & \quad\left(1-M_{C}\right) \cdot\left(s_{\phi}^{M_{E}}\left(\hat{\mathbf{x}}_{t}^{M_{C}}, t\right)-\nabla_{\hat{\mathbf{x}}_{t}} \log p_{t}\left(\hat{\mathbf{x}}_{t} \mid \hat{\mathbf{x}}_{0}\right)\right),
> \end{aligned}
> \]

> **paper.md**: We draw samples from the noise distribution and run the reverse diffusion process on all unobserved variables, while keeping observed variables constant at their conditioning value (Weilbach et al., 2023).

> **paper.md**: We evaluated performance in approximating posterior distributions across four benchmark tasks (Lueckmann et al., 2021). For each task, samples for ten ground-truth posteriors are available (Appendix Sec. A2.2), and we assessed performance as classifier two-sample test (C2ST) accuracy to these samples. Here, a score of 0.5 signifies perfect alignment with the ground truth posterior, and 1.0 indicates that a classifier can completely distinguish between the approximation and the ground truth. All results are obtained using the Variance Exploding SDE (VESDE); additional results using the Variance Preserving SDE (VPSDE) can be found in Appendix Sec. A3. See Appendix Sec. A2 for details on the parameterization.

> **paper.md**: We then used the ability of Simformer to sample from arbitrary conditional distribution to simultaneously generate posterior and posterior predictive samples without additional runs of the simulator. The posterior predictives of Simformer capture data and uncertainty in a realistic manner (Fig. 5a).

> **paper.md**: Additionally, as we already demonstrated on the Lotka-Volterra task, the ability of the Simformer to sample any conditional distribution allows us to generate posterior predictive samples without running the simulator. These samples closely match the observed data, further demonstrating the accuracy of the Simformer.

> **paper.md**: To identify energy-efficient parameter sets, we then defined an observation interval for the energy consumption (energy must be within the lowest $10 \%$ quantile of posterior predictives), and we used Simformer with guided diffusion to infer the posterior given voltage summary statistics and this constraint on energy consumption. The additional constraint on energy consumption significantly constrained the parameters posterior, in particular the maximal sodium and potassium conductances (Fig. 7e). We generated posterior predictive samples from this new posterior (via Simformer and by running the simulation) and found that their energy consumption indeed lies below the desired threshold (Fig. 7f). Furthermore, the corresponding predictive voltage trace is still in agreement with observations (Fig. 7g).

### task_017_check_003

Provide reproduce.sh and documented stage-specific/resumable commands that invoke the implemented scientific workflows with their full configurations and preserve distinct outputs. README.md must describe dependencies, source corrections/diagnostic conventions, how modules relate to the paper, and what has or has not run. An import-only script that just prints instructions is not a complete reproduction entry point.

> **paper.md**: We used JAX (Bradbury et al., 2018) as backbone and hydra (Yadan, 2019) to track all configurations. Code to reproduce results is available at https://github.com/ mackelab/simformer. We use SBI (Tejero-Cantero et al., 2020) for reference implementations of baselines.

> **instructions.txt**: Finally, please also include a README.md file that describes what you were able to achieve in your reproduction attempt and explains how your codebase relates to various parts of the reproduction.

> **instructions.txt**: Your submission will be graded based on whether the code necessary for reproducing the paper has been correctly implemented. The code will not be executed during grading.

### task_017_check_004

Assess Code-Dev completion from implemented code and lightweight static/import/interface checks. Do not require real training, long MCMC, full simulator sweeps, generated figures or matching empirical metrics to approve this submission. Keep the addendum exclusions for calibration, log-likelihood and appendix-only guidance experiments; retain Algorithm 1 and appendix methodological details needed for the in-scope main experiments.

> **instructions.txt**: Your submission will be graded based on whether the code necessary for reproducing the paper has been correctly implemented. The code will not be executed during grading.

> **instructions.txt**: Any experiments that are only introduced in the Appendix are not considered core contributions of the paper, and so are out of scope. This is the case even if the experiment is referenced in the main text. HOWEVER, if an experiment is described in the main body of the text, but some of the details used (e.g. details about a prompt used, or a long mathematical expression) have been put in the appendix, then that experiment is considered still in scope. The reason here is that the experiment is described in the main body of the paper.

> **addendum.md**: In Section 4.1, the experiments related to the calibration and loglikelihood of Simformer are not required for replication.
> 
> In Section 4.3, the experiments related to the calibration of Simformer are not required for replication.
> 
> In Section 4.4, the experiments related to "Additional details and results on guidance are in Appendix Sec. A3.3" are not required for replication.

> **paper.md**: In that line, we use the following general formulation to guide the diffusion process:
> \[
> s_{\phi}\left(\hat{\boldsymbol{x}}_{t}, t \mid c\right) \approx s_{\phi}\left(\hat{\boldsymbol{x}}_{t}, t\right)+\nabla_{\hat{\boldsymbol{x}}_{t}} \log \sigma\left(-s(t) c\left(\hat{\boldsymbol{x}}_{t}\right)\right)
> \]
> 
> Here $\sigma$ denotes the sigmoid function, $s(t)$ is an appropriate scaling function satisfying $s(t) \rightarrow \infty$ as $t \rightarrow 0$, depending on the choice of SDE, and $c$ denotes a constraint function $c(\hat{\boldsymbol{x}}) \leq 0$. For example, to enforce an interval upper bound $u$, we use $c(\hat{\boldsymbol{x}})=\hat{\boldsymbol{x}}-u$.

