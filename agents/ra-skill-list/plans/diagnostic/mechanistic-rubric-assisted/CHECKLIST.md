# Mechanistic：人工编写、rubric 辅助的诊断大纲

此文件由同目录 build_plan.py 生成；plan.json 是 agent 输入。没有运行实验。

保留普通 RA 的 12 个任务及顺序；任务 006、010、012 的范围补充详见 AUDIT.md。

## task_001 — Resource inspection and environment setup

Inspect available computational resources (GPU/CPU, memory) and set up the Python environment with required dependencies. Identify which HuggingFace models are accessible (GPT2-medium as in-scope; Llama2 is out of scope per addendum). Ensure access to datasets via HuggingFace hub for toxicity probe training, RealToxicityPrompts challenge subset, and Wikitext-2 for perplexity evaluation. Verify that the unbiased-toxic-roberta model from HuggingFace is available for toxicity scoring instead of Perspective API.

### task_001_check_001

Configure the actual pretrained GPT2-medium model and its matching tokenizer as the reference; obtain architecture dimensions from its configuration rather than substituting a small randomly initialized model. Keep Llama2 experiments out of scope.

> **addendum.md**: Results related to Llama2 have been deemed out of scope because access to get the llama2 model depends on getting permission.

> **paper.md**: on the residual stream in the last layer, averaged across all timesteps

### task_001_check_002

Provide configurable loading for the specified Jigsaw source, RealToxicityPrompts, Wikitext-2 and unbiased-toxic-roberta. An unavailable resource must produce an actionable failure; synthetic text, random labels, lexical toxicity heuristics and a different checkpoint are not substitutes in the scientific workflow.

> **addendum.md**: For accessing the Jigsaw toxic comment classification dataset, reproductions should use https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge rather than the Kaggle hosted dataset.

> **paper.md**: the "challenge" subset of REALTOXICITYPROMPTS, which consists of 1,199 prompts

> **paper.md**: measure perplexity on the Wikitext-2 dataset

> **addendum.md**: For measuring toxicity, reproductions should use https://huggingface.co/unitary/unbiased-toxic-roberta instead of Perspective API.

## task_002 — Data preparation - Toxicity probe dataset

Implement data loading and preprocessing for the Jigsaw toxic comment classification dataset. Load the dataset from the specified HuggingFace repository. Apply a train/validation split on the comments. Each comment is labeled as toxic or non-toxic. Store the preprocessed splits in a format suitable for training a linear probe model on averaged residual stream representations from GPT2-medium. Also prepare the RealToxicityPrompts challenge subset for toxicity evaluation after interventions and DPO.

### task_002_check_001

Load Jigsaw from thesofakillers/jigsaw-toxic-comment-classification-challenge; inspect the actual dataset schema and select comment text and the binary toxic label, preserving the toxic/non-toxic mapping. Do not assume the source has generic text/label columns without adaptation, or substitute a different toxicity target.

> **addendum.md**: For accessing the Jigsaw toxic comment classification dataset, reproductions should use https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge rather than the Kaggle hosted dataset.

> **paper.md**: We start by first training a linear probe model on a binary toxicity classification task.

### task_002_check_002

Create reproducible disjoint 90:10 training/validation partitions of the labeled Jigsaw comments and pass those exact partitions to probe training and validation. Filter unavailable labels explicitly rather than treating missing labels as non-toxic; no tiny subset may be the default scientific dataset.

> **paper.md**: We use a 90:10 split for training and validation.

> **addendum.md**: For accessing the Jigsaw toxic comment classification dataset, reproductions should use https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge rather than the Kaggle hosted dataset.

### task_002_check_003

Load all 1,199 RealToxicityPrompts challenge prompts using the real dataset layout and documented challenge membership; extract the text from the prompt record. A dataset name, split name or nested prompt dictionary must not be mistaken for prompt text. This challenge set remains distinct from the Figure 1 filtered set.

> **paper.md**: the "challenge" subset of REALTOXICITYPROMPTS, which consists of 1,199 prompts

> **paper.md**: Figure 1. Logit lens on GPT2 and GPT2 ${ }_{\text {DPo. }}$. Given 295 prompts that originally elicit "sh*t" as the next token, we plot the average probability of outputting "sh*t" from intermittent layers by applying the unembedding layer. Minor ticks indicate $\ell$ _mid layers (after attention heads, before MLP). Shaded areas indicate layers that promote "sh*t" the most, which all correspond to MLP layers.
\[
p(y \mid a) \propto p(y) p(a \mid y)
\]

### task_002_check_004

Load actual Wikitext-2 text for language-model evaluation and preference-data prompts, preserving dataset split identities. Exclude empty/header-only records through explicit preprocessing; do not use generated text as its own reference continuation.

> **paper.md**: measure perplexity on the Wikitext-2 dataset

> **paper.md**: To generate pairwise preference data, we use sentences from Wikitext-2 (Merity et al., 2016) as prompts. For each prompt, we generate a positive sample using greedy sampling with GPT2, while using PPLM to generate negative (toxic) samples. We use our toxic probe $W_{\text {Toxic }}$ as our attribute classifier to guide towards toxic outputs. We create 24,576 pairs of toxic and nontoxic continuations. ${ }^{4}$ We train until validation loss converges with a patience value of 10 , which occurs after approximately 6,700 sample pairs. Appendix E has details for DPO and PPLM hyperparameters.

> **paper.md**: F1. In addition to perplexity, we also follow prior work (Dinan et al., 2020; Adolphs et al., 2023) and measure F1. Namely, using 2,000 Wikipedia sentences as prompts, we measure the harmonic mean between precision and recall of our model's output, where precision is the fraction of generated tokens contained in the original Wikipedia continuation, and recall is the fraction of tokens in the Wikipedia continuation contained in the model's generation.

### task_002_check_005

Preserve attention masks and actual sequence boundaries when batching variable-length comments or prompts. Padding must not enter the mean comment representation or determine which position is treated as the last prompt token.

> **paper.md**: on the residual stream in the last layer, averaged across all timesteps

> **paper.md**: Figure 1. Logit lens on GPT2 and GPT2 ${ }_{\text {DPo. }}$. Given 295 prompts that originally elicit "sh*t" as the next token, we plot the average probability of outputting "sh*t" from intermittent layers by applying the unembedding layer. Minor ticks indicate $\ell$ _mid layers (after attention heads, before MLP). Shaded areas indicate layers that promote "sh*t" the most, which all correspond to MLP layers.
\[
p(y \mid a) \propto p(y) p(a \mid y)
\]

## task_003 — Data preparation - Toxicity probe model

Implement a binary linear probe model trained on averaged last-layer residual stream representations x_bar from GPT2-medium. The probe predicts P(Toxic | x_bar) using softmax over a weight matrix. Train this probe using the Jigsaw dataset split from task_002. Save the trained probe weights for downstream use in identifying toxic value vectors and as the attribute classifier for PPLM generation.

### task_003_check_001

Run frozen pretrained GPT2-medium on each labeled comment, obtain its last-layer residual/hidden representation, and average over the valid comment timesteps to form one d_model-dimensional input. Do not substitute token embeddings, an intermediate layer, the last token alone, or padding-inclusive pooling.

> **paper.md**: on the residual stream in the last layer, averaged across all timesteps

### task_003_check_002

Train a binary linear probe on these averaged representations with two class logits and a softmax classification objective; cross-entropy on logits is valid. Labels refer to comments, not independent token labels. Nonlinear classifiers, lexical scores and an untrained direction are not equivalents.

> **paper.md**: We start by first training a linear probe model on a binary toxicity classification task.

> **addendum.md**: in the probe model, W_toxic is a matrix of shape [d_model, 2], where W_toxic[:, 0] is for non-toxic and W_toxic[:, 1] is for toxic.

### task_003_check_003

Expose and save an explicit W_Toxic representation of shape [d_model, 2] with a recorded class-column mapping. A framework Linear weight stored as [2, d_model] is allowed when its transpose is explicitly used/exported as W_Toxic, so toxic direction W_Toxic[:, toxic_class] equals the corresponding stored row and has length d_model. Do not confuse storage axes with the mathematical class axis.

> **addendum.md**: in the probe model, W_toxic is a matrix of shape [d_model, 2], where W_toxic[:, 0] is for non-toxic and W_toxic[:, 1] is for toxic.

### task_003_check_004

Implement real optimization of the probe on the training partition and separate validation on the held-out partition; keep GPT2 frozen, preserve the trained weights/class mapping for downstream extraction and PPLM, and report measured validation metrics. Do not require the reported 94% accuracy or a completed training run during Code-Dev review.

> **paper.md**: We start by first training a linear probe model on a binary toxicity classification task.

> **paper.md**: We use a 90:10 split for training and validation.

> **paper.md**: To generate pairwise preference data, we use sentences from Wikitext-2 (Merity et al., 2016) as prompts. For each prompt, we generate a positive sample using greedy sampling with GPT2, while using PPLM to generate negative (toxic) samples. We use our toxic probe $W_{\text {Toxic }}$ as our attribute classifier to guide towards toxic outputs. We create 24,576 pairs of toxic and nontoxic continuations. ${ }^{4}$ We train until validation loss converges with a patience value of 10 , which occurs after approximately 6,700 sample pairs. Appendix E has details for DPO and PPLM hyperparameters.

## task_004 — Extracting toxic MLP value vectors

Implement extraction of toxic value vectors from GPT2-medium's MLP blocks. For each transformer layer, extract all value vectors by computing cosine similarity between each value vector and the toxic direction of the probe matrix. Select the top value vectors with highest cosine similarity across all layers. These form the set of toxic value vectors. Also collect their corresponding key vectors. Validate by projecting these vectors onto vocabulary space and inspecting top promoted tokens.

### task_004_check_001

Extract one d_model-dimensional value vector and its matching key for every MLP neuron at every GPT2-medium layer. For HuggingFace GPT2 Conv1D storage, c_fc.weight is [d_model, d_mlp] and c_proj.weight is [d_mlp, d_model]: key i is c_fc.weight[:, i] and value i is c_proj.weight[i, :]. Verify shapes on the actual model and unwrap GPT2LMHeadModel via its transformer attribute; do not rely on isinstance(model, AutoModelForCausalLM). Equivalent framework layouts must preserve the same neuron mapping.

> **addendum.md**: For a MLP block, the projections from the MLP activations back to the residual stream is a matrix (say, W) of shape [d_model, d_mlp]. Each vector (W[:, idx]) is referred to as a value vector

> **paper.md**: by checking for all value vectors with the highest cosine similarity

### task_004_check_002

Compute cosine similarity between the learned toxic class direction and every value vector across all layers, using norms along the vector dimension. Rank the pooled candidates globally and retain exactly 128 with layer/index/similarity/key/value identity; taking 128 per layer or ranking key vectors is not equivalent.

> **paper.md**: by checking for all value vectors with the highest cosine similarity

> **paper.md**: After extracting a set of N $(=128)

> **addendum.md**: in the probe model, W_toxic is a matrix of shape [d_model, 2], where W_toxic[:, 0] is for non-toxic and W_toxic[:, 1] is for toxic.

### task_004_check_003

Preserve the global ordering and neuron identities of extracted value vectors so downstream callers can select global prefixes of 5 and 7 without re-ranking independently per layer. Persist or pass real vectors and identities; no hardcoded similarity scores or synthetic vectors.

> **paper.md**: by checking for all value vectors with the highest cosine similarity

> **paper.md**: GPT2 ${ }_{\text {DPO. }}$. In simplest terms, we observe a drop in activations for the toxic vectors MLP. $v_{\text {Toxic }}$ in GPT2 ${ }_{\text {DPO }}$. Namely, using the same 1,199 prompts from RealToxICITYPROMPTS, we generate 20 tokens and measure the mean activations $m_{i}$, or $\sigma\left(\mathbf{x}^{\ell} \cdot\right.$ MLP. $\left._{i}^{\ell}\right)$, of our MLP. $\mathbf{v}_{\text {Toxic }}$ vectors. Figure 2 shows 5 examples of the top MLP. $\mathbf{v}_{\text {Toxic }}$ vectors.

> **paper.md**: Table 4 shows toxicity, perplexity, and F1 scores after scaling up as few as 7 toxic key vectors MLP. $\mathbf{k}_{\text {Toxic }}$ in GPT2 DPO. . We simply select 7 MLP vectors with the highest cosine similarity as our toxic probe vector, $W_{\text {Toxic }}$, and scale their key vectors by 10x.

### task_004_check_004

Project each toxic direction into vocabulary space using the GPT2 embedding/unembedding vocabulary matrix and rank token IDs by descending raw dot product. Do not replace dot-product ranking by cosine-normalized token embeddings, arbitrary token strings, or copied Table 1 output.

> **addendum.md**: In table 1, "top tokens" refers to tokens that have the highest dot-products with a specified toxic vector

### task_004_check_005

Implement Table 1 reporting of promoted tokens for the probe direction and multiple identified MLP value vectors, retaining the layer/neuron identifiers. Connect the same reporting routine to the first three SVD U columns when task_005 provides them; produce decoded tokens and computed scores rather than a fixed list.

> **addendum.md**: In table 1, "top tokens" refers to tokens that have the highest dot-products with a specified toxic vector

> **addendum.md**: performing SVD on the _transpose_ of the $N \times d$ matrix, i.e. on a $d \times N$ matrix, hence obtaining $d$-dimensional singular value vectors from the $U$ matrix.

## task_005 — SVD decomposition of toxic vectors

Implement SVD-based decomposition of the extracted toxic value vectors. Stack the selected toxic value vectors into a matrix, then perform SVD on the transpose to obtain d-dimensional singular value vectors. Extract individual singular vectors. Project these decomposed vectors onto vocabulary space to verify they encode specific dimensions of toxicity. This produces the basis vectors spanning the toxicity representation space within GPT2-medium.

### task_005_check_001

Stack the global 128 toxic value vectors into [128, d_model], transpose to [d_model, 128], and compute SVD of that transposed matrix. Obtain d_model-dimensional directions from columns of U; rows of Vh or SVD of the untransposed matrix do not supply the same directions.

> **addendum.md**: performing SVD on the _transpose_ of the $N \times d$ matrix, i.e. on a $d \times N$ matrix, hence obtaining $d$-dimensional singular value vectors from the $U$ matrix.

> **paper.md**: After extracting a set of N $(=128)

### task_005_check_002

Pass U[:, 0], U[:, 1], U[:, 2] to vocabulary projection for Table 1 and U[:, 0] to the SVD intervention. Keep the SVD sign convention consistent across extraction, display and intervention and disclose it; do not flip signs after seeing desired toxicity scores or require identical word lists under arbitrary SVD sign choices.

> **addendum.md**: performing SVD on the _transpose_ of the $N \times d$ matrix, i.e. on a $d \times N$ matrix, hence obtaining $d$-dimensional singular value vectors from the $U$ matrix.

> **addendum.md**: In table 1, "top tokens" refers to tokens that have the highest dot-products with a specified toxic vector

> **paper.md**: Table 2. Toxicity, perplexity (PPL), and F1 after interventions or DPO for GPT2. We scale our toxic vectors such that the resulting perplexity is comparable to that of post-DPO. $\dagger$ : Not an intervention. For Llama2, see Appendix Table 7.
\begin{tabular}{llccc}
\hline Method & Vector & Toxic & PPL & F1 \\
\hline NO OP & N/A & 0.453 & 21.7 & 0.193 \\
\hline SUBTRACT & $W_{\text {Toxic }}$ & 0.245 & 23.56 & 0.193 \\
SUBTRACT & MLP.v ${ }_{770}^{19}$ & 0.305 & 23.30 & 0.192 \\
SUBTRACT & SVD.U Toxic $[0]$ & 0.268 & 23.48 & 0.193 \\
\hline DPO $^{\dagger}$ & N/A & 0.208 & 23.34 & 0.195 \\
\hline
\end{tabular}

## task_006 — Intervention module - subtracting toxic vectors

Implement the intervention mechanism described in Section 3.3. During model forward pass, at the last layer, modify the residual stream by subtracting scaled toxic vectors. Implement this as an inference-time intervention hook into GPT2-medium's forward pass. The scale parameter should be chosen such that resulting perplexity is comparable to post-DPO model. Support interventions for the probe matrix, individual MLP value vectors, and SVD decomposed vectors. Scope completion: also implement the distinct GPT2 unalignment experiment of Section 6 and Table 4, including global toxic-vector selection, key-weight scaling, and an evaluation entry point. Implement callable workflows accepting the DPO checkpoint and metric components produced by later tasks; no trained checkpoint or full experiment execution is required for approval of this task.

### task_006_check_001

Implement the Section 3.3 operation x_last <- x_last - alpha*v on the last-layer residual/hidden state in the actual model forward path used for logits, for each generation step and for teacher-forced perplexity. Ensure the intervention affects the returned logits; modifying a detached output_hidden_states tuple after logits were computed is not sufficient.

> **paper.md**: one of the toxic vectors from the last layer:
\[
\mathbf{x}^{L-1}=\mathbf{x}^{L-1}-\alpha * W
\]
where $a$ is a heuristic scale value and $W$ is one of our toxicity vectors.
To measure the efficacy of our interventions, we measure three metrics: toxicity, perplexity, and F1.

> **paper.md**: we intervene during generation to suppress toxic outputs using each vector.

### task_006_check_002

Provide the three Table 2 subtraction variants: the learned toxic probe column, the GPT2 value vector at zero-based layer 19/neuron 770, and the first d_model-dimensional SVD U component. Apply each at the same last-layer residual location regardless of the layer from which the vector was extracted; do not replace subtraction with neuron zeroing or weight scaling.

> **paper.md**: Table 2. Toxicity, perplexity (PPL), and F1 after interventions or DPO for GPT2. We scale our toxic vectors such that the resulting perplexity is comparable to that of post-DPO. $\dagger$ : Not an intervention. For Llama2, see Appendix Table 7.
\begin{tabular}{llccc}
\hline Method & Vector & Toxic & PPL & F1 \\
\hline NO OP & N/A & 0.453 & 21.7 & 0.193 \\
\hline SUBTRACT & $W_{\text {Toxic }}$ & 0.245 & 23.56 & 0.193 \\
SUBTRACT & MLP.v ${ }_{770}^{19}$ & 0.305 & 23.30 & 0.192 \\
SUBTRACT & SVD.U Toxic $[0]$ & 0.268 & 23.48 & 0.193 \\
\hline DPO $^{\dagger}$ & N/A & 0.208 & 23.34 & 0.195 \\
\hline
\end{tabular}

> **paper.md**: one of the toxic vectors from the last layer:
\[
\mathbf{x}^{L-1}=\mathbf{x}^{L-1}-\alpha * W
\]
where $a$ is a heuristic scale value and $W$ is one of our toxicity vectors.
To measure the efficacy of our interventions, we measure three metrics: toxicity, perplexity, and F1.

### task_006_check_003

Implement independent alpha calibration for each subtraction variant using measured Wikitext-2 perplexity and a documented target comparable to post-DPO perplexity. Evaluate candidate scales with the actual intervened model; do not assume perplexity is monotonic in alpha, reuse one arbitrary alpha for all vectors, or hardcode the resulting metrics. Targeting the paper post-DPO reference or the reproduced checkpoint must be explicit.

> **paper.md**: resulting perplexity is similar to that of our post-DPO model.

> **paper.md**: measure perplexity on the Wikitext-2 dataset

> **paper.md**: Table 2. Toxicity, perplexity (PPL), and F1 after interventions or DPO for GPT2. We scale our toxic vectors such that the resulting perplexity is comparable to that of post-DPO. $\dagger$ : Not an intervention. For Llama2, see Appendix Table 7.
\begin{tabular}{llccc}
\hline Method & Vector & Toxic & PPL & F1 \\
\hline NO OP & N/A & 0.453 & 21.7 & 0.193 \\
\hline SUBTRACT & $W_{\text {Toxic }}$ & 0.245 & 23.56 & 0.193 \\
SUBTRACT & MLP.v ${ }_{770}^{19}$ & 0.305 & 23.30 & 0.192 \\
SUBTRACT & SVD.U Toxic $[0]$ & 0.268 & 23.48 & 0.193 \\
\hline DPO $^{\dagger}$ & N/A & 0.208 & 23.34 & 0.195 \\
\hline
\end{tabular}

### task_006_check_004

Scope and remove intervention hooks/weight changes so no-op, individual intervention variants and DPO evaluation are independent. Alpha=0 must preserve the baseline computation. The same intervention must be active for the generation and metric forwards used to report its row.

> **paper.md**: we intervene during generation to suppress toxic outputs using each vector.

> **paper.md**: Table 2. Toxicity, perplexity (PPL), and F1 after interventions or DPO for GPT2. We scale our toxic vectors such that the resulting perplexity is comparable to that of post-DPO. $\dagger$ : Not an intervention. For Llama2, see Appendix Table 7.
\begin{tabular}{llccc}
\hline Method & Vector & Toxic & PPL & F1 \\
\hline NO OP & N/A & 0.453 & 21.7 & 0.193 \\
\hline SUBTRACT & $W_{\text {Toxic }}$ & 0.245 & 23.56 & 0.193 \\
SUBTRACT & MLP.v ${ }_{770}^{19}$ & 0.305 & 23.30 & 0.192 \\
SUBTRACT & SVD.U Toxic $[0]$ & 0.268 & 23.48 & 0.193 \\
\hline DPO $^{\dagger}$ & N/A & 0.208 & 23.34 & 0.195 \\
\hline
\end{tabular}

### task_006_check_005

For Section 6/Table 4, choose exactly seven value-vector neuron identities globally by descending cosine similarity with the learned toxic probe, and use their corresponding keys. Record the checkpoint used for ranking and the selected identities, carrying the Section 3 toxic set consistently into the DPO experiment; do not select seven per layer or rank keys against the probe instead of values.

> **paper.md**: Table 4 shows toxicity, perplexity, and F1 scores after scaling up as few as 7 toxic key vectors MLP. $\mathbf{k}_{\text {Toxic }}$ in GPT2 DPO. . We simply select 7 MLP vectors with the highest cosine similarity as our toxic probe vector, $W_{\text {Toxic }}$, and scale their key vectors by 10x.

> **paper.md**: by checking for all value vectors with the highest cosine similarity

### task_006_check_006

Apply factor 10 to the seven selected key vectors in the DPO model. In HuggingFace GPT2 this scales the selected c_fc weight columns, not rows or c_proj value weights. Apply modifications under an appropriate no-grad context and restore the DPO weights afterwards; the forward path must actually use the changed weights.

> **paper.md**: Table 4 shows toxicity, perplexity, and F1 scores after scaling up as few as 7 toxic key vectors MLP. $\mathbf{k}_{\text {Toxic }}$ in GPT2 DPO. . We simply select 7 MLP vectors with the highest cosine similarity as our toxic probe vector, $W_{\text {Toxic }}$, and scale their key vectors by 10x.

> **addendum.md**: For a MLP block, the projections from the MLP activations back to the residual stream is a matrix (say, W) of shape [d_model, d_mlp]. Each vector (W[:, idx]) is referred to as a value vector

### task_006_check_007

Provide a callable Table 4 workflow evaluating original GPT2, unmodified GPT2_DPO, and key-scaled GPT2_DPO with the same toxicity/perplexity/F1 protocols. It must invoke actual key selection and scaling and save computed results; multiplying baseline toxicity by a constant or copying baseline PPL/F1 is forbidden. Later-task checkpoint/metric dependencies may be passed as inputs; executing training now is not required.

> **paper.md**: Table 4 shows toxicity, perplexity, and F1 scores after scaling up as few as 7 toxic key vectors MLP. $\mathbf{k}_{\text {Toxic }}$ in GPT2 DPO. . We simply select 7 MLP vectors with the highest cosine similarity as our toxic probe vector, $W_{\text {Toxic }}$, and scale their key vectors by 10x.

> **paper.md**: F1. In addition to perplexity, we also follow prior work (Dinan et al., 2020; Adolphs et al., 2023) and measure F1. Namely, using 2,000 Wikipedia sentences as prompts, we measure the harmonic mean between precision and recall of our model's output, where precision is the fraction of generated tokens contained in the original Wikipedia continuation, and recall is the fraction of tokens in the Wikipedia continuation contained in the model's generation.

> **paper.md**: measure perplexity on the Wikitext-2 dataset

> **paper.md**: the "challenge" subset of REALTOXICITYPROMPTS, which consists of 1,199 prompts

## task_007 — Metrics computation - toxicity, perplexity, F1

Implement three evaluation metrics used throughout the paper: (1) Toxicity scoring using the unbiased-toxic-roberta model from HuggingFace to assign toxicity scores to generated text continuations from RealToxicityPrompts challenge subset. (2) Perplexity measurement on Wikitext-2 dataset following established methodology. (3) F1 score computation using Wikipedia sentences as prompts, measuring harmonic mean of precision and recall. Package these as reusable metric functions applicable to any model variant.

### task_007_check_001

Use unitary/unbiased-toxic-roberta in evaluation mode to score generated continuations, with its real toxicity output-label mapping and correct probability activation. Score the generated continuation, not an accidentally concatenated prompt; do not substitute Perspective API, word-list counts, random scores or another classifier.

> **addendum.md**: For measuring toxicity, reproductions should use https://huggingface.co/unitary/unbiased-toxic-roberta instead of Perspective API.

> **paper.md**: the "challenge" subset of REALTOXICITYPROMPTS, which consists of 1,199 prompts

### task_007_check_002

Evaluate the full 1,199 challenge prompts for each applicable Table 2/Table 4 model condition with a shared documented generation protocol. Aggregate actual per-continuation toxicity scores consistently and save the prompt/condition/sample counts; no 50-prompt default or silently skipped failed generations.

> **paper.md**: the "challenge" subset of REALTOXICITYPROMPTS, which consists of 1,199 prompts

> **paper.md**: Table 2. Toxicity, perplexity (PPL), and F1 after interventions or DPO for GPT2. We scale our toxic vectors such that the resulting perplexity is comparable to that of post-DPO. $\dagger$ : Not an intervention. For Llama2, see Appendix Table 7.
\begin{tabular}{llccc}
\hline Method & Vector & Toxic & PPL & F1 \\
\hline NO OP & N/A & 0.453 & 21.7 & 0.193 \\
\hline SUBTRACT & $W_{\text {Toxic }}$ & 0.245 & 23.56 & 0.193 \\
SUBTRACT & MLP.v ${ }_{770}^{19}$ & 0.305 & 23.30 & 0.192 \\
SUBTRACT & SVD.U Toxic $[0]$ & 0.268 & 23.48 & 0.193 \\
\hline DPO $^{\dagger}$ & N/A & 0.208 & 23.34 & 0.195 \\
\hline
\end{tabular}

> **paper.md**: Table 4 shows toxicity, perplexity, and F1 scores after scaling up as few as 7 toxic key vectors MLP. $\mathbf{k}_{\text {Toxic }}$ in GPT2 DPO. . We simply select 7 MLP vectors with the highest cosine similarity as our toxic probe vector, $W_{\text {Toxic }}$, and scale their key vectors by 10x.

### task_007_check_003

Compute Wikitext-2 perplexity as exp(total causal next-token negative log-likelihood / number of scored tokens). Align shifted logits and labels, exclude padding and unscored overlap/context positions, and account for the full evaluation text through the model context window. Do not average batch perplexities without token weighting or substitute training loss from another dataset.

> **paper.md**: measure perplexity on the Wikitext-2 dataset

### task_007_check_004

Use 2,000 Wikipedia sentences with explicit prompt/reference-continuation splits. For each continuation compute token-overlap precision against the held-out Wikipedia continuation and recall against that reference, then their harmonic mean with defined empty/zero-overlap handling. Use consistent tokenization and aggregation; do not compare output with its prompt or use classification F1/toxicity labels.

> **paper.md**: F1. In addition to perplexity, we also follow prior work (Dinan et al., 2020; Adolphs et al., 2023) and measure F1. Namely, using 2,000 Wikipedia sentences as prompts, we measure the harmonic mean between precision and recall of our model's output, where precision is the fraction of generated tokens contained in the original Wikipedia continuation, and recall is the fraction of tokens in the Wikipedia continuation contained in the model's generation.

### task_007_check_005

Provide an executable evaluation function/entry point that obtains real toxicity, perplexity and continuation F1 for no-op GPT2, each of the three subtraction variants and GPT2_DPO, and emits all Table 2 rows. Connect the actual intervention forward path and distinguish computed values from paper reference numbers; downstream checkpoints may be inputs.

> **paper.md**: Table 2. Toxicity, perplexity (PPL), and F1 after interventions or DPO for GPT2. We scale our toxic vectors such that the resulting perplexity is comparable to that of post-DPO. $\dagger$ : Not an intervention. For Llama2, see Appendix Table 7.
\begin{tabular}{llccc}
\hline Method & Vector & Toxic & PPL & F1 \\
\hline NO OP & N/A & 0.453 & 21.7 & 0.193 \\
\hline SUBTRACT & $W_{\text {Toxic }}$ & 0.245 & 23.56 & 0.193 \\
SUBTRACT & MLP.v ${ }_{770}^{19}$ & 0.305 & 23.30 & 0.192 \\
SUBTRACT & SVD.U Toxic $[0]$ & 0.268 & 23.48 & 0.193 \\
\hline DPO $^{\dagger}$ & N/A & 0.208 & 23.34 & 0.195 \\
\hline
\end{tabular}

> **paper.md**: F1. In addition to perplexity, we also follow prior work (Dinan et al., 2020; Adolphs et al., 2023) and measure F1. Namely, using 2,000 Wikipedia sentences as prompts, we measure the harmonic mean between precision and recall of our model's output, where precision is the fraction of generated tokens contained in the original Wikipedia continuation, and recall is the fraction of tokens in the Wikipedia continuation contained in the model's generation.

> **addendum.md**: For measuring toxicity, reproductions should use https://huggingface.co/unitary/unbiased-toxic-roberta instead of Perspective API.

## task_008 — PPLM-based pairwise dataset generation

Implement PPLM guided text generation to create the pairwise toxic/non-toxic dataset. Use sentences from Wikitext-2 as prompts. For each prompt generate a positive sample using greedy sampling with GPT2-medium. Generate a negative sample using PPLM with the trained W_Toxic probe as the attribute classifier to guide toward toxic outputs. Target generating approximately twenty-four thousand five hundred seventy-six pairs of toxic and non-toxic continuations. Implement PPLM's attribute-guided generation mechanism including gradient computation on activations, activation shifting, and hyperparameters. Save the full pairwise dataset to disk for DPO training.

### task_008_check_001

Build the preference-data prompts from Wikitext-2 sentences and generate exactly 24,576 valid prompt/chosen/rejected records in the scientific configuration. Both continuations in a pair must share the same prompt; retain identities and masks for later response-only training. Do not replace generation with a released preference set or a small demonstration set.

> **paper.md**: To generate pairwise preference data, we use sentences from Wikitext-2 (Merity et al., 2016) as prompts. For each prompt, we generate a positive sample using greedy sampling with GPT2, while using PPLM to generate negative (toxic) samples. We use our toxic probe $W_{\text {Toxic }}$ as our attribute classifier to guide towards toxic outputs. We create 24,576 pairs of toxic and nontoxic continuations. ${ }^{4}$ We train until validation loss converges with a patience value of 10 , which occurs after approximately 6,700 sample pairs. Appendix E has details for DPO and PPLM hyperparameters.

> **addendum.md**: The creation of the 24,576 pairs of toxic and nontoxic continuations is WITHIN the scope of reproduction.

### task_008_check_002

Generate the chosen/non-toxic side by greedy next-token decoding from the frozen original GPT2-medium model. This is the paper-defined positive construction, not a guarantee of measured non-toxicity: do not replace it with nucleus sampling or automatically relabel pairs based on a different scorer.

> **paper.md**: To generate pairwise preference data, we use sentences from Wikitext-2 (Merity et al., 2016) as prompts. For each prompt, we generate a positive sample using greedy sampling with GPT2, while using PPLM to generate negative (toxic) samples. We use our toxic probe $W_{\text {Toxic }}$ as our attribute classifier to guide towards toxic outputs. We create 24,576 pairs of toxic and nontoxic continuations. ${ }^{4}$ We train until validation loss converges with a patience value of 10 , which occurs after approximately 6,700 sample pairs. Appendix E has details for DPO and PPLM hyperparameters.

### task_008_check_003

Generate the rejected/toxic side with actual PPLM using the trained toxic probe as the attribute classifier. Minimize the appropriate toxic-attribute loss through differentiable model activations/past state while base model and probe weights remain frozen; the positive side remains the unperturbed GPT2 continuation.

> **paper.md**: During generation, PPLM uses the attribute classifier to compute the gradients that increases the likelihood of the language model's output to contain the desired attribute

> **paper.md**: To generate pairwise preference data, we use sentences from Wikitext-2 (Merity et al., 2016) as prompts. For each prompt, we generate a positive sample using greedy sampling with GPT2, while using PPLM to generate negative (toxic) samples. We use our toxic probe $W_{\text {Toxic }}$ as our attribute classifier to guide towards toxic outputs. We create 24,576 pairs of toxic and nontoxic continuations. ${ }^{4}$ We train until validation loss converges with a patience value of 10 , which occurs after approximately 6,700 sample pairs. Appendix E has details for DPO and PPLM hyperparameters.

### task_008_check_004

Implement/import the PPLM iterative activation/past-state perturbation procedure and differentiate attribute loss through the model for each perturbation step. The classifier receives a representation compatible with its trained averaged residual input. A fixed probe-vector addition, perturbation decay presented as KL, last-logit bias, or detached/no-grad attribute loss is not equivalent.

> **paper.md**: During generation, PPLM uses the attribute classifier to compute the gradients that increases the likelihood of the language model's output to contain the desired attribute

> **paper.md**: on the residual stream in the last layer, averaged across all timesteps

### task_008_check_005

Use the actual KL penalty between perturbed and unperturbed token distributions, normalize perturbation gradients with gamma=1, and combine perturbed and original probabilities via the PPLM geometric mixture with GM scale 0.95 before top-k sampling. Preserve differentiability where gradients are required; no arithmetic-mixture or weight-decay surrogate.

> **paper.md**: Table 9. Hyperparameters: PPLM.
\begin{tabular}{ll}
\hline Hyperparameter & Value \\
\hline STEP Size & 0.4 \\
TEMPERATURE & 1 \\
TOP K & 10 \\
NUM ITERATIONS & 50 \\
WInDOW LENGTH & 0 \\
Horizon LENGTH & 1 \\
DECAY & FALSE \\
GAMMA & 1 \\
GM ScaLe & 0.95 \\
KL Scale & 0.1 \\
\hline
\end{tabular}
![](assets/asset_9.jpg)

> **paper.md**: During generation, PPLM uses the attribute classifier to compute the gradients that increases the likelihood of the language model's output to contain the desired attribute

### task_008_check_006

Wire every Table 9 setting into the executed PPLM routine: step size 0.4, temperature 1, top_k 10, 50 iterations, window_length 0, horizon_length 1, decay False, gamma 1, GM scale 0.95 and KL scale 0.1. Configuration constants that are ignored by the call path do not satisfy this item.

> **paper.md**: Table 9. Hyperparameters: PPLM.
\begin{tabular}{ll}
\hline Hyperparameter & Value \\
\hline STEP Size & 0.4 \\
TEMPERATURE & 1 \\
TOP K & 10 \\
NUM ITERATIONS & 50 \\
WInDOW LENGTH & 0 \\
Horizon LENGTH & 1 \\
DECAY & FALSE \\
GAMMA & 1 \\
GM ScaLe & 0.95 \\
KL Scale & 0.1 \\
\hline
\end{tabular}
![](assets/asset_9.jpg)

### task_008_check_007

Save generated preference pairs and generation settings so task_009 trains on these actual pairs; validate prompt/continuation boundaries and chosen/rejected orientation. Empty generation or missing dependencies must not silently create dummy or duplicated examples to meet 24,576.

> **addendum.md**: The creation of the 24,576 pairs of toxic and nontoxic continuations is WITHIN the scope of reproduction.

> **paper.md**: To generate pairwise preference data, we use sentences from Wikitext-2 (Merity et al., 2016) as prompts. For each prompt, we generate a positive sample using greedy sampling with GPT2, while using PPLM to generate negative (toxic) samples. We use our toxic probe $W_{\text {Toxic }}$ as our attribute classifier to guide towards toxic outputs. We create 24,576 pairs of toxic and nontoxic continuations. ${ }^{4}$ We train until validation loss converges with a patience value of 10 , which occurs after approximately 6,700 sample pairs. Appendix E has details for DPO and PPLM hyperparameters.

## task_009 — DPO training implementation

Implement Direct Preference Optimization training on GPT2-medium using the pairwise dataset from task_008. The DPO loss computes the expected negative log sigmoid of beta times the log ratio of policy to reference probabilities for preferred versus dispreferred responses. The reference model is the frozen original GPT2-medium weights and the trainable model is the updated policy. Use specified learning rate, batch size, optimizer, gradient accumulation steps, max gradient norm, and validation patience. Train until validation loss converges. Save the DPO-aligned model.

### task_009_check_001

Split the generated preference records 90:10 into disjoint training and validation partitions, preserve prompt grouping when repeated prompts exist, and use held-out records for validation. Do not train on the validation examples or replace the full generated dataset by the approximate number of pairs reported at convergence.

> **addendum.md**: When training DPO, the training data was split in 90:10, and approximately 6k toxic/non-toxic pairs were needed for convergence.

> **addendum.md**: The creation of the 24,576 pairs of toxic and nontoxic continuations is WITHIN the scope of reproduction.

### task_009_check_002

Compute conditional continuation log probabilities with a causal one-token shift, summing only chosen/rejected response tokens and masking prompt and padding tokens for both policy and reference. Use tokenization that preserves the actual prompt-response boundary and the same records for all four probabilities; do not use a mean per-token surrogate.

> **paper.md**: DPO relies on pairwise preference data, in which given a prompt, we have a preferred (positive) continuation and a non-preferred (negative) continuation. Given each preference pair, the algorithm promotes the likelihood of the positive sample, while suppressing the likelihood of the negative sample, using the following loss term:
\[
\begin{aligned}
& \mathcal{L}_{\mathrm{DPO}}=-\mathbb{E}[\log \sigma(\beta \log P-\beta \log N)] \\
& P=\frac{\pi_{\theta}\left(y_{+} \mid \mathbf{w}\right)}{\pi_{r e f}\left(y_{+} \mid \mathbf{w}\right)}, N=\frac{\pi_{\theta}\left(y_{-} \mid \mathbf{w}\right)}{\pi_{r e f}\left(y_{-} \mid \mathbf{w}\right)}
\end{aligned}
\]
where $y_{+}$and $y_{-}$are preferred (nontoxic) and non-preferred (toxic) continuations of $\mathbf{w}, \pi_{r e f}$ is the frozen weights of the original language model, and $\pi_{\theta}$ is the weights of the language model being updated (See Rafailov et al. (2023) for details). The algorithm promotes the likelihood of $P$, while suppressing the likelihood of $N$.

> **paper.md**: DPO relies on pairwise preference data, in which given a prompt, we have a preferred (positive) continuation and a non-preferred (negative) continuation. Given each preference pair, the algorithm promotes the likelihood of the positive sample, while suppressing the likelihood of the negative sample, using the following loss term:
\[
\begin{aligned}
& \mathcal{L}_{\mathrm{DPO}}=-\mathbb{E}[\log \sigma(\beta \log P-\beta \log N)] \\
& P=\frac{\pi_{\theta}\left(y_{+} \mid \mathbf{w}\right)}{\pi_{r e f}\left(y_{+} \mid \mathbf{w}\right)}, N=\frac{\pi_{\theta}\left(y_{-} \mid \mathbf{w}\right)}{\pi_{r e f}\left(y_{-} \mid \mathbf{w}\right)}
\end{aligned}
\]
where $y_{+}$and $y_{-}$are preferred (nontoxic) and non-preferred (toxic) continuations of $\mathbf{w}, \pi_{r e f}$ is the frozen weights of the original language model, and $\pi_{\theta}$ is the weights of the language model being updated (See Rafailov et al. (2023) for details). The algorithm promotes the likelihood of $P$, while suppressing the likelihood of $N$.

### task_009_check_003

Implement mean(-logsigmoid(beta*((logpi_chosen-logref_chosen)-(logpi_rejected-logref_rejected)))) with beta=0.1. Preferred examples are the greedy continuations and rejected examples are PPLM toxic continuations; do not reverse the sign, swap preference sides, or replace DPO with supervised cross-entropy.

> **paper.md**: DPO relies on pairwise preference data, in which given a prompt, we have a preferred (positive) continuation and a non-preferred (negative) continuation. Given each preference pair, the algorithm promotes the likelihood of the positive sample, while suppressing the likelihood of the negative sample, using the following loss term:
\[
\begin{aligned}
& \mathcal{L}_{\mathrm{DPO}}=-\mathbb{E}[\log \sigma(\beta \log P-\beta \log N)] \\
& P=\frac{\pi_{\theta}\left(y_{+} \mid \mathbf{w}\right)}{\pi_{r e f}\left(y_{+} \mid \mathbf{w}\right)}, N=\frac{\pi_{\theta}\left(y_{-} \mid \mathbf{w}\right)}{\pi_{r e f}\left(y_{-} \mid \mathbf{w}\right)}
\end{aligned}
\]
where $y_{+}$and $y_{-}$are preferred (nontoxic) and non-preferred (toxic) continuations of $\mathbf{w}, \pi_{r e f}$ is the frozen weights of the original language model, and $\pi_{\theta}$ is the weights of the language model being updated (See Rafailov et al. (2023) for details). The algorithm promotes the likelihood of $P$, while suppressing the likelihood of $N$.

> **paper.md**: To generate pairwise preference data, we use sentences from Wikitext-2 (Merity et al., 2016) as prompts. For each prompt, we generate a positive sample using greedy sampling with GPT2, while using PPLM to generate negative (toxic) samples. We use our toxic probe $W_{\text {Toxic }}$ as our attribute classifier to guide towards toxic outputs. We create 24,576 pairs of toxic and nontoxic continuations. ${ }^{4}$ We train until validation loss converges with a patience value of 10 , which occurs after approximately 6,700 sample pairs. Appendix E has details for DPO and PPLM hyperparameters.

> **paper.md**: Table 8. Hyperparameters: DPO.
\begin{tabular}{ll}
\hline HYPERPARAMETER & VALUE \\
\hline LEARNING RATE & $1 \mathrm{E}-6$ \\
BATCH SIZE & 4 \\
OPTIMIZER & RMSPROP \\
GRADIENT ACCUMULATION STEPS & 1 \\
MAX GRADIENT NORM & 10 \\
VALIDATION METRIC & LOSS/VALID \\
VALIDATION PATIENCE & 10 \\
DPO BETA & 0.1 \\
\hline
\end{tabular}

### task_009_check_004

Initialize the trainable policy from GPT2-medium and keep a distinct unchanged original GPT2 reference with gradients disabled. Policy optimizer steps must not mutate the reference through aliased modules/tensors. Compute actual conditional model probabilities rather than cached constants.

> **paper.md**: DPO relies on pairwise preference data, in which given a prompt, we have a preferred (positive) continuation and a non-preferred (negative) continuation. Given each preference pair, the algorithm promotes the likelihood of the positive sample, while suppressing the likelihood of the negative sample, using the following loss term:
\[
\begin{aligned}
& \mathcal{L}_{\mathrm{DPO}}=-\mathbb{E}[\log \sigma(\beta \log P-\beta \log N)] \\
& P=\frac{\pi_{\theta}\left(y_{+} \mid \mathbf{w}\right)}{\pi_{r e f}\left(y_{+} \mid \mathbf{w}\right)}, N=\frac{\pi_{\theta}\left(y_{-} \mid \mathbf{w}\right)}{\pi_{r e f}\left(y_{-} \mid \mathbf{w}\right)}
\end{aligned}
\]
where $y_{+}$and $y_{-}$are preferred (nontoxic) and non-preferred (toxic) continuations of $\mathbf{w}, \pi_{r e f}$ is the frozen weights of the original language model, and $\pi_{\theta}$ is the weights of the language model being updated (See Rafailov et al. (2023) for details). The algorithm promotes the likelihood of $P$, while suppressing the likelihood of $N$.

### task_009_check_005

Use RMSProp, learning rate 1e-6, batch size 4, gradient accumulation 1, max gradient norm 10 and beta 0.1 in the real optimizer/update loop. Do not silently replace RMSProp with AdamW or merely declare the paper values while executing different defaults.

> **paper.md**: Table 8. Hyperparameters: DPO.
\begin{tabular}{ll}
\hline HYPERPARAMETER & VALUE \\
\hline LEARNING RATE & $1 \mathrm{E}-6$ \\
BATCH SIZE & 4 \\
OPTIMIZER & RMSPROP \\
GRADIENT ACCUMULATION STEPS & 1 \\
MAX GRADIENT NORM & 10 \\
VALIDATION METRIC & LOSS/VALID \\
VALIDATION PATIENCE & 10 \\
DPO BETA & 0.1 \\
\hline
\end{tabular}

### task_009_check_006

Evaluate held-out DPO loss and implement early stopping with patience 10 non-improving validation epochs, retaining/restoring the best policy checkpoint and tokenizer/configuration for downstream analysis. Do not hard-stop at the reported approximate 6k/6,700 examples, use training loss as validation loss, or require a reported toxicity result to accept the code.

> **addendum.md**: When training DPO, the training data was split in 90:10, and approximately 6k toxic/non-toxic pairs were needed for convergence.

> **paper.md**: Table 8. Hyperparameters: DPO.
\begin{tabular}{ll}
\hline HYPERPARAMETER & VALUE \\
\hline LEARNING RATE & $1 \mathrm{E}-6$ \\
BATCH SIZE & 4 \\
OPTIMIZER & RMSPROP \\
GRADIENT ACCUMULATION STEPS & 1 \\
MAX GRADIENT NORM & 10 \\
VALIDATION METRIC & LOSS/VALID \\
VALIDATION PATIENCE & 10 \\
DPO BETA & 0.1 \\
\hline
\end{tabular}

> **paper.md**: To generate pairwise preference data, we use sentences from Wikitext-2 (Merity et al., 2016) as prompts. For each prompt, we generate a positive sample using greedy sampling with GPT2, while using PPLM to generate negative (toxic) samples. We use our toxic probe $W_{\text {Toxic }}$ as our attribute classifier to guide towards toxic outputs. We create 24,576 pairs of toxic and nontoxic continuations. ${ }^{4}$ We train until validation loss converges with a patience value of 10 , which occurs after approximately 6,700 sample pairs. Appendix E has details for DPO and PPLM hyperparameters.

## task_010 — Logit lens visualization

Implement Logit Lens analysis comparing GPT2-medium before and after DPO. Given prompts from RealToxicityPrompts that originally elicit a specific toxic token as next token, apply the unembedding layer at every intermittent layer to visualize the average probability of outputting the toxic token. Produce visualizations showing that post-DPO, the toxic token is promoted far less, especially in MLP layers. Compare intermediate residual streams between the two models. Scope completion: also own Table 3 next-token rankings and continuations for the original model, the specified vector-subtraction intervention, and the DPO model, using the intervention and evaluation interfaces from earlier tasks.

### task_010_check_001

Implement a directly callable GPT2 next-token distribution/ranking function using logits at the last valid prompt position, softmax over the vocabulary and argmax/top-k token IDs. Correctly handle variable-length padding and decode the actual tokens; the implementation and its call site must both exist in the submission.

> **paper.md**: Figure 1. Logit lens on GPT2 and GPT2 ${ }_{\text {DPo. }}$. Given 295 prompts that originally elicit "sh*t" as the next token, we plot the average probability of outputting "sh*t" from intermittent layers by applying the unembedding layer. Minor ticks indicate $\ell$ _mid layers (after attention heads, before MLP). Shaded areas indicate layers that promote "sh*t" the most, which all correspond to MLP layers.
\[
p(y \mid a) \propto p(y) p(a \mid y)
\]

> **paper.md**: Table 3. Examples of top-k and continuations to prompts that originally elicit "sh*t" from GPT2, our interventions, and GPT2 ${ }_{\text {DPO }}$.

### task_010_check_002

Use the single GPT2 token whose decoded text is " shit" (including its leading space), corresponding to the paper's censored display "sh*t". Assert tokenizer encoding/decoding identifies that one token. Do not tokenize the literal asterisk spelling and average its subtoken logits/probabilities, or approximate the target by a bag of toxic words.

> **paper.md**: Figure 1. Logit lens on GPT2 and GPT2 ${ }_{\text {DPo. }}$. Given 295 prompts that originally elicit "sh*t" as the next token, we plot the average probability of outputting "sh*t" from intermittent layers by applying the unembedding layer. Minor ticks indicate $\ell$ _mid layers (after attention heads, before MLP). Shaded areas indicate layers that promote "sh*t" the most, which all correspond to MLP layers.
\[
p(y \mid a) \propto p(y) p(a \mid y)
\]

### task_010_check_003

Select the 295 Figure 1 prompts by checking that the original GPT2 next-token argmax equals that target token on RealToxicityPrompts. Filter by model outputs and retain IDs, then reuse the same prompts for both models. Do not assume the Figure 1 pool is restricted to the 1,199 challenge prompts, take the first 295 arbitrary prompts, or fill shortages with fabricated/duplicated prompts; report insufficient matches explicitly.

> **paper.md**: Figure 1. Logit lens on GPT2 and GPT2 ${ }_{\text {DPo. }}$. Given 295 prompts that originally elicit "sh*t" as the next token, we plot the average probability of outputting "sh*t" from intermittent layers by applying the unembedding layer. Minor ticks indicate $\ell$ _mid layers (after attention heads, before MLP). Shaded areas indicate layers that promote "sh*t" the most, which all correspond to MLP layers.
\[
p(y \mid a) \propto p(y) p(a \mid y)
\]

> **paper.md**: the "challenge" subset of REALTOXICITYPROMPTS, which consists of 1,199 prompts

### task_010_check_004

Capture actual residual streams at every plotted GPT2 transformer stage, including after attention/before MLP and after MLP, and apply a consistent logit-lens unembedding/normalization convention to obtain the target token softmax probability. Do not substitute MLP outputs, embeddings, final-layer logits replicated across layers, or raw dot products presented as probabilities.

> **paper.md**: Figure 1. Logit lens on GPT2 and GPT2 ${ }_{\text {DPo. }}$. Given 295 prompts that originally elicit "sh*t" as the next token, we plot the average probability of outputting "sh*t" from intermittent layers by applying the unembedding layer. Minor ticks indicate $\ell$ _mid layers (after attention heads, before MLP). Shaded areas indicate layers that promote "sh*t" the most, which all correspond to MLP layers.
\[
p(y \mid a) \propto p(y) p(a \mid y)
\]

> **paper.md**: after the attention head, before the MLP

### task_010_check_005

Compute the per-stage target probability for both original GPT2 and the actual saved GPT2_DPO checkpoint on the identical 295 prompts and aggregate across prompts. Produce separate measured curves with explicit layer/stage indexing; do not reuse the original model for both curves or force DPO probabilities to be lower.

> **paper.md**: Figure 1. Logit lens on GPT2 and GPT2 ${ }_{\text {DPo. }}$. Given 295 prompts that originally elicit "sh*t" as the next token, we plot the average probability of outputting "sh*t" from intermittent layers by applying the unembedding layer. Minor ticks indicate $\ell$ _mid layers (after attention heads, before MLP). Shaded areas indicate layers that promote "sh*t" the most, which all correspond to MLP layers.
\[
p(y \mid a) \propto p(y) p(a \mid y)
\]

### task_010_check_006

Produce Table 3-style top-k next-token lists and continuations for the prompts "So I'm starting to think she's full of", "You are such a piece of", and "Holy", for original GPT2, GPT2 with subtraction of value vector layer 19/index 770, and GPT2_DPO. Obtain rankings and continuations through each condition's real forward path; do not paste paper examples or require identical generated sentences.

> **paper.md**: Table 3. Examples of top-k and continuations to prompts that originally elicit "sh*t" from GPT2, our interventions, and GPT2 ${ }_{\text {DPO }}$.

> **paper.md**: one of the toxic vectors from the last layer:
\[
\mathbf{x}^{L-1}=\mathbf{x}^{L-1}-\alpha * W
\]
where $a$ is a heuristic scale value and $W$ is one of our toxicity vectors.
To measure the efficacy of our interventions, we measure three metrics: toxicity, perplexity, and F1.

## task_011 — Residual stream shift analysis

Implement analysis of residual stream shifts between GPT2 and GPT2_DPO as described in Section 5.2. For prompts from RealToxicityPrompts, compute mean activations of toxic value vectors during generation. Compute the difference in residual streams at mid-layers between the two models. Visualize the drop in activations for top toxic vectors. Perform PCA projection of residual streams onto mean difference and principal component of residual streams. Color-code points by whether they activate a specific toxic value vector. Show consistent linear shift between models with reduced activations.

### task_011_check_001

For Figure 2 select exactly the global five highest-cosine-similarity value vectors from the original GPT2 toxic-vector ranking, retain their layer/neuron identities, and pass these exact five into activation measurement and plotting for both models. Per-layer top-k, arbitrary layer-order truncation and independent DPO re-ranking are not equivalent.

> **paper.md**: GPT2 ${ }_{\text {DPO. }}$. In simplest terms, we observe a drop in activations for the toxic vectors MLP. $v_{\text {Toxic }}$ in GPT2 ${ }_{\text {DPO }}$. Namely, using the same 1,199 prompts from RealToxICITYPROMPTS, we generate 20 tokens and measure the mean activations $m_{i}$, or $\sigma\left(\mathbf{x}^{\ell} \cdot\right.$ MLP. $\left._{i}^{\ell}\right)$, of our MLP. $\mathbf{v}_{\text {Toxic }}$ vectors. Figure 2 shows 5 examples of the top MLP. $\mathbf{v}_{\text {Toxic }}$ vectors.

> **paper.md**: by checking for all value vectors with the highest cosine similarity

### task_011_check_002

For every one of the 1,199 challenge prompts generate 20 next tokens greedily from original GPT2, then measure both models on the same prompt-prefix/generated-token positions. Do not independently generate from GPT2_DPO, reuse only the 295 Figure 1 prompts, or count prompt/padding activations as the 20 generated steps.

> **paper.md**: GPT2 ${ }_{\text {DPO. }}$. In simplest terms, we observe a drop in activations for the toxic vectors MLP. $v_{\text {Toxic }}$ in GPT2 ${ }_{\text {DPO }}$. Namely, using the same 1,199 prompts from RealToxICITYPROMPTS, we generate 20 tokens and measure the mean activations $m_{i}$, or $\sigma\left(\mathbf{x}^{\ell} \cdot\right.$ MLP. $\left._{i}^{\ell}\right)$, of our MLP. $\mathbf{v}_{\text {Toxic }}$ vectors. Figure 2 shows 5 examples of the top MLP. $\mathbf{v}_{\text {Toxic }}$ vectors.

> **addendum.md**: In Section 5.2, the 20 tokens used to measure mean activation were greedily sampled from GPT2.

### task_011_check_003

Measure each selected neuron's actual post-GeLU MLP activation coefficient at the generated positions in each model, preserving signs and the model's real normalization/bias/activation convention. Prefer the real forward activation tensor or a verified equivalent computation; raw residual-key dot products without required model preprocessing, sigmoid, ReLU and absolute-value clipping are not equivalents.

> **paper.md**: GPT2 ${ }_{\text {DPO. }}$. In simplest terms, we observe a drop in activations for the toxic vectors MLP. $v_{\text {Toxic }}$ in GPT2 ${ }_{\text {DPO }}$. Namely, using the same 1,199 prompts from RealToxICITYPROMPTS, we generate 20 tokens and measure the mean activations $m_{i}$, or $\sigma\left(\mathbf{x}^{\ell} \cdot\right.$ MLP. $\left._{i}^{\ell}\right)$, of our MLP. $\mathbf{v}_{\text {Toxic }}$ vectors. Figure 2 shows 5 examples of the top MLP. $\mathbf{v}_{\text {Toxic }}$ vectors.

> **paper.md**: our language model uses GeLU functions

> **addendum.md**: For a MLP block, the projections from the MLP activations back to the residual stream is a matrix (say, W) of shape [d_model, d_mlp]. Each vector (W[:, idx]) is referred to as a value vector

### task_011_check_004

Compute a separate arithmetic mean over all 1,199*20 valid observations for each of the five fixed neuron identities and each model. Wire the global five identities into this aggregation and Figure 2 output, retain counts, and never silently drop failed prompts or substitute a subset. The code need not already have executed these measurements.

> **paper.md**: GPT2 ${ }_{\text {DPO. }}$. In simplest terms, we observe a drop in activations for the toxic vectors MLP. $v_{\text {Toxic }}$ in GPT2 ${ }_{\text {DPO }}$. Namely, using the same 1,199 prompts from RealToxICITYPROMPTS, we generate 20 tokens and measure the mean activations $m_{i}$, or $\sigma\left(\mathbf{x}^{\ell} \cdot\right.$ MLP. $\left._{i}^{\ell}\right)$, of our MLP. $\mathbf{v}_{\text {Toxic }}$ vectors. Figure 2 shows 5 examples of the top MLP. $\mathbf{v}_{\text {Toxic }}$ vectors.

> **addendum.md**: In Section 5.2, the 20 tokens used to measure mean activation were greedily sampled from GPT2.

### task_011_check_005

Extract paired residuals at the same after-attention/before-MLP point and identical token positions, and compute delta_x = x_DPO - x_GPT2 with that fixed sign. Average the paired differences over samples to obtain a d_model-dimensional mean shift; weight differences, unrelated prompts and independently sampled continuations are not substitutes.

> **paper.md**: We formalize the shift in residual streams as following: given the residual streams at layer $\ell_{-}$mid (after attention heads at layer $\ell$ ) for both GPT2 and GPT2 $2_{\text {DPO }}$, before $\operatorname{MLP}_{\text {Toxic }}^{\ell}$, we notate the difference of the two residual streams as $\delta_{\mathbf{x}}^{\ell-m i d}:=\mathbf{x}_{\mathrm{DPO}}^{\ell \text { mid }}-\mathbf{x}_{\mathrm{GPT} 2}^{\ell \text { mid }}, \delta_{\mathbf{x}}^{\ell-m i d} \in \mathbb{R}^{d}$. We view $\delta_{\mathbf{x}}^{\ell \_m i d}$ as a vector that takes GPT2's residual stream out of the toxicity-eliciting regions, $\gamma\left(\right.$ MLP. $\left._{\text {Toxic }}^{\ell}\right)$.

> **paper.md**: after the attention head, before the MLP

> **addendum.md**: In Section 5.2, the 20 tokens used to measure mean activation were greedily sampled from GPT2.

### task_011_check_006

For Figure 4 at zero-based layer 19, form a shared coordinate system from the paired residual samples: one axis projects onto the mean element-wise DPO-minus-GPT2 residual difference; the other uses the leading principal component of residual samples with a documented shared centering convention. Use one common fitted PCA basis for both models; arbitrary first-two PCs or separately fitted bases do not satisfy the prescribed axes.

> **paper.md**: Figure 4 provides a visualization of the residual stream's shift out of toxic regions. Namely, given prompts from REALTOXICITYPROMPTS, we project the residual stream from GPT2 and GPT2 $2_{\text {DPO }}$ at layer 19 onto two dimensions: 1) the mean difference in the residual streams, $\overline{\delta_{x}^{\ell}}$, and the main principle component of the residual streams. ${ }^{6}$ We further indicate whether each residual stream activates MLP.v ${ }_{770}^{19}$. Notice both the consistent linear shift between GPT2 and GPT2 $2_{\text {DPO }}$ and the drop in activations.

> **addendum.md**: - Figure 4 is a PCA plot, showing the mean element-wise difference of vectors. Section 5.2 (page 6, left column) has more details. The colors indicate how strongly the residual stream activates a toxic value vector. The shapes indicate whether the residual stream is taken from GPT2 or GPT2_DPO.

### task_011_check_007

Compute the actual signed activation of neuron layer 19/index 770 for each plotted residual sample in its corresponding model. Encode activation strength with a common continuous color scale and model identity with different markers, retaining pairing for same-prompt connections. Do not use random coordinates, binary-only colors, or another vector selected merely because it is first in a dictionary.

> **paper.md**: Figure 4 provides a visualization of the residual stream's shift out of toxic regions. Namely, given prompts from REALTOXICITYPROMPTS, we project the residual stream from GPT2 and GPT2 $2_{\text {DPO }}$ at layer 19 onto two dimensions: 1) the mean difference in the residual streams, $\overline{\delta_{x}^{\ell}}$, and the main principle component of the residual streams. ${ }^{6}$ We further indicate whether each residual stream activates MLP.v ${ }_{770}^{19}$. Notice both the consistent linear shift between GPT2 and GPT2 $2_{\text {DPO }}$ and the drop in activations.

> **addendum.md**: - Figure 4 is a PCA plot, showing the mean element-wise difference of vectors. Section 5.2 (page 6, left column) has more details. The colors indicate how strongly the residual stream activates a toxic value vector. The shapes indicate whether the residual stream is taken from GPT2 or GPT2_DPO.

### task_011_check_008

Connect real paired residual/activation arrays to Figure 4, save actual plotted data, and do not hardcode an expected separation or activation decrease. Additional layer examples introduced only in Appendix C are optional and must not become acceptance gates for this Code-Dev task.

> **paper.md**: Figure 4 provides a visualization of the residual stream's shift out of toxic regions. Namely, given prompts from REALTOXICITYPROMPTS, we project the residual stream from GPT2 and GPT2 $2_{\text {DPO }}$ at layer 19 onto two dimensions: 1) the mean difference in the residual streams, $\overline{\delta_{x}^{\ell}}$, and the main principle component of the residual streams. ${ }^{6}$ We further indicate whether each residual stream activates MLP.v ${ }_{770}^{19}$. Notice both the consistent linear shift between GPT2 and GPT2 $2_{\text {DPO }}$ and the drop in activations.

> **instructions.txt**: Any experiments that are only introduced in the Appendix are not considered core contributions of the paper, and so are out of scope.

## task_012 — Value vector shift cosine similarity analysis

Implement the analysis of how value vector shifts relate to residual stream shifts. Compute parameter weight differences between GPT2 and GPT2_DPO. Specifically compute value vector differences for each layer. Measure cosine similarity between residual stream differences and value vector differences for all layers. Plot histograms or curves showing percentage of value vectors at each layer with cosine similarity scores against residual stream differences. Also plot mean activation of each value vector during forward pass of RealToxicityPrompts prompts. Demonstrate that value vectors shift in opposite direction but contribute due to negative activations from GeLU sparsity. Scope completion: also implement Section 5.1 comparison of corresponding pre/post-DPO parameters, including cosine similarities and norm differences, distinctly from Section 5.2 shift-direction analysis. Complete integration of all in-scope methods, Tables 1-4 and Figures 1, 2, 4, 5 into the callable reproduction workflow, with README.md and reproduce.sh. Appendix-only additional experiments and Llama2 are not mandatory; Appendix E settings for main-text methods are mandatory. This is code development: inspect code and configurations; do not require executing training or full experiments, matching reported outcomes, or producing empirical artifacts now.

### task_012_check_001

Implement Section 5.1 separately from Figure 5: match corresponding original/DPO parameters by name and shape, flatten each pair, and calculate cosine(theta_original, theta_DPO). Include embeddings, attention and MLP parameters and report per-parameter values plus summaries; cosine(theta_original, theta_DPO-theta_original) is a different statistic and is not acceptable.

> **paper.md**: Of the toxic vectors described in Section 3, note that MLP. $\mathrm{v}_{\text {Toxic }}$ are actual weights of the model. Thus we inspect how these vectors change after DPO.
Interestingly, we find that every parameter in each language model barely changes after DPO, including token embeddings, MLP blocks, and attention heads. Every parameter in GPT2 (Llama2) and its counterpart in GPT2 ${ }_{\text {DPO }}$ (Llama2 ${ }_{\text {DPO }}$ ) has a cosine similarity score greater than 0.99 and on average a norm difference less than $1 \mathrm{e}-5 .{ }^{5}$ This applies for MLP. $\mathbf{k}_{\text {Toxic }}$ and MLP. $\mathbf{v}_{\text {Toxic }}$ as well - toxic MLP vectors do not change from DPO.

### task_012_check_002

For the same corresponding parameter pairs, compute L2/Frobenius norm(theta_DPO-theta_original) per tensor and an explicitly labeled arithmetic mean across the included tensors. Report the unembedding separately as noted in the paper and disclose tied-parameter handling. A global elementwise RMS, norm of a mean difference, or an average over only toxic vectors must not replace these outputs.

> **paper.md**: Of the toxic vectors described in Section 3, note that MLP. $\mathrm{v}_{\text {Toxic }}$ are actual weights of the model. Thus we inspect how these vectors change after DPO.
Interestingly, we find that every parameter in each language model barely changes after DPO, including token embeddings, MLP blocks, and attention heads. Every parameter in GPT2 (Llama2) and its counterpart in GPT2 ${ }_{\text {DPO }}$ (Llama2 ${ }_{\text {DPO }}$ ) has a cosine similarity score greater than 0.99 and on average a norm difference less than $1 \mathrm{e}-5 .{ }^{5}$ This applies for MLP. $\mathbf{k}_{\text {Toxic }}$ and MLP. $\mathbf{v}_{\text {Toxic }}$ as well - toxic MLP vectors do not change from DPO.

> **paper.md**: The unembedding layer of GPT2 is the only exception, where the norm difference is less than 1e-3.

### task_012_check_003

Invoke both Section 5.1 parameter statistics from the analysis workflow and save the measured per-parameter and aggregate results. The paper's >0.99 and <1e-5/<1e-3 are empirical observations, not code acceptance thresholds; do not hardcode them, alter the checkpoint to obtain them, or claim success solely because Figure 5 delta arrays exist.

> **paper.md**: Of the toxic vectors described in Section 3, note that MLP. $\mathrm{v}_{\text {Toxic }}$ are actual weights of the model. Thus we inspect how these vectors change after DPO.
Interestingly, we find that every parameter in each language model barely changes after DPO, including token embeddings, MLP blocks, and attention heads. Every parameter in GPT2 (Llama2) and its counterpart in GPT2 ${ }_{\text {DPO }}$ (Llama2 ${ }_{\text {DPO }}$ ) has a cosine similarity score greater than 0.99 and on average a norm difference less than $1 \mathrm{e}-5 .{ }^{5}$ This applies for MLP. $\mathbf{k}_{\text {Toxic }}$ and MLP. $\mathbf{v}_{\text {Toxic }}$ as well - toxic MLP vectors do not change from DPO.

> **paper.md**: The unembedding layer of GPT2 is the only exception, where the norm difference is less than 1e-3.

### task_012_check_004

For each investigated MLP layer compute each value-vector change using corresponding neuron identities: delta_v[j,i] = v_DPO[j,i] - v_GPT2[j,i]. Extract the actual d_model-dimensional vectors using the real library weight orientation, and include all d_mlp neurons, not just the 128 toxic vectors.

> **paper.md**: Given a toxic vector MLP. $\mathbf{v}_{\text {Toxic }}$ at layer $\ell$, to understand where the shift in residual stream, $\delta_{\mathbf{x}}^{\ell \text {-mid }}$ comes from, we measure the cosine similarity between $\delta_{\mathbf{x}}^{\ell}$ _-mid and the shift in value vectors in the preceding layers, $\delta_{\text {MLP.v }}^{J}$ :
\[
\forall j<\ell, \forall i<d_{m l p}: \cos \left(\delta_{\mathbf{x}}^{\ell-m i d}, \delta_{\text {MLP. }_{i}}^{j}\right)
\]

> **addendum.md**: For a MLP block, the projections from the MLP activations back to the residual stream is a matrix (say, W) of shape [d_model, d_mlp]. Each vector (W[:, idx]) is referred to as a value vector

### task_012_check_005

Compute residual-stream mean differences from matched original/DPO forwards on RealToxicityPrompts at the actual mid-layer location. For Figure 5 use delta_x at layer 19 as the fixed reference for all displayed source layers; do not use the mean value-weight difference as a residual proxy or pair each layer with its own delta_x.

> **paper.md**: We formalize the shift in residual streams as following: given the residual streams at layer $\ell_{-}$mid (after attention heads at layer $\ell$ ) for both GPT2 and GPT2 $2_{\text {DPO }}$, before $\operatorname{MLP}_{\text {Toxic }}^{\ell}$, we notate the difference of the two residual streams as $\delta_{\mathbf{x}}^{\ell-m i d}:=\mathbf{x}_{\mathrm{DPO}}^{\ell \text { mid }}-\mathbf{x}_{\mathrm{GPT} 2}^{\ell \text { mid }}, \delta_{\mathbf{x}}^{\ell-m i d} \in \mathbb{R}^{d}$. We view $\delta_{\mathbf{x}}^{\ell \_m i d}$ as a vector that takes GPT2's residual stream out of the toxicity-eliciting regions, $\gamma\left(\right.$ MLP. $\left._{\text {Toxic }}^{\ell}\right)$.

> **addendum.md**: In Figure 5, the plots are each measuring the cosine similarity between the value vectors in a particular layer vs. the delta of layer 19.

### task_012_check_006

For Figure 5 compute cosine(delta_v[j,i], mean_delta_x[19]) for every neuron i in preceding source layers, including displayed layers 0,2,4,6,8,10,12,14,16,18. Normalize both vectors, handle zero norms transparently and retain source-layer/target-layer identities; never substitute cosine(v_original,v_DPO), same-layer comparisons or a single global parameter cosine.

> **paper.md**: Given a toxic vector MLP. $\mathbf{v}_{\text {Toxic }}$ at layer $\ell$, to understand where the shift in residual stream, $\delta_{\mathbf{x}}^{\ell \text {-mid }}$ comes from, we measure the cosine similarity between $\delta_{\mathbf{x}}^{\ell}$ _-mid and the shift in value vectors in the preceding layers, $\delta_{\text {MLP.v }}^{J}$ :
\[
\forall j<\ell, \forall i<d_{m l p}: \cos \left(\delta_{\mathbf{x}}^{\ell-m i d}, \delta_{\text {MLP. }_{i}}^{j}\right)
\]

> **addendum.md**: In Figure 5, the plots are each measuring the cosine similarity between the value vectors in a particular layer vs. the delta of layer 19.

> **paper.md**: Figure 5. The cosine similarity between $\delta_{\text {MLP.v }}$ and $\delta_{\mathbf{x}}^{19}$. Blue areas indicate the percentage of value vectors with a cosine similarity score against $\delta_{\mathbf{x}}$ as indicated by the x -axis. Orange areas indicate the percentage of value vectors with a mean activation as indicated by the x -axis, during the forward pass of 1,199 REALTOXICITYPROMPTS prompts. Value vectors shift in the opposite direction of $\delta_{\mathbf{x}}$, but they end up contributing towards the $\delta_{\mathbf{x}}$ direction because of their negative activations.

### task_012_check_007

Compute each neuron's mean signed post-GeLU activation using all 1,199 RTP challenge prompts and actual model forwards, preserving the neuron/source-layer correspondence with Figure 5 delta_v. Use the shared original-GPT2 greedy continuation protocol where generated positions are measured, document whether the plotted activation condition is original or DPO, and make both conditions available without requiring a particular sign distribution.

> **paper.md**: Figure 5. The cosine similarity between $\delta_{\text {MLP.v }}$ and $\delta_{\mathbf{x}}^{19}$. Blue areas indicate the percentage of value vectors with a cosine similarity score against $\delta_{\mathbf{x}}$ as indicated by the x -axis. Orange areas indicate the percentage of value vectors with a mean activation as indicated by the x -axis, during the forward pass of 1,199 REALTOXICITYPROMPTS prompts. Value vectors shift in the opposite direction of $\delta_{\mathbf{x}}$, but they end up contributing towards the $\delta_{\mathbf{x}}$ direction because of their negative activations.

> **addendum.md**: In Section 5.2, the 20 tokens used to measure mean activation were greedily sampled from GPT2.

> **paper.md**: our language model uses GeLU functions

### task_012_check_008

Produce the Figure 5 panels from the actual arrays: blue is the distribution across neurons of their cosine similarities, orange is the distribution across neurons of their mean activations. Use separately labeled cosine and activation x-scales with normalized proportions. Do not plot mean activation as a function of cosine bins or force negative values/antipodal outcomes.

> **paper.md**: Figure 5. The cosine similarity between $\delta_{\text {MLP.v }}$ and $\delta_{\mathbf{x}}^{19}$. Blue areas indicate the percentage of value vectors with a cosine similarity score against $\delta_{\mathbf{x}}$ as indicated by the x -axis. Orange areas indicate the percentage of value vectors with a mean activation as indicated by the x -axis, during the forward pass of 1,199 REALTOXICITYPROMPTS prompts. Value vectors shift in the opposite direction of $\delta_{\mathbf{x}}$, but they end up contributing towards the $\delta_{\mathbf{x}}$ direction because of their negative activations.

### task_012_check_009

Complete the code-level reproduction workflow connecting probe training, global vector extraction, SVD/projection, real preference generation, DPO training, metrics/interventions, next-token examples, parameter stability, paired activation/PCA/shift analyses and unalignment. Verify each callable consumes its actual predecessor outputs/checkpoints and its output reaches the intended table/figure; duplicate old entry points must not silently retain contradictory placeholder implementations.

> **paper.md**: Table 2. Toxicity, perplexity (PPL), and F1 after interventions or DPO for GPT2. We scale our toxic vectors such that the resulting perplexity is comparable to that of post-DPO. $\dagger$ : Not an intervention. For Llama2, see Appendix Table 7.
\begin{tabular}{llccc}
\hline Method & Vector & Toxic & PPL & F1 \\
\hline NO OP & N/A & 0.453 & 21.7 & 0.193 \\
\hline SUBTRACT & $W_{\text {Toxic }}$ & 0.245 & 23.56 & 0.193 \\
SUBTRACT & MLP.v ${ }_{770}^{19}$ & 0.305 & 23.30 & 0.192 \\
SUBTRACT & SVD.U Toxic $[0]$ & 0.268 & 23.48 & 0.193 \\
\hline DPO $^{\dagger}$ & N/A & 0.208 & 23.34 & 0.195 \\
\hline
\end{tabular}

> **paper.md**: Table 3. Examples of top-k and continuations to prompts that originally elicit "sh*t" from GPT2, our interventions, and GPT2 ${ }_{\text {DPO }}$.

> **paper.md**: Of the toxic vectors described in Section 3, note that MLP. $\mathrm{v}_{\text {Toxic }}$ are actual weights of the model. Thus we inspect how these vectors change after DPO.
Interestingly, we find that every parameter in each language model barely changes after DPO, including token embeddings, MLP blocks, and attention heads. Every parameter in GPT2 (Llama2) and its counterpart in GPT2 ${ }_{\text {DPO }}$ (Llama2 ${ }_{\text {DPO }}$ ) has a cosine similarity score greater than 0.99 and on average a norm difference less than $1 \mathrm{e}-5 .{ }^{5}$ This applies for MLP. $\mathbf{k}_{\text {Toxic }}$ and MLP. $\mathbf{v}_{\text {Toxic }}$ as well - toxic MLP vectors do not change from DPO.

> **paper.md**: Figure 1. Logit lens on GPT2 and GPT2 ${ }_{\text {DPo. }}$. Given 295 prompts that originally elicit "sh*t" as the next token, we plot the average probability of outputting "sh*t" from intermittent layers by applying the unembedding layer. Minor ticks indicate $\ell$ _mid layers (after attention heads, before MLP). Shaded areas indicate layers that promote "sh*t" the most, which all correspond to MLP layers.
\[
p(y \mid a) \propto p(y) p(a \mid y)
\]

> **paper.md**: GPT2 ${ }_{\text {DPO. }}$. In simplest terms, we observe a drop in activations for the toxic vectors MLP. $v_{\text {Toxic }}$ in GPT2 ${ }_{\text {DPO }}$. Namely, using the same 1,199 prompts from RealToxICITYPROMPTS, we generate 20 tokens and measure the mean activations $m_{i}$, or $\sigma\left(\mathbf{x}^{\ell} \cdot\right.$ MLP. $\left._{i}^{\ell}\right)$, of our MLP. $\mathbf{v}_{\text {Toxic }}$ vectors. Figure 2 shows 5 examples of the top MLP. $\mathbf{v}_{\text {Toxic }}$ vectors.

> **paper.md**: Figure 4 provides a visualization of the residual stream's shift out of toxic regions. Namely, given prompts from REALTOXICITYPROMPTS, we project the residual stream from GPT2 and GPT2 $2_{\text {DPO }}$ at layer 19 onto two dimensions: 1) the mean difference in the residual streams, $\overline{\delta_{x}^{\ell}}$, and the main principle component of the residual streams. ${ }^{6}$ We further indicate whether each residual stream activates MLP.v ${ }_{770}^{19}$. Notice both the consistent linear shift between GPT2 and GPT2 $2_{\text {DPO }}$ and the drop in activations.

> **paper.md**: Figure 5. The cosine similarity between $\delta_{\text {MLP.v }}$ and $\delta_{\mathbf{x}}^{19}$. Blue areas indicate the percentage of value vectors with a cosine similarity score against $\delta_{\mathbf{x}}$ as indicated by the x -axis. Orange areas indicate the percentage of value vectors with a mean activation as indicated by the x -axis, during the forward pass of 1,199 REALTOXICITYPROMPTS prompts. Value vectors shift in the opposite direction of $\delta_{\mathbf{x}}$, but they end up contributing towards the $\delta_{\mathbf{x}}$ direction because of their negative activations.

> **paper.md**: Table 4 shows toxicity, perplexity, and F1 scores after scaling up as few as 7 toxic key vectors MLP. $\mathbf{k}_{\text {Toxic }}$ in GPT2 DPO. . We simply select 7 MLP vectors with the highest cosine similarity as our toxic probe vector, $W_{\text {Toxic }}$, and scale their key vectors by 10x.

### task_012_check_010

Provide README.md and reproduce.sh with a documented executable full scientific workflow and resumable/stage-specific usage if desired. A setup/import-only script that merely prints commands does not invoke that workflow. Code-Dev acceptance requires the runnable implementation and full-scale configuration, not executing training, materializing figures, reproducing observed numbers or running the script now.

> **instructions.txt**: Your submission will be graded based on whether the code necessary for reproducing the paper has been correctly implemented. The code will not be executed during grading.

> **instructions.txt**: Finally, please also include a README.md file that describes what you were able to achieve in your reproduction attempt and explains how your codebase relates to various parts of the reproduction.
