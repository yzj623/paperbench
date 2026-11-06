"""Offline, deterministic authoring/validation; prints artifacts, never runs an agent.

Human-authored diagnostic checklist. Rubric access is intentional and disclosed in
AUDIT.md. Sources embedded in the agent-facing plan remain public-document quotes.
Run from anywhere with the ra-skill-list Python environment; stdout is JSON containing
the plan, coverage audit and readable checklist. This script does not write files.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
LAB = ROOT.parent
PAPER = LAB / 'PaperBench-Official/project/paperbench/data/papers/mechanistic-understanding'
BASE = LAB / 'PaperBench-Experiments/mechanistic-understanding/2026-09-13T04-33-18Z/agent-run/research_agent/plan.json'
sys.path.insert(0, str(ROOT / 'src'))
from research_agent.plan import parse_plan

DOCS = {name: (PAPER / name).read_text() for name in ('paper.md', 'addendum.md')}
INSTRUCTIONS = LAB / 'PaperBench-Official/project/paperbench/paperbench/instructions/code_only_instructions.txt'
DOCS['instructions.txt'] = INSTRUCTIONS.read_text()
SOURCES = {}


def source(key, document, quote):
    assert quote in DOCS[document], (key, 'quote not present verbatim')
    SOURCES[key] = {'document': document, 'quote': quote}


def paragraph(key, document, marker):
    matches = [p for p in DOCS[document].split('\n\n') if marker in p]
    assert len(matches) == 1, (key, len(matches))
    source(key, document, matches[0].strip())


source('scope', 'addendum.md', 'Results related to Llama2 have been deemed out of scope because access to get the llama2 model depends on getting permission.')
source('jigsaw', 'addendum.md', 'For accessing the Jigsaw toxic comment classification dataset, reproductions should use https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge rather than the Kaggle hosted dataset.')
source('scorer', 'addendum.md', 'For measuring toxicity, reproductions should use https://huggingface.co/unitary/unbiased-toxic-roberta instead of Perspective API.')
source('split', 'paper.md', 'We use a 90:10 split for training and validation.')
source('probe_input', 'paper.md', 'on the residual stream in the last layer, averaged across all timesteps')
source('probe_train', 'paper.md', 'We start by first training a linear probe model on a binary toxicity classification task.')
source('probe_shape', 'addendum.md', 'in the probe model, W_toxic is a matrix of shape [d_model, 2], where W_toxic[:, 0] is for non-toxic and W_toxic[:, 1] is for toxic.')
source('rank', 'paper.md', 'by checking for all value vectors with the highest cosine similarity')
source('dot', 'addendum.md', 'In table 1, "top tokens" refers to tokens that have the highest dot-products with a specified toxic vector')
source('svd', 'addendum.md', 'performing SVD on the _transpose_ of the $N \\times d$ matrix, i.e. on a $d \\times N$ matrix, hence obtaining $d$-dimensional singular value vectors from the $U$ matrix.')
source('n128', 'paper.md', 'After extracting a set of N $(=128)')
source('idx', 'addendum.md', 'For a MLP block, the projections from the MLP activations back to the residual stream is a matrix (say, W) of shape [d_model, d_mlp]. Each vector (W[:, idx]) is referred to as a value vector')
source('mid', 'paper.md', 'after the attention head, before the MLP')
source('rtp', 'paper.md', 'the "challenge" subset of REALTOXICITYPROMPTS, which consists of 1,199 prompts')
source('ppl', 'paper.md', 'measure perplexity on the Wikitext-2 dataset')
paragraph('f1', 'paper.md', 'F1. In addition to perplexity')
source('intervene', 'paper.md', 'we intervene during generation to suppress toxic outputs using each vector.')
paragraph('subtract', 'paper.md', r'\mathbf{x}^{L-1}=\mathbf{x}^{L-1}-\alpha * W')
source('alpha', 'paper.md', 'resulting perplexity is similar to that of our post-DPO model.')
paragraph('table2', 'paper.md', 'Table 2. Toxicity, perplexity')
paragraph('table3', 'paper.md', 'Table 3. Examples of top-k')
paragraph('pairs', 'paper.md', 'To generate pairwise preference data, we use sentences')
source('pplm', 'paper.md', 'During generation, PPLM uses the attribute classifier to compute the gradients that increases the likelihood of the language model\'s output to contain the desired attribute')
source('pair_scope', 'addendum.md', 'The creation of the 24,576 pairs of toxic and nontoxic continuations is WITHIN the scope of reproduction.')
source('dpo_split', 'addendum.md', 'When training DPO, the training data was split in 90:10, and approximately 6k toxic/non-toxic pairs were needed for convergence.')
paragraph('dpo_loss', 'paper.md', r'& \mathcal{L}_{\mathrm{DPO}}=')
paragraph('dpo_roles', 'paper.md', 'is the frozen weights of the original language model')
paragraph('dpo_hp', 'paper.md', 'VALIDATION PATIENCE & 10')
paragraph('pplm_hp', 'paper.md', 'GM ScaLe & 0.95')
paragraph('fig1', 'paper.md', 'Figure 1. Logit lens')
paragraph('params', 'paper.md', 'Interestingly, we find that every parameter')
source('unembed', 'paper.md', 'The unembedding layer of GPT2 is the only exception, where the norm difference is less than 1e-3.')
paragraph('fig2', 'paper.md', 'Namely, using the same 1,199 prompts')
source('greedy', 'addendum.md', 'In Section 5.2, the 20 tokens used to measure mean activation were greedily sampled from GPT2.')
paragraph('delta', 'paper.md', 'We formalize the shift in residual streams')
paragraph('fig4', 'paper.md', 'Figure 4 provides a visualization')
paragraph('pca', 'addendum.md', 'Figure 4 is a PCA plot')
paragraph('fig5', 'paper.md', 'Figure 5. The cosine similarity')
source('ref19', 'addendum.md', 'In Figure 5, the plots are each measuring the cosine similarity between the value vectors in a particular layer vs. the delta of layer 19.')
paragraph('preceding', 'paper.md', 'Given a toxic vector')
source('gelu', 'paper.md', 'our language model uses GeLU functions')
paragraph('unalign', 'paper.md', 'Table 4 shows toxicity, perplexity, and F1 scores')
paragraph('appendix_c', 'paper.md', 'Figure 7. Shift in residual streams')
paragraph('appendix_d', 'paper.md', 'Figure 8. Shift in residual streams')

# Keep source evidence precise: markdown paragraphs sometimes combine unrelated
# captions, equations and the next section. These slices remain exact substrings.
def excerpt(key, start, end=None):
    old = SOURCES[key]
    text = old['quote']
    a = text.index(start)
    b = text.index(end, a) if end else len(text)
    source(key, old['document'], text[a:b].strip())


excerpt('subtract', 'one of the toxic vectors from the last layer:')
excerpt('table3', 'Table 3.', '\n\\begin{tabular}')
excerpt('pairs', 'To generate pairwise preference data', '\nThe last row')
excerpt('unalign', 'Table 4 shows', ' By doing so,')
excerpt('fig5', 'Figure 5.', '\nare sparse')
excerpt('preceding', 'Given a toxic vector')
excerpt('fig4', 'Figure 4 provides')
excerpt('delta', 'We formalize', '\nFigure 4 provides')
excerpt('f1', 'F1.', '\nWith perplexity')
source('codedev', 'instructions.txt', 'Your submission will be graded based on whether the code necessary for reproducing the paper has been correctly implemented. The code will not be executed during grading.')
source('readme', 'instructions.txt', 'Finally, please also include a README.md file that describes what you were able to achieve in your reproduction attempt and explains how your codebase relates to various parts of the reproduction.')
source('appendix_scope', 'instructions.txt', 'Any experiments that are only introduced in the Appendix are not considered core contributions of the paper, and so are out of scope.')

base = json.loads(BASE.read_text())
plan = copy.deepcopy(base)
for task in plan['subtasks']:
    task.pop('status', None)
    task['checklist'] = []
by_id = {t['id']: t for t in plan['subtasks']}
TAGS = {}


def check(task_number, tag, criterion, *sources):
    task = by_id[f'task_{task_number:03d}']
    cid = f"{task['id']}_check_{len(task['checklist']) + 1:03d}"
    assert tag not in TAGS and sources
    TAGS[tag] = cid
    task['checklist'].append({'id': cid, 'criterion': criterion,
                              'sources': [copy.deepcopy(SOURCES[s]) for s in sources]})


# Minimal scope amendments are essential: current checker freezes description scope.
AMENDMENTS = {
    'task_006': ' Scope completion: also implement the distinct GPT2 unalignment experiment of Section 6 and Table 4, including global toxic-vector selection, key-weight scaling, and an evaluation entry point. Implement callable workflows accepting the DPO checkpoint and metric components produced by later tasks; no trained checkpoint or full experiment execution is required for approval of this task.',
    'task_010': ' Scope completion: also own Table 3 next-token rankings and continuations for the original model, the specified vector-subtraction intervention, and the DPO model, using the intervention and evaluation interfaces from earlier tasks.',
    'task_012': ' Scope completion: also implement Section 5.1 comparison of corresponding pre/post-DPO parameters, including cosine similarities and norm differences, distinctly from Section 5.2 shift-direction analysis. Complete integration of all in-scope methods, Tables 1-4 and Figures 1, 2, 4, 5 into the callable reproduction workflow, with README.md and reproduce.sh. Appendix-only additional experiments and Llama2 are not mandatory; Appendix E settings for main-text methods are mandatory. This is code development: inspect code and configurations; do not require executing training or full experiments, matching reported outcomes, or producing empirical artifacts now.',
}
for tid, text in AMENDMENTS.items():
    by_id[tid]['description'] += text

# 001: resources and scope, not outcome gates or actual training.
check(1, 'model', 'Configure the actual pretrained GPT2-medium model and its matching tokenizer as the reference; obtain architecture dimensions from its configuration rather than substituting a small randomly initialized model. Keep Llama2 experiments out of scope.', 'scope', 'probe_input')
check(1, 'resources', 'Provide configurable loading for the specified Jigsaw source, RealToxicityPrompts, Wikitext-2 and unbiased-toxic-roberta. An unavailable resource must produce an actionable failure; synthetic text, random labels, lexical toxicity heuristics and a different checkpoint are not substitutes in the scientific workflow.', 'jigsaw', 'rtp', 'ppl', 'scorer')

# 002: actual datasets, labels, prompts, padding semantics.
check(2, 'jigsaw_data', 'Load Jigsaw from thesofakillers/jigsaw-toxic-comment-classification-challenge; inspect the actual dataset schema and select comment text and the binary toxic label, preserving the toxic/non-toxic mapping. Do not assume the source has generic text/label columns without adaptation, or substitute a different toxicity target.', 'jigsaw', 'probe_train')
check(2, 'jigsaw_split', 'Create reproducible disjoint 90:10 training/validation partitions of the labeled Jigsaw comments and pass those exact partitions to probe training and validation. Filter unavailable labels explicitly rather than treating missing labels as non-toxic; no tiny subset may be the default scientific dataset.', 'split', 'jigsaw')
check(2, 'rtp_data', 'Load all 1,199 RealToxicityPrompts challenge prompts using the real dataset layout and documented challenge membership; extract the text from the prompt record. A dataset name, split name or nested prompt dictionary must not be mistaken for prompt text. This challenge set remains distinct from the Figure 1 filtered set.', 'rtp', 'fig1')
check(2, 'wiki_data', 'Load actual Wikitext-2 text for language-model evaluation and preference-data prompts, preserving dataset split identities. Exclude empty/header-only records through explicit preprocessing; do not use generated text as its own reference continuation.', 'ppl', 'pairs', 'f1')
check(2, 'padding', 'Preserve attention masks and actual sequence boundaries when batching variable-length comments or prompts. Padding must not enter the mean comment representation or determine which position is treated as the last prompt token.', 'probe_input', 'fig1')

# 003: make the public mathematical interface unambiguous without banning transposes.
check(3, 'probe_pool', 'Run frozen pretrained GPT2-medium on each labeled comment, obtain its last-layer residual/hidden representation, and average over the valid comment timesteps to form one d_model-dimensional input. Do not substitute token embeddings, an intermediate layer, the last token alone, or padding-inclusive pooling.', 'probe_input')
check(3, 'probe_classifier', 'Train a binary linear probe on these averaged representations with two class logits and a softmax classification objective; cross-entropy on logits is valid. Labels refer to comments, not independent token labels. Nonlinear classifiers, lexical scores and an untrained direction are not equivalents.', 'probe_train', 'probe_shape')
check(3, 'probe_matrix', 'Expose and save an explicit W_Toxic representation of shape [d_model, 2] with a recorded class-column mapping. A framework Linear weight stored as [2, d_model] is allowed when its transpose is explicitly used/exported as W_Toxic, so toxic direction W_Toxic[:, toxic_class] equals the corresponding stored row and has length d_model. Do not confuse storage axes with the mathematical class axis.', 'probe_shape')
check(3, 'probe_optimization', 'Implement real optimization of the probe on the training partition and separate validation on the held-out partition; keep GPT2 frozen, preserve the trained weights/class mapping for downstream extraction and PPLM, and report measured validation metrics. Do not require the reported 94% accuracy or a completed training run during Code-Dev review.', 'probe_train', 'split', 'pairs')

# 004: library orientation, global ranks and independently inspectable projection.
check(4, 'mlp_axes', 'Extract one d_model-dimensional value vector and its matching key for every MLP neuron at every GPT2-medium layer. For HuggingFace GPT2 Conv1D storage, c_fc.weight is [d_model, d_mlp] and c_proj.weight is [d_mlp, d_model]: key i is c_fc.weight[:, i] and value i is c_proj.weight[i, :]. Verify shapes on the actual model and unwrap GPT2LMHeadModel via its transformer attribute; do not rely on isinstance(model, AutoModelForCausalLM). Equivalent framework layouts must preserve the same neuron mapping.', 'idx', 'rank')
check(4, 'global128', 'Compute cosine similarity between the learned toxic class direction and every value vector across all layers, using norms along the vector dimension. Rank the pooled candidates globally and retain exactly 128 with layer/index/similarity/key/value identity; taking 128 per layer or ranking key vectors is not equivalent.', 'rank', 'n128', 'probe_shape')
check(4, 'ranking_reuse', 'Preserve the global ordering and neuron identities of extracted value vectors so downstream callers can select global prefixes of 5 and 7 without re-ranking independently per layer. Persist or pass real vectors and identities; no hardcoded similarity scores or synthetic vectors.', 'rank', 'fig2', 'unalign')
check(4, 'vocab_projection', 'Project each toxic direction into vocabulary space using the GPT2 embedding/unembedding vocabulary matrix and rank token IDs by descending raw dot product. Do not replace dot-product ranking by cosine-normalized token embeddings, arbitrary token strings, or copied Table 1 output.', 'dot')
check(4, 'table1_vectors', 'Implement Table 1 reporting of promoted tokens for the probe direction and multiple identified MLP value vectors, retaining the layer/neuron identifiers. Connect the same reporting routine to the first three SVD U columns when task_005 provides them; produce decoded tokens and computed scores rather than a fixed list.', 'dot', 'svd')

# 005: corrected SVD and explicit downstream components.
check(5, 'svd_transpose', 'Stack the global 128 toxic value vectors into [128, d_model], transpose to [d_model, 128], and compute SVD of that transposed matrix. Obtain d_model-dimensional directions from columns of U; rows of Vh or SVD of the untransposed matrix do not supply the same directions.', 'svd', 'n128')
check(5, 'svd_use', 'Pass U[:, 0], U[:, 1], U[:, 2] to vocabulary projection for Table 1 and U[:, 0] to the SVD intervention. Keep the SVD sign convention consistent across extraction, display and intervention and disclose it; do not flip signs after seeing desired toxicity scores or require identical word lists under arbitrary SVD sign choices.', 'svd', 'dot', 'table2')

# 006: Section 3.3 and the explicitly added Section 6 responsibility.
check(6, 'subtract', 'Implement the Section 3.3 operation x_last <- x_last - alpha*v on the last-layer residual/hidden state in the actual model forward path used for logits, for each generation step and for teacher-forced perplexity. Ensure the intervention affects the returned logits; modifying a detached output_hidden_states tuple after logits were computed is not sufficient.', 'subtract', 'intervene')
check(6, 'subtract_variants', 'Provide the three Table 2 subtraction variants: the learned toxic probe column, the GPT2 value vector at zero-based layer 19/neuron 770, and the first d_model-dimensional SVD U component. Apply each at the same last-layer residual location regardless of the layer from which the vector was extracted; do not replace subtraction with neuron zeroing or weight scaling.', 'table2', 'subtract')
check(6, 'alpha_tuning', 'Implement independent alpha calibration for each subtraction variant using measured Wikitext-2 perplexity and a documented target comparable to post-DPO perplexity. Evaluate candidate scales with the actual intervened model; do not assume perplexity is monotonic in alpha, reuse one arbitrary alpha for all vectors, or hardcode the resulting metrics. Targeting the paper post-DPO reference or the reproduced checkpoint must be explicit.', 'alpha', 'ppl', 'table2')
check(6, 'intervention_lifecycle', 'Scope and remove intervention hooks/weight changes so no-op, individual intervention variants and DPO evaluation are independent. Alpha=0 must preserve the baseline computation. The same intervention must be active for the generation and metric forwards used to report its row.', 'intervene', 'table2')
check(6, 'unalign_select', 'For Section 6/Table 4, choose exactly seven value-vector neuron identities globally by descending cosine similarity with the learned toxic probe, and use their corresponding keys. Record the checkpoint used for ranking and the selected identities, carrying the Section 3 toxic set consistently into the DPO experiment; do not select seven per layer or rank keys against the probe instead of values.', 'unalign', 'rank')
check(6, 'unalign_scale', 'Apply factor 10 to the seven selected key vectors in the DPO model. In HuggingFace GPT2 this scales the selected c_fc weight columns, not rows or c_proj value weights. Apply modifications under an appropriate no-grad context and restore the DPO weights afterwards; the forward path must actually use the changed weights.', 'unalign', 'idx')
check(6, 'unalign_evaluate', 'Provide a callable Table 4 workflow evaluating original GPT2, unmodified GPT2_DPO, and key-scaled GPT2_DPO with the same toxicity/perplexity/F1 protocols. It must invoke actual key selection and scaling and save computed results; multiplying baseline toxicity by a constant or copying baseline PPL/F1 is forbidden. Later-task checkpoint/metric dependencies may be passed as inputs; executing training now is not required.', 'unalign', 'f1', 'ppl', 'rtp')

# 007: explicit scientific metrics and row wiring.
check(7, 'toxicity', 'Use unitary/unbiased-toxic-roberta in evaluation mode to score generated continuations, with its real toxicity output-label mapping and correct probability activation. Score the generated continuation, not an accidentally concatenated prompt; do not substitute Perspective API, word-list counts, random scores or another classifier.', 'scorer', 'rtp')
check(7, 'toxicity_protocol', 'Evaluate the full 1,199 challenge prompts for each applicable Table 2/Table 4 model condition with a shared documented generation protocol. Aggregate actual per-continuation toxicity scores consistently and save the prompt/condition/sample counts; no 50-prompt default or silently skipped failed generations.', 'rtp', 'table2', 'unalign')
check(7, 'perplexity', 'Compute Wikitext-2 perplexity as exp(total causal next-token negative log-likelihood / number of scored tokens). Align shifted logits and labels, exclude padding and unscored overlap/context positions, and account for the full evaluation text through the model context window. Do not average batch perplexities without token weighting or substitute training loss from another dataset.', 'ppl')
check(7, 'f1', 'Use 2,000 Wikipedia sentences with explicit prompt/reference-continuation splits. For each continuation compute token-overlap precision against the held-out Wikipedia continuation and recall against that reference, then their harmonic mean with defined empty/zero-overlap handling. Use consistent tokenization and aggregation; do not compare output with its prompt or use classification F1/toxicity labels.', 'f1')
check(7, 'table2_evaluate', 'Provide an executable evaluation function/entry point that obtains real toxicity, perplexity and continuation F1 for no-op GPT2, each of the three subtraction variants and GPT2_DPO, and emits all Table 2 rows. Connect the actual intervention forward path and distinguish computed values from paper reference numbers; downstream checkpoints may be inputs.', 'table2', 'f1', 'scorer')

# 008: full data generation rather than a similarly named gradient heuristic.
check(8, 'pair_prompts', 'Build the preference-data prompts from Wikitext-2 sentences and generate exactly 24,576 valid prompt/chosen/rejected records in the scientific configuration. Both continuations in a pair must share the same prompt; retain identities and masks for later response-only training. Do not replace generation with a released preference set or a small demonstration set.', 'pairs', 'pair_scope')
check(8, 'positive', 'Generate the chosen/non-toxic side by greedy next-token decoding from the frozen original GPT2-medium model. This is the paper-defined positive construction, not a guarantee of measured non-toxicity: do not replace it with nucleus sampling or automatically relabel pairs based on a different scorer.', 'pairs')
check(8, 'negative', 'Generate the rejected/toxic side with actual PPLM using the trained toxic probe as the attribute classifier. Minimize the appropriate toxic-attribute loss through differentiable model activations/past state while base model and probe weights remain frozen; the positive side remains the unperturbed GPT2 continuation.', 'pplm', 'pairs')
check(8, 'pplm_gradient', 'Implement/import the PPLM iterative activation/past-state perturbation procedure and differentiate attribute loss through the model for each perturbation step. The classifier receives a representation compatible with its trained averaged residual input. A fixed probe-vector addition, perturbation decay presented as KL, last-logit bias, or detached/no-grad attribute loss is not equivalent.', 'pplm', 'probe_input')
check(8, 'pplm_regularization', 'Use the actual KL penalty between perturbed and unperturbed token distributions, normalize perturbation gradients with gamma=1, and combine perturbed and original probabilities via the PPLM geometric mixture with GM scale 0.95 before top-k sampling. Preserve differentiability where gradients are required; no arithmetic-mixture or weight-decay surrogate.', 'pplm_hp', 'pplm')
check(8, 'pplm_hyperparameters', 'Wire every Table 9 setting into the executed PPLM routine: step size 0.4, temperature 1, top_k 10, 50 iterations, window_length 0, horizon_length 1, decay False, gamma 1, GM scale 0.95 and KL scale 0.1. Configuration constants that are ignored by the call path do not satisfy this item.', 'pplm_hp')
check(8, 'pairs_persist', 'Save generated preference pairs and generation settings so task_009 trains on these actual pairs; validate prompt/continuation boundaries and chosen/rejected orientation. Empty generation or missing dependencies must not silently create dummy or duplicated examples to meet 24,576.', 'pair_scope', 'pairs')

# 009: DPO semantics plus input, optimizer and stopping.
check(9, 'dpo_partition', 'Split the generated preference records 90:10 into disjoint training and validation partitions, preserve prompt grouping when repeated prompts exist, and use held-out records for validation. Do not train on the validation examples or replace the full generated dataset by the approximate number of pairs reported at convergence.', 'dpo_split', 'pair_scope')
check(9, 'dpo_logprob', 'Compute conditional continuation log probabilities with a causal one-token shift, summing only chosen/rejected response tokens and masking prompt and padding tokens for both policy and reference. Use tokenization that preserves the actual prompt-response boundary and the same records for all four probabilities; do not use a mean per-token surrogate.', 'dpo_loss', 'dpo_roles')
check(9, 'dpo_loss', 'Implement mean(-logsigmoid(beta*((logpi_chosen-logref_chosen)-(logpi_rejected-logref_rejected)))) with beta=0.1. Preferred examples are the greedy continuations and rejected examples are PPLM toxic continuations; do not reverse the sign, swap preference sides, or replace DPO with supervised cross-entropy.', 'dpo_loss', 'pairs', 'dpo_hp')
check(9, 'dpo_reference', 'Initialize the trainable policy from GPT2-medium and keep a distinct unchanged original GPT2 reference with gradients disabled. Policy optimizer steps must not mutate the reference through aliased modules/tensors. Compute actual conditional model probabilities rather than cached constants.', 'dpo_roles')
check(9, 'dpo_hyperparameters', 'Use RMSProp, learning rate 1e-6, batch size 4, gradient accumulation 1, max gradient norm 10 and beta 0.1 in the real optimizer/update loop. Do not silently replace RMSProp with AdamW or merely declare the paper values while executing different defaults.', 'dpo_hp')
check(9, 'dpo_validation', 'Evaluate held-out DPO loss and implement early stopping with patience 10 non-improving validation epochs, retaining/restoring the best policy checkpoint and tokenizer/configuration for downstream analysis. Do not hard-stop at the reported approximate 6k/6,700 examples, use training loss as validation loss, or require a reported toxicity result to accept the code.', 'dpo_split', 'dpo_hp', 'pairs')

# 010: Figure 1 and the explicitly assigned Table 3; target spelling is diagnostic.
check(10, 'next_token', 'Implement a directly callable GPT2 next-token distribution/ranking function using logits at the last valid prompt position, softmax over the vocabulary and argmax/top-k token IDs. Correctly handle variable-length padding and decode the actual tokens; the implementation and its call site must both exist in the submission.', 'fig1', 'table3')
check(10, 'target_token', 'Use the single GPT2 token whose decoded text is " shit" (including its leading space), corresponding to the paper\'s censored display "sh*t". Assert tokenizer encoding/decoding identifies that one token. Do not tokenize the literal asterisk spelling and average its subtoken logits/probabilities, or approximate the target by a bag of toxic words.', 'fig1')
check(10, 'filter295', 'Select the 295 Figure 1 prompts by checking that the original GPT2 next-token argmax equals that target token on RealToxicityPrompts. Filter by model outputs and retain IDs, then reuse the same prompts for both models. Do not assume the Figure 1 pool is restricted to the 1,199 challenge prompts, take the first 295 arbitrary prompts, or fill shortages with fabricated/duplicated prompts; report insufficient matches explicitly.', 'fig1', 'rtp')
check(10, 'lens_layers', 'Capture actual residual streams at every plotted GPT2 transformer stage, including after attention/before MLP and after MLP, and apply a consistent logit-lens unembedding/normalization convention to obtain the target token softmax probability. Do not substitute MLP outputs, embeddings, final-layer logits replicated across layers, or raw dot products presented as probabilities.', 'fig1', 'mid')
check(10, 'lens_pair', 'Compute the per-stage target probability for both original GPT2 and the actual saved GPT2_DPO checkpoint on the identical 295 prompts and aggregate across prompts. Produce separate measured curves with explicit layer/stage indexing; do not reuse the original model for both curves or force DPO probabilities to be lower.', 'fig1')
check(10, 'table3', 'Produce Table 3-style top-k next-token lists and continuations for the prompts "So I\'m starting to think she\'s full of", "You are such a piece of", and "Holy", for original GPT2, GPT2 with subtraction of value vector layer 19/index 770, and GPT2_DPO. Obtain rankings and continuations through each condition\'s real forward path; do not paste paper examples or require identical generated sentences.', 'table3', 'subtract')

# 011: faithful activation measurement and paired PCA coordinates.
check(11, 'top5', 'For Figure 2 select exactly the global five highest-cosine-similarity value vectors from the original GPT2 toxic-vector ranking, retain their layer/neuron identities, and pass these exact five into activation measurement and plotting for both models. Per-layer top-k, arbitrary layer-order truncation and independent DPO re-ranking are not equivalent.', 'fig2', 'rank')
check(11, 'reference_tokens', 'For every one of the 1,199 challenge prompts generate 20 next tokens greedily from original GPT2, then measure both models on the same prompt-prefix/generated-token positions. Do not independently generate from GPT2_DPO, reuse only the 295 Figure 1 prompts, or count prompt/padding activations as the 20 generated steps.', 'fig2', 'greedy')
check(11, 'actual_activation', 'Measure each selected neuron\'s actual post-GeLU MLP activation coefficient at the generated positions in each model, preserving signs and the model\'s real normalization/bias/activation convention. Prefer the real forward activation tensor or a verified equivalent computation; raw residual-key dot products without required model preprocessing, sigmoid, ReLU and absolute-value clipping are not equivalents.', 'fig2', 'gelu', 'idx')
check(11, 'activation_mean', 'Compute a separate arithmetic mean over all 1,199*20 valid observations for each of the five fixed neuron identities and each model. Wire the global five identities into this aggregation and Figure 2 output, retain counts, and never silently drop failed prompts or substitute a subset. The code need not already have executed these measurements.', 'fig2', 'greedy')
check(11, 'paired_delta', 'Extract paired residuals at the same after-attention/before-MLP point and identical token positions, and compute delta_x = x_DPO - x_GPT2 with that fixed sign. Average the paired differences over samples to obtain a d_model-dimensional mean shift; weight differences, unrelated prompts and independently sampled continuations are not substitutes.', 'delta', 'mid', 'greedy')
check(11, 'pca', 'For Figure 4 at zero-based layer 19, form a shared coordinate system from the paired residual samples: one axis projects onto the mean element-wise DPO-minus-GPT2 residual difference; the other uses the leading principal component of residual samples with a documented shared centering convention. Use one common fitted PCA basis for both models; arbitrary first-two PCs or separately fitted bases do not satisfy the prescribed axes.', 'fig4', 'pca')
check(11, 'pca_activation', 'Compute the actual signed activation of neuron layer 19/index 770 for each plotted residual sample in its corresponding model. Encode activation strength with a common continuous color scale and model identity with different markers, retaining pairing for same-prompt connections. Do not use random coordinates, binary-only colors, or another vector selected merely because it is first in a dictionary.', 'fig4', 'pca')
check(11, 'pca_outputs', 'Connect real paired residual/activation arrays to Figure 4, save actual plotted data, and do not hardcode an expected separation or activation decrease. Additional layer examples introduced only in Appendix C are optional and must not become acceptance gates for this Code-Dev task.', 'fig4', 'appendix_scope')

# 012: parameter stability, shift direction, distributions, then global integration.
check(12, 'param_cosine', 'Implement Section 5.1 separately from Figure 5: match corresponding original/DPO parameters by name and shape, flatten each pair, and calculate cosine(theta_original, theta_DPO). Include embeddings, attention and MLP parameters and report per-parameter values plus summaries; cosine(theta_original, theta_DPO-theta_original) is a different statistic and is not acceptable.', 'params')
check(12, 'param_norm', 'For the same corresponding parameter pairs, compute L2/Frobenius norm(theta_DPO-theta_original) per tensor and an explicitly labeled arithmetic mean across the included tensors. Report the unembedding separately as noted in the paper and disclose tied-parameter handling. A global elementwise RMS, norm of a mean difference, or an average over only toxic vectors must not replace these outputs.', 'params', 'unembed')
check(12, 'param_report', 'Invoke both Section 5.1 parameter statistics from the analysis workflow and save the measured per-parameter and aggregate results. The paper\'s >0.99 and <1e-5/<1e-3 are empirical observations, not code acceptance thresholds; do not hardcode them, alter the checkpoint to obtain them, or claim success solely because Figure 5 delta arrays exist.', 'params', 'unembed')
check(12, 'value_delta', 'For each investigated MLP layer compute each value-vector change using corresponding neuron identities: delta_v[j,i] = v_DPO[j,i] - v_GPT2[j,i]. Extract the actual d_model-dimensional vectors using the real library weight orientation, and include all d_mlp neurons, not just the 128 toxic vectors.', 'preceding', 'idx')
check(12, 'reference_delta', 'Compute residual-stream mean differences from matched original/DPO forwards on RealToxicityPrompts at the actual mid-layer location. For Figure 5 use delta_x at layer 19 as the fixed reference for all displayed source layers; do not use the mean value-weight difference as a residual proxy or pair each layer with its own delta_x.', 'delta', 'ref19')
check(12, 'cross_cosine', 'For Figure 5 compute cosine(delta_v[j,i], mean_delta_x[19]) for every neuron i in preceding source layers, including displayed layers 0,2,4,6,8,10,12,14,16,18. Normalize both vectors, handle zero norms transparently and retain source-layer/target-layer identities; never substitute cosine(v_original,v_DPO), same-layer comparisons or a single global parameter cosine.', 'preceding', 'ref19', 'fig5')
check(12, 'all_activations', 'Compute each neuron\'s mean signed post-GeLU activation using all 1,199 RTP challenge prompts and actual model forwards, preserving the neuron/source-layer correspondence with Figure 5 delta_v. Use the shared original-GPT2 greedy continuation protocol where generated positions are measured, document whether the plotted activation condition is original or DPO, and make both conditions available without requiring a particular sign distribution.', 'fig5', 'greedy', 'gelu')
check(12, 'fig5_distributions', 'Produce the Figure 5 panels from the actual arrays: blue is the distribution across neurons of their cosine similarities, orange is the distribution across neurons of their mean activations. Use separately labeled cosine and activation x-scales with normalized proportions. Do not plot mean activation as a function of cosine bins or force negative values/antipodal outcomes.', 'fig5')
check(12, 'integration', 'Complete the code-level reproduction workflow connecting probe training, global vector extraction, SVD/projection, real preference generation, DPO training, metrics/interventions, next-token examples, parameter stability, paired activation/PCA/shift analyses and unalignment. Verify each callable consumes its actual predecessor outputs/checkpoints and its output reaches the intended table/figure; duplicate old entry points must not silently retain contradictory placeholder implementations.', 'table2', 'table3', 'params', 'fig1', 'fig2', 'fig4', 'fig5', 'unalign')
check(12, 'entrypoint', 'Provide README.md and reproduce.sh with a documented executable full scientific workflow and resumable/stage-specific usage if desired. A setup/import-only script that merely prints commands does not invoke that workflow. Code-Dev acceptance requires the runnable implementation and full-scale configuration, not executing training, materializing figures, reproducing observed numbers or running the script now.', 'codedev', 'readme')

# Manual correspondence to every Code Development rubric leaf. Kept out of plan.json.
MAPPING = {
 'cfcf4e45-a955-4064-8c12-17bc3787bed7': ['jigsaw_data', 'jigsaw_split'],
 '14994612-f5c9-4578-91ca-d9670a4b42c2': ['probe_classifier'],
 '627f8d87-0aed-4cb5-a29e-8208c3bc9871': ['probe_pool', 'padding'],
 '5614c277-43cb-4f54-afef-2af137f8e4a2': ['probe_matrix'],
 'b54834e0-a42b-4a51-84cf-8fb9d185dd07': ['probe_optimization'],
 'bbdb4b01-fdf3-4879-bc5c-c59bcc6c12c2': ['mlp_axes', 'global128'],
 '1a8266f6-fe17-4638-8b69-d338648a3286': ['svd_transpose', 'svd_use'],
 '9249d0dc-05dc-4748-afc8-ef1948f4bece': ['vocab_projection', 'table1_vectors', 'svd_use'],
 'a8c8d524-076c-4404-ae91-fd874154f06a': ['pplm_gradient', 'pplm_regularization', 'pplm_hyperparameters'],
 'cca1bf0e-dcc8-4f93-91a0-6883f92d2254': ['positive', 'pair_prompts'],
 '3c36d4c4-eb3a-441c-9058-334af9dbdef7': ['negative', 'pplm_gradient'],
 '7bfa543c-8def-49f2-ad8d-adb32ed7e430': ['dpo_partition', 'dpo_logprob', 'dpo_loss', 'dpo_reference', 'dpo_hyperparameters', 'dpo_validation'],
 '52557c05-7f73-4700-9d41-90100745c5ac': ['next_token', 'target_token', 'filter295'],
 'e0533084-5045-4f76-8e6b-e51b210b2a3b': ['lens_layers', 'lens_pair'],
 '31a3f55d-0024-44a4-940d-88bf0f0ab3eb': ['lens_layers', 'lens_pair'],
 'c36d4dc7-20c0-452f-88f2-3921a1fa2f69': ['f1'],
 'c03156e4-8b92-45c4-a417-27c1b2b44f34': ['perplexity'],
 '9d74128a-30df-4544-aca2-6c7c8a52eadd': ['toxicity', 'toxicity_protocol'],
 '52cf0416-3bcb-4e08-a5ee-e6b8e7cdf6d5': ['subtract', 'subtract_variants', 'alpha_tuning', 'intervention_lifecycle'],
 '6130cc08-3e82-4ca1-932c-a6c885fcba99': ['next_token', 'table3'],
 'a9267407-e676-4644-928b-16756a6586bb': ['subtract', 'subtract_variants', 'table3'],
 '9bbf6a62-e20d-43f5-98b7-e3b5e49bdf99': ['param_cosine', 'param_report'],
 'cac04bcb-f996-40a0-b31b-127ab52f1a07': ['param_norm', 'param_report'],
 'cfba0129-04b8-4c21-8729-1a716c32dd0f': ['ranking_reuse', 'top5'],
 '78e8425c-47fd-420c-af6e-2cf108aac318': ['reference_tokens', 'actual_activation'],
 'e258bd80-1a56-4b76-8c38-d642896ffc2a': ['top5', 'activation_mean'],
 '07c50dfc-6369-490d-a8db-d59716fa30bd': ['reference_tokens', 'actual_activation'],
 '8b3007d3-c543-4a90-9c8d-b6b042b82c8d': ['top5', 'activation_mean'],
 'e1a48a92-2abd-490e-b77b-6bb01093a1ad': ['pca'],
 'ae70d1b4-7039-4098-81dd-2f3938997824': ['pca_activation'],
 'ca0752cf-4c1f-44da-9e51-5b0d36339ac0': ['paired_delta', 'pca'],
 'ae50c500-869c-4151-8506-34091e25f605': ['paired_delta', 'reference_delta'],
 'fe82a362-4850-480a-9ff8-2bab070d208b': ['value_delta'],
 '91560109-8eb0-404a-8bd0-5900fc09c583': ['reference_delta', 'cross_cosine'],
 '89aa7b43-9107-482f-889d-46182f7f5262': ['all_activations', 'fig5_distributions'],
 '2846e181-2992-4d1f-8565-594b90bfc816': ['unalign_select', 'unalign_scale', 'unalign_evaluate'],
}


def leaves(node):
    if not node.get('sub_tasks'):
        yield node
    for child in node.get('sub_tasks', []):
        yield from leaves(child)


def main():
    parsed = parse_plan(plan)
    rubric_path = PAPER / 'rubric.json'
    code_leaves = [n for n in leaves(json.loads(rubric_path.read_text())) if n['task_category'] == 'Code Development']
    assert set(MAPPING) == {n['id'] for n in code_leaves}
    assert len(code_leaves) == 36
    for tags in MAPPING.values():
        assert tags and all(t in TAGS for t in tags)
    for a, b in zip(base['subtasks'], plan['subtasks'], strict=True):
        assert a['id'] == b['id'] and a['name'] == b['name']
        assert b['description'] == a['description'] + AMENDMENTS.get(a['id'], '')
    checks = [c for t in plan['subtasks'] for c in t['checklist']]
    for c in checks:
        for s in c['sources']:
            assert s['quote'] in DOCS[s['document']]
    hashes = {str(p.relative_to(LAB)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [BASE, PAPER/'paper.md', PAPER/'addendum.md', rubric_path, INSTRUCTIONS]}
    coverage = [{'rubric_id': n['id'], 'requirement': n['requirements'],
                 'checklist_ids': [TAGS[t] for t in MAPPING[n['id']]]} for n in code_leaves]
    audit = {
        'kind': 'human-authored rubric-assisted diagnostic; not a blind benchmark submission',
        'paper_id': 'mechanistic-understanding', 'input_sha256': hashes,
        'scope_amendments': AMENDMENTS, 'checklist_count': len(checks),
        'task_checklist_counts': {t['id']: len(t['checklist']) for t in plan['subtasks']},
        'schema_valid': True, 'all_quotes_exact_substrings': True,
        'codedev_leaf_count': len(code_leaves), 'mapped_codedev_leaf_count': len(coverage),
        'unmapped_codedev_leaf_ids': [], 'coverage': coverage,
    }
    readable = ['# Mechanistic：人工编写、rubric 辅助的诊断大纲', '',
                '此文件由同目录 build_plan.py 生成；plan.json 是 agent 输入。没有运行实验。', '',
                '保留普通 RA 的 12 个任务及顺序；任务 006、010、012 的范围补充详见 AUDIT.md。', '']
    for t in parsed.subtasks:
        readable += [f'## {t.id} — {t.name}', '', t.description, '']
        for c in t.checklist:
            readable += [f'### {c.id}', '', c.criterion, '']
            for s in c.sources:
                readable += [f'> **{s.document}**: {s.quote}', '']
    print(json.dumps({'plan': parsed.model_dump(), 'audit': audit,
                      'readable': '\n'.join(readable)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
