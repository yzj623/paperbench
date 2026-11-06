"""Human-authored rubric-assisted diagnostic plan; offline validation only.

Prints a JSON bundle; does not write files, call an LLM, or start experiments.
Run with ra-skill-list/.venv-wsl/bin/python. See AUDIT.md for source conflicts.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
LAB = ROOT.parent
PAPER = LAB / 'PaperBench-Official/project/paperbench/data/papers/all-in-one'
BASE = LAB / 'PaperBench-Experiments/all-in-one/2026-09-15T02-55-50Z/agent-run/research_agent/plan.json'
INSTRUCTIONS = LAB / 'PaperBench-Official/project/paperbench/paperbench/instructions/code_only_instructions.txt'
sys.path.insert(0, str(ROOT / 'src'))
from research_agent.plan import parse_plan
from research_agent.plan_enrichment import is_legacy_outline

DOCS = {n: (PAPER / n).read_text() for n in ('paper.md', 'addendum.md')}
DOCS['instructions.txt'] = INSTRUCTIONS.read_text()
SOURCES = {}


def quote(key, document, text):
    assert text in DOCS[document], (key, 'non-verbatim quote')
    SOURCES[key] = {'document': document, 'quote': text}


def paragraph(key, document, marker):
    found = [p.strip() for p in DOCS[document].split('\n\n') if marker in p]
    assert len(found) == 1, (key, len(found))
    quote(key, document, found[0])


def span(key, document, start, end):
    text = DOCS[document]
    a = text.index(start)
    b = text.index(end, a + len(start))
    quote(key, document, text[a:b].strip())


paragraph('codedev', 'instructions.txt', 'The code will not be executed during grading.')
paragraph('readme', 'instructions.txt', 'Finally, please also include a README.md')
paragraph('scope', 'instructions.txt', 'Any experiments that are only introduced in the Appendix')
paragraph('project', 'paper.md', 'We used JAX')
span('ve', 'paper.md', 'For VESDE:', 'For VPSDE:')
span('vp', 'paper.md', 'For VPSDE:', 'We set')
paragraph('sde_constants', 'paper.md', 'We set $\\sigma_{\\max }=15')
span('reverse', 'paper.md', 'Samples from the generative model', '\\section*{3. Methods}')
paragraph('arch', 'paper.md', 'The employed transformer model features')
paragraph('time', 'addendum.md', 'Diffusion time is embedded')
span('tokens', 'paper.md', '\\subsection*{3.1.', '\\subsection*{3.2.')
paragraph('repeat', 'addendum.md', 'To correctly compute the embedding for values')
paragraph('condition_embed', 'addendum.md', 'To embed the condition mask')
paragraph('concat', 'addendum.md', 'For each input, the tokenizer concatenates')
span('masks', 'addendum.md', 'During training,', '## Graph Inversion')
span('graph', 'addendum.md', 'Algorithm Graph Inversion', '## Tasks')
span('base_masks', 'addendum.md', '## Task Dependencies', '# Tokenization')
span('dependency', 'paper.md', '\\subsection*{3.2.', '\\subsection*{3.3.')
span('conditional_loss', 'paper.md', '\\subsection*{3.3.', '\\subsection*{3.4.')
paragraph('sampling', 'paper.md', 'For inference, we solved the reverse SDE')
span('guidance', 'paper.md', '\\subsection*{3.4.', '\\section*{4. Results}')
paragraph('guidance_details', 'paper.md', 'In our methodology, we integrate two principal strategies')
span('algorithm1', 'paper.md', 'Algorithm 1 General Guidance', 'Benchmarking the Guidance Methods:')
paragraph('sbi', 'paper.md', 'For implementing Neural Posterior Estimation')
paragraph('rf', 'addendum.md', 'C2ST is implemented')
span('gaussian', 'paper.md', 'Gaussian Linear: The prior', 'Gaussian Mixture This')
span('mixture', 'paper.md', 'Gaussian Mixture This', 'Two Moons :')
span('moons', 'paper.md', 'Two Moons :', 'SLCP Task:')
span('slcp', 'paper.md', 'SLCP Task:', 'Tree: This')
span('tree', 'paper.md', 'Tree: This', 'HMM: This')
span('hmm', 'paper.md', 'HMM: This', 'Lotka Volterra The')
span('lv_model', 'paper.md', 'Lotka Volterra The', 'SIRD Model with')
span('sird_model', 'paper.md', 'SIRD Model with', 'Hodgkin-Huxley Model:')
span('hh_model', 'paper.md', 'Hodgkin-Huxley Model:', '\\section*{A3. Additional experiments}')
span('hh_stats', 'addendum.md', 'The summary statistics computed', 'In the Hodgkin-Huxley task, the energy')
span('hh_energy', 'addendum.md', 'In the Hodgkin-Huxley task, the energy', '## Task Dependencies')
span('bench', 'paper.md', '\\subsection*{4.1.', '\\subsection*{4.2.')
span('lv_app', 'paper.md', '\\subsection*{4.2.', '\\subsection*{4.3.')
span('sird_app', 'paper.md', '\\subsection*{4.3.', '\\subsection*{4.4.')
span('hh_app', 'paper.md', '\\subsection*{4.4.', '\\section*{5. Discussion}')
paragraph('grid', 'addendum.md', 'infer the posterior distributions on a uniform grid')
quote('excluded', 'addendum.md', DOCS['addendum.md'].split('# Experiments\n', 1)[1].strip())

# Attach focused evidence instead of copying a whole application section to every
# item. Full source documents and their hashes remain available in the audit.
def fragment(key, parent, start, end=None):
    s = SOURCES[parent]
    a = s['quote'].index(start)
    b = s['quote'].index(end, a+len(start)) if end else len(s['quote'])
    quote(key, s['document'], s['quote'][a:b].strip())


fragment('token_identity', 'tokens', 'The tokenizer represents', 'In our experiments,')
fragment('token_function', 'tokens', 'The Simformer uses learnable', 'Finally, specialized')
fragment('encoder_figure', 'tokens', 'Figure 2.', 'of space or time,')
fragment('dep_semantics', 'dependency', 'These constraints can be implemented', '![](assets/asset_3.jpg)')
fragment('loss_formula', 'conditional_loss', 'Having defined the tokenizer', 'where $s_')
fragment('sample_condition', 'conditional_loss', 'We draw samples', 'Having access to all conditional')
quote('loss_weight', 'paper.md', r'where $\lambda$ denotes a positive weighting function.')
fragment('reverse_equation', 'reverse', 'Samples from the generative model', 'where $\\tilde')
fragment('guide_formula', 'guidance', 'In that line,', 'We detail the algorithm')
fragment('guide_scale', 'guidance_details', 'General guidance requires', '```')
fragment('guide_denoise', 'guidance_details', 'The second strategy entails', 'For comparative purposes,')
fragment('moons_ref', 'moons', 'To obtain reference samples', 'This procedure yielded')
fragment('slcp_ref', 'slcp', 'To obtain reference samples', 'This procedure yielded')
fragment('tree_ref', 'tree', 'To obtain reference samples')
fragment('hmm_ref', 'hmm', 'To obtain reference samples')
fragment('bench_posterior', 'bench', 'We evaluated performance', '![](assets/asset_4.jpg)')
fragment('bench_variants', 'bench', 'Across all four benchmark tasks,', 'Averaged across all benchmark tasks')
fragment('bench_conditional', 'bench', 'Next, we evaluated', '(Fig. 4b).')
fragment('lv_figure', 'lv_app', 'Figure 5.', 'note that, unlike')
fragment('lv_predict', 'lv_app', 'We then used the ability', 'As a second synthetic observation')
fragment('sird_figure', 'sird_app', 'Figure 6.', 'and death rate where')
fragment('sird_measure', 'sird_app', 'We generated synthetic measurements', 'Additionally,')
fragment('sird_predict', 'sird_app', 'Additionally, as we already', 'Next, we demonstrate')
fragment('hh_figure', 'sird_app', 'Figure 7.', 'parameter and a single measurement')
fragment('hh_unconstrained', 'hh_app', 'We first inferred', 'To identify energy-efficient')
fragment('hh_threshold', 'hh_app', 'To identify energy-efficient', 'Additional details and results')

SOURCE_OVERRIDES = {
 've_kernel':'ve reverse_equation sde_constants',
 'weighted_score':'loss_weight loss_formula', 'em':'reverse_equation sampling',
 'clamp':'sample_condition', 'encoder':'encoder_figure arch',
 'mask_application':'dep_semantics', 'identifiers':'token_identity token_function',
 'metadata':'token_function arch', 'concatenation':'concat arch token_identity',
 'functional_tokens':'token_function sird_figure lv_figure',
 'graph_output':'graph dep_semantics',
 'constraint':'guide_formula', 'denoise':'guide_denoise algorithm1',
 'guidance_sign':'guide_formula guide_scale algorithm1',
 'recurrence':'algorithm1', 'guidance_clamp':'sample_condition algorithm1 guide_formula',
 'logjoint':'bench_conditional moons_ref slcp_ref tree_ref hmm_ref',
 'chains':'moons_ref slcp_ref tree_ref hmm_ref', 'slice':'moons_ref slcp_ref',
 'moons_protocol':'moons_ref', 'slcp_protocol':'slcp_ref',
 'hmc':'tree_ref hmm_ref', 'endpoints':'moons_ref slcp_ref tree_ref hmm_ref',
 'lv_observe':'lv_model lv_figure', 'lv_reference':'lv_predict lv_model',
 'sird_noise':'sird_model sird_measure', 'sird_function':'token_function sird_figure sird_model',
 'hh_setup':'hh_model hh_figure', 'energy_scalar':'hh_energy hh_stats hh_unconstrained',
 'sbi_sampling':'sbi bench_posterior',
 'train_masks':'loss_formula masks graph', 'train_noise_loss':'loss_formula loss_weight',
 'train_optimizer':'arch bench_posterior sde_constants',
 'inference_path':'sample_condition sampling guide_formula',
 'c2st':'rf bench_posterior', 'c2st_target':'rf bench_posterior',
 'fig4a':'bench_posterior bench_variants arch excluded',
 'ten_observations':'bench_posterior moons_ref slcp_ref', 'fig4b':'bench_conditional',
 'benchmark_results':'bench_variants excluded codedev',
 'lv_configs':'arch lv_figure', 'lv_cases':'lv_figure',
 'lv_grid':'lv_predict grid', 'lv_compare':'lv_figure lv_predict rf codedev',
 'sird_dense':'arch sird_figure', 'sird_case_a':'sird_figure sird_measure token_function',
 'sird_case_b':'sird_figure sample_condition', 'sird_predict':'sird_figure sird_predict excluded codedev',
 'hh_dense':'arch hh_figure hh_stats', 'hh_unconstrained':'hh_unconstrained codedev',
 'energy_quantile':'hh_threshold guide_formula', 'hh_guided':'hh_threshold guide_formula algorithm1',
 'hh_comparison':'hh_threshold hh_figure codedev',
 'final_figures':'bench_variants lv_figure sird_figure hh_figure',
 'integration':'loss_formula sample_condition bench_posterior lv_predict sird_predict hh_threshold',
 'no_run_gate':'codedev scope excluded guide_formula',
}

base = json.loads(BASE.read_text())
plan = copy.deepcopy(base)
AMENDMENTS = {
 'task_002': ' Diagnostic clarification: use the SDE conventions and consistent perturbation/score units specified in the checklist; time embedding is implemented by task_003. No training or full sampling run is required at this stage.',
 'task_003': ' Diagnostic dimensional convention: diffusion time uses 128 sampled frequencies and concatenated sine/cosine features of width 256 before each block projection. This is a disclosed rubric-assisted choice where the printed paper says 128-dimensional; metadata Fourier features are separately width 128 before projection. Token width is 50, while each of four attention heads has Q/K/V width 10.',
 'task_007': ' Scope completion: include HMM as a sixth benchmark, exact joint log-density/conditional interfaces, and the distinct N-chain reference protocols for Two Moons, SLCP, Tree and HMM. Use the dimensionally consistent task/mask definitions and disclosed source-error resolutions in the checklist. Implement the samplers; do not generate large reference datasets during Code-Dev approval.',
 'task_013': ' Code-Dev scope: implement runnable experiment orchestration and result collection, without requiring experiments to execute now. Figure 4(b) includes Two Moons, SLCP, Tree and HMM and 100 random conditional/joint queries per task. Figures 4(a,b) use 1,000/10,000/100,000 simulations. Execution success and reported numeric results are not approval conditions.',
 'task_014': ' Code-Dev scope: implement the full experiment without running it now. The 100,000-simulation model supplies the two illustrative cases; the quantitative Figure 5(c) sweep additionally uses 1,000/10,000/100,000 simulations and dense/undirected/directed masks. Predictions include a uniform 0-to-15 time grid. Dense attention is used for the illustrative cases.',
 'task_015': ' Code-Dev scope: implement these workflows without running training or inference now. Use dense attention and eight layers. Implement both Figure 6 conditions; use four beta(t) observations plus one deceased-population observation for panel (b), following the figure caption rather than the conflicting infected-population sentence. Grid queries span 0 to 40; expose irregular query times too.',
 'task_016': ' Code-Dev scope: implement both unconstrained and energy-guided inference and both predictive routes, without running experiments now or requiring their reported outcomes. Use dense attention and eight layers. Guidance must use the full denoised-score/self-recurrence procedure. The energy threshold is derived from the unconstrained energy posterior predictive and is also explicitly configurable.',
 'task_017': ' Code-Dev scope: inspect the complete executable workflow and scientific configurations; training, large simulations, MCMC and generation of final figures are not required during approval. Inspect saved-result plotting code and data provenance rather than demanding empirical agreement with the paper.',
}
for t in plan['subtasks']:
    t['description'] += AMENDMENTS.get(t['id'], '')
    t['checklist'] = []

TAGS = {}
MAP = {}


def check(task, tag, criterion, sources, rubrics=''):
    t = plan['subtasks'][task - 1]
    cid = f"{t['id']}_check_{len(t['checklist']) + 1:03d}"
    assert tag not in TAGS
    TAGS[tag] = cid
    sources = SOURCE_OVERRIDES.get(tag, sources)
    t['checklist'].append({'id': cid, 'criterion': criterion,
                           'sources': [copy.deepcopy(SOURCES[s]) for s in sources.split()]})
    for rid in rubrics.split():
        MAP.setdefault(rid, []).append(cid)


# 001: preparation; final integration belongs to task_017.
check(1, 'environment', 'Provide an installable environment with explicitly recorded versions for the chosen numerical backend, transformer implementation, sbi and scikit-learn. Imports and configuration paths must resolve from the submission, without depending on another experiment directory. Record available resources; setup must not implicitly launch training or download generated experimental results.', 'project codedev')
check(1, 'config', 'Provide task-specific scientific configuration and output/checkpoint locations, with separate simulation budget, batch size, optimizer, attention variant, random seed and inference-step settings. The scientific defaults must remain full-scale; any small diagnostic preset must be explicitly separate. The initial reproduce.sh may be a scaffold, with full orchestration owned by task_017.', 'arch sde_constants sampling codedev')

# 002: units, score target and the actual reverse-time numerical update.
check(2, 've_coeff', 'Implement VE drift f(x,t)=0 and diffusion g(t)=sigma_min*(sigma_max/sigma_min)**t*sqrt(2*log(sigma_max/sigma_min)), with sigma_min=0.0001 and sigma_max=15. g is a diffusion coefficient, not the marginal variance or standard deviation; the actual sampling/training configuration must use these constants.', 've sde_constants', '0ac16f05 1402caf0 07f7aac1 c6cd56ae')
check(2, 've_kernel', 'For VE use mean x0 and marginal standard deviation sigma(t)=sigma_min*(sigma_max/sigma_min)**t; sample xt=x0+sigma(t)*epsilon and target score=-(xt-x0)/sigma(t)**2=-epsilon/sigma(t). Distinguish variance sigma(t)**2 from standard deviation at library calls. Initialize terminal unobserved variables with scale sigma(1)=15. Do not copy the malformed rubric kernel mean xt as a data-dependent forward mean.', 've reverse sde_constants', '396c0b27')
check(2, 'time_range', 'Sample a diffusion time uniformly in [1e-5,1] for each training sample and use the same bounds for inference; keep this time distinct from physical simulator observation times. Handle scalar and batched time broadcasting across variable axes explicitly.', 'sde_constants conditional_loss', 'ac3759f1 34b6fc70')
check(2, 'vp_optional', 'Implement the planned VP option consistently with beta(t)=0.01+t*(10-0.01), drift=-0.5*beta(t)*x and diffusion=sqrt(beta(t)). The marginal mean coefficient is exp(-0.5*integral_0^t beta), with variance 1-exp(-integral_0^t beta). Document the missing x factor in the printed drift as a source correction. The main experiment configurations must select VE; extra VP benchmark runs are not required.', 'vp sde_constants bench excluded')
check(2, 'weighted_score', 'Implement a per-example squared denoising-score error multiplied by a positive explicit time weight before batch reduction. For this diagnostic use lambda(t)=g(t)**2 (or a mathematically equivalent explicit weighting), and preserve the unobserved-coordinate mask. A helper defining lambda without its use in the training loss does not satisfy this criterion.', 'reverse conditional_loss', '0e335268 b3e915ef e3cd228e 2cb4d86f')
check(2, 'em', 'For positive step h on a decreasing time grid, implement reverse Euler-Maruyama as x_next=x-(f(x,t)-g(t)**2*score(x,t))*h+g(t)*sqrt(h)*epsilon, with fresh standard Gaussian noise. An equivalent noise sign is valid. Use 500 reverse steps by default and actual endpoint differences; noise-free ODE integration or a DDPM update is not a replacement.', 'reverse sampling', '7c953e3a 23d6d4c9')
check(2, 'clamp', 'Initialize only unobserved coordinates from the terminal noise distribution, pass the fixed observed values and condition state to the score model, and keep observed coordinates exactly constant after every reverse update. Support joint, posterior, likelihood and mixed parameter/data queries. Return samples in a documented variable order; implement all-observed and all-unobserved edge cases without NaNs.', 'conditional_loss', '79df6718')

# 003: token width 50 is NOT divisible by 4; separate per-head width is necessary.
check(3, 'encoder', 'Use an encoder-only transformer with multi-head self-attention, feed-forward blocks, residual paths and layer normalization. Accept diffusion time, token sequence and attention mask, preserve batch/token identity, and finish with a shared single linear map from each token hidden state to one scalar score. Return shape [batch,variables] (or a documented trailing singleton); no token pooling, batch/token diagonal extraction, vector-valued score with later slicing, or nonlinear multi-layer output head.', 'tokens dependency arch', '09008e24 25eecc1a')
check(3, 'heads', 'Use token/model width 50 and four attention heads, each with query/key/value width 10. Project 50-dimensional tokens to 4*10-dimensional Q/K/V and the concatenated head outputs back to 50. Do not silently change model width to 40/48/52/64 to accommodate a stock attention layer that requires divisibility by four.', 'arch', 'cdf1bfdd e9edc2a0 32e4ef3f')
check(3, 'ff_depth', 'Use feed-forward hidden width 150 (widening factor 3), six transformer layers for Section 4.1 benchmarks and eight for Lotka-Volterra, SIRD and Hodgkin-Huxley. Expose and actually use the layer count in each application constructor; a common constructor default alone does not establish the requested per-experiment depth.', 'arch', '85db9bf8 237efc4f aa888ef9 a0b66551 4dce6753')
check(3, 'time_features', 'Use random Gaussian Fourier features of diffusion time. For this rubric-assisted diagnostic choose 128 fixed sampled frequencies, concatenate sine and cosine into 256 features, and linearly project to width 50. This explicit diagnostic choice resolves the paper-versus-rubric width discrepancy for this input; it is not a claim that the paper explicitly states 256. Keep time features separate from metadata features.', 'arch time', '5963d717')
check(3, 'time_blocks', 'Add a learned linear projection of diffusion-time features to the output of every transformer feed-forward block, broadcasting per-example time across tokens. Time injection solely into input tokens, only one block, or an unused projection does not satisfy the specified architecture.', 'time arch', 'b1b80f04')
check(3, 'mask_application', 'Apply each sample\'s attention mask to query-key logits before softmax in every layer/head, with explicit allowed/blocked boolean semantics and self-attention on the diagonal. Support a per-batch-item mask without leaking masks across examples. Dense means every real token can attend to every other real token; padding, if used, remains excluded.', 'dependency base_masks', '3e515973 8e07cc4f 08a90ef1')

# 004: preserve field meaning through tokenization, training and decoding.
check(4, 'identifiers', 'Assign integer identifiers by scientific variable identity, with learned embedding parameters. Multiple time points/observations of the same function or population share its identifier and are distinguished by metadata; independent scalar parameter identities remain distinct. Fixed one-hot vectors and a new learned ID for every sequence position are not equivalent.', 'tokens', '437abe9f 714bdd7e')
check(4, 'value_repeat', 'Embed each scalar value by repeating that scalar along the configured value-embedding dimension. Preserve its continuous numeric value and gradient; do not replace repetition by a learned scalar projection, discretized token ID or positional feature.', 'repeat', 'af5cd308')
check(4, 'metadata', 'For function-valued parameters or observations, encode the actual index-set coordinates such as observation time using random Fourier features, followed by a learned linear projection to the metadata field width. For this diagnostic the Fourier feature vector before projection has 128 components (e.g. 64 sine/cosine pairs). Route this field into the token; a diffusion-time embedding does not provide observation-time metadata.', 'tokens arch', '2e46fc8d da87d682')
check(4, 'condition_vector', 'Use a shared trainable condition vector for True/observed and an exact zero vector for False/unobserved. These flags must agree with the loss/sampler convention that True means held fixed. Two nonzero class embeddings or an inverted boolean convention are not equivalent.', 'condition_embed masks', '2ff0a481')
check(4, 'concatenation', 'Concatenate identifier, repeated value, optional metadata, and condition-state embeddings in that order. Their configured widths must yield token width 50, with a documented layout for tasks without metadata. Do not add the fields together. Input scalar values, integer IDs, condition mask and optional metadata must share the same sample and variable axes.', 'concat arch tokens', '407dcc3b c6eb15a5')
check(4, 'mask_distribution', 'For every element of every training batch independently choose uniformly among five mask families: joint all False; posterior parameters False/data True; likelihood parameters True/data False; independently resampled per-variable Bernoulli(0.3); independently resampled per-variable Bernoulli(0.7). Use 0.7, correcting the paper typo p=7. Choosing one family for an entire batch or reusing a fixed random mask changes the specified protocol.', 'masks', 'be65afa3')
check(4, 'functional_tokens', 'Support variable observation counts and arbitrary requested index-set points using shared variable identifiers plus physical-time metadata. Preserve which tokens are parameters versus data when constructing likelihood/posterior masks. Function values at query times must be model variables, not post-hoc interpolation of a fixed set of output neurons.', 'tokens sird_app lv_app')

# 005: complete Webb procedure and dimension/orientation corrections.
check(5, 'graph_init', 'Treat the base Bayesian graph as a DAG without self-loops for graph operations, and set Z to the unobserved nodes from M_C. Moralize by removing direction and connecting every pair of co-parents; set all vertices unmarked; initialize H with all variables and no edges. Self-attention loops belong to the final attention mask, not DAG parent tests.', 'graph', '6f05f0cf')
check(5, 'graph_candidates', 'Initialize the eligible set S to latent nodes with no latent parent in the original G. At each iteration select an eligible node minimizing the number of missing edges between its currently unmarked J-neighbours; a documented deterministic tie-break is acceptable. Eliminating arbitrary nodes, ignoring the eligibility set or using degree as a substitute for min-fill is not equivalent.', 'graph', '6f05f0cf')
check(5, 'graph_elimination', 'For selected v, add all fill edges between unmarked neighbours in J and make those neighbours parents of v in H. Mark v, remove it from S, and inspect its unmarked latent children in original G; add a child only when all its latent parents in G are marked. Continue until no eligible nodes remain. Keep the evolving J, original G and output H separate.', 'graph', '6f05f0cf')
check(5, 'graph_output', 'Form the final directed attention mask as base M_E union edges(H), retaining its diagonal. Translate parent-to-child graph edges into the chosen query-row/key-column convention explicitly. Recompute or cache by the full condition mask and base graph, so two different conditionings may receive different valid masks in training and inference. A fixed triangular, symmetrized or dense fallback is not the directed variant.', 'graph dependency', '6f05f0cf 3e515973')
check(5, 'mask_benchmarks', 'Construct task masks with actual simulator dimensions and a documented query-row/key-column orientation. Gaussian Linear uses independent theta_i-to-x_i edges; Two Moons/Gaussian Mixture use both parameter parents for each of their TWO data variables and a lower-triangular data block; SLCP uses FIVE parameter parents and four 2-by-2 lower-triangular observation blocks. Repair the addendum\'s illustrative 10-data/4-parameter sizes rather than adding phantom variables. Self-edges remain True.', 'base_masks gaussian mixture moons slcp', '08a90ef1')
check(5, 'mask_tree_hmm', 'For Tree use exactly 3 parameter and 4 data nodes: theta0 parents theta1/theta2, theta1 parents x0/x1, and theta2 parents x2/x3. For HMM use 10 parameter and 10 data nodes with theta_i parenting theta_(i+1) and x_i. The addendum Tree matrix has inconsistent size/orientation; use these generative dependencies with the same orientation as other tasks. Undirected variants are the boolean union of directed base adjacency and its transpose, including the diagonal.', 'base_masks tree hmm', '08a90ef1')

# 006: guidance must influence the integrated score, not filter completed samples.
check(6, 'constraint', 'Expose interval conditioning as an optional sampling mode with configurable upper bound u and recurrence count r. Use c(x)=x-u for an upper bound, with feasible c<=0; support lower/two-sided or multiple constraints through the same interface. Distinguish this soft interval from exact observed-value clamping, and preserve the unconditional sampling path when guidance is disabled.', 'guidance algorithm1', '39362fd2 f5b9aa3e')
check(6, 'denoise', 'At the current reverse state estimate the clean variables by x0_hat=(xt+sigma(t)**2*s_phi(xt,t))/mu(t), using the chosen SDE\'s mean coefficient and marginal standard deviation. Evaluate the constraint on this denoised estimate. Calling the constraint solely on noisy xt or using a fixed trained-data energy statistic is not Algorithm 1.', 'guidance_details algorithm1', '58d5bd8c')
check(6, 'guidance_sign', 'For the c<=0 convention add grad_xt log(sigmoid(-s(t)*c(x0_hat))) to the base score, with s(t)=1/sigma(t)**2. Use a numerically stable log-sigmoid and maintain the derivative through the denoised estimate to xt. Resolve the printed Algorithm 1 sign discrepancy by this Section 3.4 convention (or use the exactly equivalent reversed definition of c). An upper-energy constraint must push energy downward, not upward.', 'guidance guidance_details algorithm1', '091fd46a 58d5bd8c')
check(6, 'recurrence', 'Implement self-recurrence as a genuine reverse/forward cycle at each diffusion-time interval: integrate the guided reverse step to the next lower-noise state; before an additional recurrence, resample forward to the previous higher-noise level with the forward SDE and fresh Gaussian noise, then recompute score and guidance. Return the final LOWER-time state; do not end the outer interval on a re-noised sample or treat r as repeated deterministic gradient descent.', 'algorithm1 guidance_details', '386472cc 58d5bd8c')
check(6, 'guidance_clamp', 'During both forward and reverse recurrence, update only latent coordinates and restore exact observations at every step. Define r unambiguously as either total guided reverse passes or extra cycles and handle zero/one consistently, so disabling extra recurrence still permits one reverse step. Connect the actual guided score to the reverse drift; no clipping/rejection-only implementation may replace guidance.', 'conditional_loss algorithm1 guidance', '58d5bd8c f5b9aa3e')

# 007: faithful distributions AND distinct reference-sampling protocols.
check(7, 'gaussian', 'Implement theta~N(0,0.1 I_10), x|theta~N(theta,0.1 I_10), with the second argument interpreted as covariance. Keep theta and data both 10-dimensional; a Normal API taking standard deviation must receive sqrt(0.1). The prior sampler, simulator and joint density must use the same convention.', 'gaussian', '23688299')
check(7, 'mixture', 'Implement theta~Uniform([-10,10]^2) and x|theta as equal weights of N(theta,I_2) and N(theta,0.01 I_2). The component indicator is sampled per observation and both coordinates use that component; log density uses the mixture log-sum-exp, not an average Gaussian or independent per-coordinate mixture choices.', 'mixture', '23688299')
check(7, 'moons', 'Implement Two Moons with theta in [-1,1]^2, alpha uniform on [-pi/2,pi/2], r of mean 0.1 and standard deviation 0.01, and x=(r*cos(alpha)+0.25-|theta1+theta2|/sqrt(2), r*sin(alpha)+(-theta1+theta2)/sqrt(2)). The printed 0.012 denotes 0.01 squared. Preserve the absolute value, rotation, offset and support; generic two-cluster Gaussians are not this simulator.', 'moons', '23688299')
check(7, 'slcp', 'Implement SLCP with five Uniform(-3,3) parameters and four conditionally i.i.d. bivariate observations flattened to eight data coordinates. Use the cited benchmark\'s valid covariance: s1=theta3**2, s2=theta4**2, rho=tanh(theta5), diagonal=(s1**2,s2**2), off-diagonal=rho*s1*s2, mean=(theta1,theta2). Document the printed diagonal typo; do not implement its generally indefinite covariance or change rho/clamp eigenvalues to conceal it. Tiny declared numerical jitter is acceptable.', 'slcp', '23688299')
check(7, 'tree', 'Implement theta0~N(0,1), theta1/theta2 independently conditional on theta0 with means theta0 and standard deviations 1. Observations x0,x1,x2,x3 have respective means sin(theta1)**2,0.1*theta1**2,0.1*theta2**2,cos(theta2)**2 and standard deviations 0.2,0.2,0.6,0.1. Preserve the 3+4 dimensions and factorization in both sampling and log-joint evaluation.', 'tree', '1f652a29')
check(7, 'hmm', 'Implement ten latent states theta0~N(0,0.5**2) and theta_(i+1)|theta_i~N(theta_i,0.5**2) for i=0,...,8; ten observations x_i|theta_i~N(theta_i**2,0.5**2). Use standard deviation 0.5 in Normal APIs; do not create an eleventh state from the printed indexing typo or replace squared emissions with linear observations.', 'hmm', '6fc3436c')
check(7, 'logjoint', 'Provide a differentiable joint log density consistent with each simulator, including prior support, covariance determinants and dependencies. For arbitrary conditioning, reconstruct the full joint state from fixed observed and free coordinates and evaluate that density; it is not sufficient to implement only p(theta|all x). For transformed free variables include the log-Jacobian. Do not treat a learned Simformer density or an ad hoc distance kernel as ground truth.', 'moons slcp tree hmm bench')
check(7, 'chains', 'For each requested conditional in Two Moons, SLCP, Tree and HMM initialize N independent chains from N draws of the full joint (sample parameters then simulate associated data), retain chain identity, then impose the fixed observed coordinates. If clamping creates an out-of-support state, repair/reinitialize transparently with a valid target state. A single chain, repeated common initial point, or prior-only initialization without the joint draw is not the prescribed protocol.', 'moons slcp tree hmm', '9fa888e9 af3a4299 acc664bc 52f952b5')
check(7, 'slice', 'Implement genuine random-direction slice transitions over the free coordinates, with a slice-height draw, a bracket search and shrinkage/acceptance inside the slice; reject invalid-support proposals. A function named slice that merely applies Gaussian random-walk MH is not equivalent. Keep all conditioned coordinates fixed.', 'moons slcp', 'ed3ab4fa 028a6cbe')
check(7, 'moons_protocol', 'Run 1,000 random-direction slice transitions followed by 3,000 Metropolis-Hastings transitions per Two Moons chain, with MH proposal step size 0.01. The stages act on the same chain states and target conditional; counts and proposal scale must reach the executed code, not just a configuration description.', 'moons', 'ed3ab4fa e4d0c4d5')
check(7, 'slcp_protocol', 'Run 600 random-direction slice transitions followed by 2,000 Metropolis-Hastings transitions per SLCP chain, with MH proposal step size 0.1. Include the correct prior/likelihood in acceptance and reject out-of-support proposals rather than clipping them into the parameter box.', 'slcp', '028a6cbe 1e941abb')
check(7, 'hmc', 'For Tree and HMM implement 5,000 HMC transitions per chain: sample momentum, use log-target gradients in leapfrog integration, compute Hamiltonian energy change and perform Metropolis acceptance/rejection. Document mass/step-size/leapfrog settings. Random-walk MH, unadjusted Langevin and a function only named HMC are not acceptable substitutes.', 'tree hmm', '35e0a7bf 293a5a7e')
check(7, 'endpoints', 'After the complete task-specific transition protocol, return exactly the last state of each of the N chains, yielding N reference samples of the requested free variables (or full states with fixed coordinates identified). Keep the chain axis separate from transition history. Concatenating one chain\'s retained trajectory, thinning intermediate states, or replicating chain endpoints to reach N is not equivalent.', 'moons slcp tree hmm', '92486973 0051bf87 462ad866 bc5f359e')

# 008: unstructured observations and metadata-sensitive graph.
check(8, 'lv_prior', 'Implement four rate parameters alpha,beta,gamma,delta from a sigmoid-transformed Normal prior scaled by 1+2*sigmoid(z), giving range (1,3). Document the Normal location/scale and initial populations because their numerical values are not fully specified in the supplied text. Use dx/dt=alpha*x-beta*x*y and dy/dt=delta*x*y-gamma*y; a uniform prior or analytic sine-wave surrogate is not equivalent.', 'lv_model', '9f70a4e5')
check(8, 'lv_observe', 'Numerically solve the coupled predator/prey dynamics and sample observations with additive independent Gaussian noise of standard deviation 0.1. Accept species-specific irregular time arrays and different observation counts; align each returned scalar with its true species and time. Inference inputs are full time-series values, not summary statistics or a common-grid interpolation substituted for observations.', 'lv_model lv_app', '9f70a4e5 bcf546a2 efe8edda')
check(8, 'lv_mask', 'Build the Lotka-Volterra base mask from actual time metadata: independent rate-prior nodes; prey receives its alpha/beta parents and predator receives gamma/delta; consecutive times within each population form Markov edges; each prey node additionally receives edges from all earlier predator nodes. Use true sorted physical times and restore caller order. Never infer temporal precedence from raw token position or assume equal counts/times for both species.', 'base_masks lv_model')
check(8, 'lv_reference', 'Provide an independent likelihood/prior-based posterior or arbitrary-conditional reference route for Lotka-Volterra, using the same ODE/noise model and exact observed time/species tuples. References must not be Simformer samples relabeled as truth. Expose reproducible initial conditions and solver tolerances; numerical failures must not silently emit zero trajectories.', 'lv_app lv_model')

# 009: function-valued rate, consistent priors/noise and metadata.
check(9, 'sird_ode', 'Implement dS/dt=-beta(t)*S*I, dI/dt=beta(t)*S*I-gamma*I-mu*I, dR/dt=gamma*I, dD/dt=mu*I, with all four compartments and documented population normalization/initial conditions. gamma and mu are constant within a trajectory; only beta varies with time. The paper later calls mortality delta: map that alias to the same mu, not to an extra parameter.', 'sird_model', 'b96b17cd 4332dc3c')
check(9, 'sird_global', 'Sample independent global recovery and mortality rates uniformly from (0,0.5), and use the same parameters in simulation, joint data construction and inference. Preserve the gamma/mu (or explicitly mapped gamma/delta) names and shapes; no Normal or fixed-rate replacement.', 'sird_model', '6c80714a')
check(9, 'sird_gp', 'Sample a zero-mean correlated Gaussian process on actual time points with RBF covariance K_ij=2.5**2*exp(-0.5*(t_i-t_j)**2/7**2), then apply sigmoid to obtain beta(t) in [0,1]. A single constant beta, independent per-time Gaussian draws or a hand-chosen sinusoid is not this prior. Use coherent GP paths when solving/querying between sampled points and disclose the numerical representation.', 'sird_model', '22bc638a 4332dc3c')
check(9, 'sird_noise', 'Implement log-normal observation noise around the selected S/I/R/D compartment trajectory with the stated noise parameter 0.05. Document whether 0.05 is a data-space standard deviation or log-space scale and map physical means to log-normal parameters consistently; for an arithmetic mean m/std 0.05, use log variance log(1+(0.05/m)**2) and log location log(m)-log_variance/2. Handle zero populations explicitly. Never pass a raw positive state as the log-location (which exponentiates the state), or replace the model by additive Gaussian noise.', 'sird_model sird_app', '19f4319f')
check(9, 'sird_function', 'Construct training examples by subsampling the same latent GP path and simulated trajectories at varying physical times, with beta tokens sharing a beta identifier and time metadata. At inference accept requested times beyond the training subsample, condition on observed beta values where supplied, and sample remaining beta values jointly with global rates/data. A fixed output vector followed only by interpolation does not implement the stated functional inference interface.', 'tokens sird_app sird_model', 'b96b17cd')

# 010: exact gating, sodium charge and separately testable energy conversions.
check(10, 'hh_setup', 'Implement the seven-parameter Hodgkin-Huxley simulator described in the paper, with an explicit parameter/prior schema and units including the plotted g_Na,g_K,C_m,g_L. Document choices not numerically specified by the paper rather than claiming invented prior ranges are paper values. Start V at -65 mV, integrate 200 ms, and apply a 4 mA current only in the 50-to-150 ms stimulus window. Keep physical-unit conversions consistent with the membrane-area convention.', 'hh_model hh_app', '64541d71 4884d911 548511a5 688df55a')
check(10, 'hh_rates', 'Implement all six Appendix A2.2 gating rates with its coefficients and shifts: alpha_m=0.32*efun(-0.25*(V-V0-13))/0.25; beta_m=0.28*efun(0.2*(V-V0-40))/0.2; alpha_h=0.128*exp(-(V-V0-17)/18); beta_h=4/(1+exp(-(V-V0-40)/5)); alpha_n=0.032*efun(-0.2*(V-V0-15))/0.2; beta_n=0.5*exp(-(V-V0-10)/40). Use efun(z)=z/expm1(z), with the series 1-z/2 near ZERO (absolute smallness), not for every negative z. This resolves the printed one-sided test without changing the rate law.', 'hh_model', '57d03e3d')
check(10, 'hh_integrate', 'Integrate coupled V,m,h,n and cumulative sodium charge H: dV=[Iinj-gNa*m**3*h*(V-ENa)-gK*n**4*(V-EK)-gL*(V-EL)]/Cm plus the stated 0.05 Wiener noise; dgating=alpha(V)*(1-gating)-beta(V)*gating; dH=gNa*m**3*h*(V-ENa). Scale stochastic increments with sqrt(dt), use the evolving gates/voltage and a numerically adequate step/solver. A spike-count model, synthetic voltage trace or energy independent of the sodium current is not equivalent.', 'hh_model', '64541d71 5b2d5ce9')
check(10, 'hh_features', 'Extract exactly seven voltage features from each actual noisy simulated trace: spike count; pre-stimulus/resting mean and standard deviation; stimulus/spiking-window mean; and second, third and fourth CENTRAL moments in that window. Define windows and spike detection explicitly in physical time and count events rather than above-threshold samples. Follow the addendum\'s raw central moments; optional standardized moments must be separately named, not substituted.', 'hh_stats hh_model', 'e2719676 0b14bb53')
check(10, 'energy_trace', 'For the cumulative sodium-charge trajectory H, implement the addendum trace conversion: first difference to obtain per-step noncumulative charge, then convolve with the length-5 uniform kernel in same mode, then call the charge-to-power conversion. These are charge increments, so do not multiply by dt a second time or treat the cumulative trajectory as per-step energy.', 'hh_energy hh_model', '5b2d5ce9')
check(10, 'energy_units', 'Implement charge-to-power with the addendum factors in order: negate inward sodium charge; divide by 1000 for mS-to-S and by 1000 for mV-to-V; multiply membrane area 0.628e-3; divide by elementary charge 1.602176634e-19, Na valence 1 and 3 Na transported per ATP; multiply ATP energy 10e-19 J; divide by 0.2 s; multiply by 1e6 to obtain microJ/s. Equivalent algebra is acceptable, but omitted area, missing sign/ATP conversion or arbitrary rescaling is not.', 'hh_energy', '5b2d5ce9')
check(10, 'energy_scalar', 'Produce a scalar whole-stimulus-duration energy/power statistic consistently from integrated charge H[-1]-H[0] via convert_total_energy; retain convert_charge_to_energy for the time-resolved trace. Do not average the already duration-normalized incremental trace as a substitute for total power. Append the energy scalar as a distinct eighth data statistic after the seven voltage statistics; keep its index/units available to conditional masks and guidance.', 'hh_energy hh_stats hh_app', 'd342c64a 5b2d5ce9')

# 011: installed SBI API determines valid calls; no guessed keyword gates.
check(11, 'sbi_methods', 'Instantiate NPE, NRE and NLE through the sbi Python library. NPE/NLE must select neural spline flows (NSF) rather than the default estimator; NRE must use the library ratio estimator. Feed actual matched theta/x simulations from the same benchmark definitions. Handwritten MLP regression, Gaussian surrogates or a custom non-sbi implementation cannot replace these baselines.', 'sbi', 'ad5f1aac')
check(11, 'sbi_train', 'Train through each selected sbi method\'s provided training API, with training_batch_size=1000 or the equivalent supported setting. Wire Adam into the optimizer path; when Adam is the pinned library\'s internal default, verify and document that implementation rather than inventing an unsupported optimizer keyword. Save the trained estimator and prior/transform information required for its actual posterior construction.', 'sbi', 'ee41ea7c 31ed240f 6a1ddafa')
check(11, 'sbi_early_stop', 'Use held-out validation loss and the sbi-supported validation split/early-stopping controls to stop training on convergence. Verify argument names against the pinned version\'s signatures; validation_fraction or other supported mechanisms are valid, and a nonexistent validate_data parameter must not be required. Do not substitute training-loss convergence or a tiny fixed epoch cap for the scientific early-stopping protocol.', 'sbi', '56dcd071')
check(11, 'sbi_sampling', 'Expose a common posterior-sampling interface conditioned on actual observations. NPE uses the learned posterior; NLE/NRE combine their learned likelihood/ratio with the prior through the appropriate library posterior/MCMC machinery. Preserve sample dimensions and physical parameter units so the benchmark C2ST compares the same target distributions. Do not sample estimator weights, regress to a point or compare likelihood samples to posterior references.', 'sbi bench')

# 012: training and inference integration; scientific execution is not approval.
check(12, 'train_masks', 'For each batch use matched prior/simulator joint samples and independently resample the five-family condition mask per example. Compute the graph adaptation for each directed-mask example and pass its condition mask, metadata, variable IDs and attention mask through the same tokenizer/model call. Do not train with one mask distribution and silently infer with another fixed dependency mask.', 'conditional_loss masks graph', 'be65afa3 34b6fc70')
check(12, 'train_noise_loss', 'Generate xt from the chosen SDE, replace observed positions with clean x0, and compute the scalar score error only at unobserved positions. Sum the squared coordinate errors per example, apply the explicit positive lambda(t), then reduce the batch. Conditioned and padded entries must contribute zero; an all-observed example must not cause division by zero. Assert or otherwise enforce matching [batch,variables] output/target shapes so unintended broadcasting cannot train a different objective.', 'conditional_loss reverse', '34b6fc70 0e335268 b3e915ef e3cd228e 2cb4d86f')
check(12, 'train_optimizer', 'Use Adam and batch size 1000 for all Simformer scientific configurations. Select VE in the main experiments; honor the declared simulation budget as the number of simulated training pairs instead of silently simulating fresh unlimited pairs each epoch. A last smaller batch is acceptable, and an explicit memory-equivalent accumulation scheme must preserve the intended update. Save actual trained weights and data/model configuration for inference.', 'arch bench sde_constants', 'b18e9e5a 831ca2ca 7ec03b27')
check(12, 'inference_path', 'Load the correct task/budget/attention-variant checkpoint, reconstruct its tokenizer and score model, and execute the 500-step reverse sampler on unobserved variables while clamping observed values throughout. Guidance, when selected, must call task_006 with the same scaling and variable ordering. Preserve transforms/units for reported physical samples; initializing a new random model at evaluation is not acceptable.', 'conditional_loss sampling guidance', '79df6718 7c953e3a 23d6d4c9')
check(12, 'c2st', 'Implement C2ST with a RandomForestClassifier of 100 trees, labels identifying two independent sample sets, and mean held-out accuracy over FIVE cross-validation folds. Use balanced sample counts or an explicitly balanced evaluation, shuffle/stratify reproducibly, and fit any learned preprocessing within each training fold. Do not report training accuracy, a single holdout split, a neural classifier or a fixed 0.5.', 'rf bench', '3957cfb9 982f7456')
check(12, 'c2st_target', 'Compare model and independent reference samples for the identical task, observation, free-coordinate set and units. Report raw classifier accuracy: near 0.5 means indistinguishability and near 1 means separation. Do not invert the score, force it into a desired range, include labels/chain IDs as features or reuse model samples as references. Save sample counts, condition masks and observation identifiers with metrics.', 'rf bench', '973db5a3')

# 013: complete Figure 4 pathways, without extra appendix-only experiments.
check(13, 'fig4a', 'Implement the Figure 4(a) experiment matrix: Gaussian Linear, Gaussian Mixture, Two Moons and SLCP; simulation budgets 10**3,10**4,10**5; six-layer VE Simformer with dense, undirected and directed masks plus NPE comparison. Route variant choice into actual masks/models and keep budget/task/model outputs distinct. NRE/NLE APIs remain implemented, but reproducing the appendix-only extended comparison is not an approval requirement.', 'bench arch excluded', '237efc4f 7ec03b27')
check(13, 'ten_observations', 'For each of the four standard posterior tasks provide ten fixed reference observations and their matching reference posterior samples, reusing them across model variants/budgets. Use official benchmark reference data where available or the specified validated target sampler; infer N model samples for N reference samples and compute/aggregate C2ST per observation. Preserve all ten observation identities, not ten repeats of one observation.', 'bench moons slcp')
check(13, 'fig4b', 'Implement Figure 4(b) for Two Moons, SLCP, Tree and HMM, using 100 reproducibly selected mixed conditional or full-joint queries per task. Generate reference samples with that task\'s N-chain protocol and model samples for the identical free variables and conditions. Evaluate all three attention variants over the same 10**3/10**4/10**5 budget grid; do not replace arbitrary conditionals by posterior-only queries.', 'bench tree hmm moons slcp')
check(13, 'benchmark_results', 'Record actual configurations, checkpoint identities, query definitions, sample counts and per-query/aggregated C2ST values in machine-readable outputs consumed by Figure 4 plotting. Missing computations must be reported as missing, not fabricated curves or constants. Provide runnable code and full scientific defaults; observed superiority over NPE, target C2ST thresholds, calibration and log-likelihood experiments are not Code-Dev approval requirements.', 'bench excluded codedev')

# 014: grid prediction and same trained model for changed observation sets.
check(14, 'lv_configs', 'Configure eight-layer VE Simformers; implement a 10**3/10**4/10**5 budget sweep with dense/undirected/directed attention for Figure 5(c), and use the 10**5 dense model for Figure 5(a,b) illustrative observations. Pass actual metadata-derived structured masks for structured variants; store separate artifacts for each configuration.', 'arch lv_app', 'aa888ef9')
check(14, 'lv_cases', 'Generate one synthetic underlying Lotka-Volterra system and first condition on four irregularly spaced prey observations. The second case retains those same four observations and adds nine irregular predator observations from that same system. Reuse the trained model, expose variable counts/times through the tokenizer, and retain true parameters for comparison; do not resimulate an unrelated truth for the second panel.', 'lv_app')
check(14, 'lv_grid', 'Infer the joint conditional of unknown global parameters and unobserved prey/predator values on a UNIFORM grid from t=0 to t=15, retaining off-grid observed tokens with their exact times. Sample the posterior and predictive trajectories directly from Simformer without extra simulator runs in this predictive route. Interpolating the four observations or only re-running the ODE at estimated parameters does not reproduce that capability.', 'lv_app grid')
check(14, 'lv_compare', 'Compute posterior and arbitrary-conditional C2ST with independent MCMC/target references using the same observation/time tuple and free coordinates. Save posterior marginals, predictive means/uncertainty and ground-truth/observation overlays for both illustrative cases and the quantitative budget sweep. Code must compute these quantities; narrower uncertainty or a published numerical score is not required during static approval.', 'lv_app rf codedev')

# 015: explicit resolution of the main-text/caption death-vs-infected conflict.
check(15, 'sird_dense', 'Use an eight-layer VE Simformer with a fully DENSE attention mask in both Section 4.3 scenarios. Confirm the executed application constructor and sampler actually receive all-real-token attention, rather than inheriting a structured mask default from another task.', 'arch sird_app', 'a0b66551 e0c85d33')
check(15, 'sird_case_a', 'Implement Figure 6(a) using five irregular time-point observations of infected/recovered/deceased populations, with explicit compartment/time identity; support different observation masks/counts through the same interface. Infer global recovery/mortality and the time-varying contact rate and predict missing I/R/D values on the regular 0-to-40 grid. Preserve physical-time metadata instead of treating token position as time.', 'sird_app tokens')
check(15, 'sird_case_b', 'Implement Figure 6(b) by conditioning on FOUR measurements of beta(t) and ONE deceased-population measurement, sampling unobserved beta values, global parameters and data jointly. This diagnostic follows the Figure 6(b) fatalities caption and rubric; the main-body sentence says infected and is inconsistent. Exact observed parameter values must remain fixed during sampling; do not use them merely as initialization.', 'sird_app conditional_loss')
check(15, 'sird_predict', 'Generate the predictive I/R/D trajectories directly through Simformer conditional sampling; save means and 99% uncertainty quantiles for trajectories and beta(t), with true curves and actual measurement points for both scenarios. Plot global-parameter posterior marginals from generated samples. Do not substitute only simulator rollouts or interpolation for Simformer predictions, force uncertainty to a chosen trend, or require the out-of-scope calibration experiment.', 'sird_app excluded codedev')

# 016: energy is a latent random quantity, and threshold provenance matters.
check(16, 'hh_dense', 'Use an eight-layer VE Simformer with fully DENSE attention for Section 4.4 and preserve the seven voltage-statistic coordinates plus the distinct energy coordinate. Model construction, training and both unguided/guided sampling must use the same token/variable convention.', 'arch hh_app hh_stats', '4dce6753 c9544022')
check(16, 'hh_unconstrained', 'First condition only on the observed voltage summary statistics, leaving energy and parameters unobserved. Obtain parameter posterior and energy posterior predictive jointly with Simformer. Also run the implemented HH simulator on corresponding posterior parameter samples through a separately callable evaluation path to compute independent voltage traces and energy predictions for comparison. Both routes must be implemented, not executed during approval.', 'hh_app codedev')
check(16, 'energy_quantile', 'Compute the energy upper bound u as the empirical 0.10 quantile of the UNCONSTRAINED posterior-predictive energy distribution for the same voltage observation. Persist this threshold and expose an explicit configurable override. Do not use a tenth of the mean/max, the prior-energy distribution, an already filtered/guided sample or a hardcoded paper number.', 'hh_app guidance', 'fa934dce')
check(16, 'hh_guided', 'Run the actual task_006 denoised-score/self-recurrence guidance for c(x)=energy(x)-u while exactly holding the voltage summaries fixed. Energy remains a latent variable constrained by an inequality, not an observed value clamped to u. Compute guidance in consistent physical energy units, including any inverse normalization inside the differentiable constraint; simple sample rejection or clipping cannot replace the guided posterior.', 'hh_app guidance algorithm1')
check(16, 'hh_comparison', 'From the guided posterior implement both direct Simformer energy prediction and simulator-based voltage/energy prediction. Save the original threshold, energy distributions and below-threshold fraction, parameter marginals including gNa,gK,Cm,gL, and comparison traces for the unguided/guided cases. Do not fabricate agreement or discard constraint violations solely to claim perfect guidance; the empirical success level is not a static code gate.', 'hh_app codedev')

# 017: final code-level integration catches correct but disconnected helpers.
check(17, 'final_figures', 'Connect saved real experiment outputs to Figures 4-7: benchmark C2ST across budgets/variants, both Lotka-Volterra conditions, both SIRD conditions and unguided/guided HH parameter-energy-trace comparisons. Preserve labels, physical units and uncertainty definitions. Plotting only synthetic example arrays or copied paper numbers does not implement the result workflow.', 'bench lv_app sird_app hh_app')
check(17, 'integration', 'Provide a complete runnable chain from task/budget configuration to simulator data, tokenizer/masks, training, saved checkpoint, conditional sampler, independent references, C2ST and plotting. Check actual call sites and parameter flow, including scalar score shapes, time-versus-metadata features, task-specific N-chain reference protocol, dense application masks and guidance threshold provenance. Correct helpers that are unused while old placeholder paths remain active do not satisfy the workflow.', 'conditional_loss bench lv_app sird_app hh_app')
check(17, 'reproduce', 'Provide reproduce.sh and documented stage-specific/resumable commands that invoke the implemented scientific workflows with their full configurations and preserve distinct outputs. README.md must describe dependencies, source corrections/diagnostic conventions, how modules relate to the paper, and what has or has not run. An import-only script that just prints instructions is not a complete reproduction entry point.', 'project readme codedev')
check(17, 'no_run_gate', 'Assess Code-Dev completion from implemented code and lightweight static/import/interface checks. Do not require real training, long MCMC, full simulator sweeps, generated figures or matching empirical metrics to approve this submission. Keep the addendum exclusions for calibration, log-likelihood and appendix-only guidance experiments; retain Algorithm 1 and appendix methodological details needed for the in-scope main experiments.', 'codedev scope excluded guidance')


def leaves(node):
    if not node.get('sub_tasks'):
        yield node
    for child in node.get('sub_tasks', []):
        yield from leaves(child)


def main():
    parsed = parse_plan(plan)
    assert not is_legacy_outline(plan)
    code = [n for n in leaves(json.loads((PAPER/'rubric.json').read_text()))
            if n['task_category'] == 'Code Development']
    assert len(code) == 92
    assert len({n['id'][:8] for n in code}) == 92
    assert set(MAP) == {n['id'][:8] for n in code}, {
        'missing': sorted({n['id'][:8] for n in code} - set(MAP)),
        'extra': sorted(set(MAP) - {n['id'][:8] for n in code})}
    checks = [c for t in plan['subtasks'] for c in t['checklist']]
    ids = {c['id'] for c in checks}
    assert len(ids) == len(checks)
    for t in plan['subtasks']:
        assert t['checklist']
        assert [c['id'] for c in t['checklist']] == [f"{t['id']}_check_{i:03d}" for i in range(1,len(t['checklist'])+1)]
    for a,b in zip(base['subtasks'],plan['subtasks'],strict=True):
        assert (a['id'],a['name']) == (b['id'],b['name'])
        assert b['description'] == a['description'] + AMENDMENTS.get(a['id'],'')
    for c in checks:
        for s in c['sources']:
            assert s['quote'] in DOCS[s['document']], c['id']
    assert all(set(v) <= ids for v in MAP.values())
    inputs = [BASE, PAPER/'paper.md',PAPER/'paper.pdf',PAPER/'addendum.md',PAPER/'rubric.json',INSTRUCTIONS]
    inputs += [PAPER/'assets'/n for n in ('asset_4.jpg','asset_6.jpg','asset_12.jpg')]
    audit = {
        'kind': 'human-authored rubric-assisted diagnostic; not a blind benchmark result',
        'paper_id':'all-in-one',
        'input_sha256':{str(p.relative_to(LAB)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        'scope_amendments': AMENDMENTS,
        'task_count':len(parsed.subtasks), 'checklist_count':len(checks),
        'task_checklist_counts':{t['id']:len(t['checklist']) for t in plan['subtasks']},
        'schema_valid':True, 'all_quotes_exact_substrings':True,
        'codedev_leaf_count':len(code), 'mapped_codedev_leaf_count':len(code),
        'unmapped_codedev_leaf_ids':[],
        'coverage_meaning':'Manual correspondence, not measured semantic recall or an assurance of grader acceptance. See AUDIT.md for conflicts and rubric-derived details.',
        'coverage':[{'rubric_id':n['id'],'requirement':n['requirements'],
                     'checklist_ids':MAP[n['id'][:8]]} for n in code],
        'semantic_tags':TAGS,
    }
    readable = ['# All-in-one：人工编写、rubric 辅助的诊断清单','',
                'plan.json 是 agent 输入。保留普通 RA 的 17 个任务及顺序；description 的补充与来源冲突见 AUDIT.md。',
                '目标为 Code-Dev 实现检查，不要求现在执行训练或达到论文指标。','']
    for t in parsed.subtasks:
        readable += [f'## {t.id} — {t.name}','',t.description,'']
        for c in t.checklist:
            readable += [f'### {c.id}','',c.criterion,'']
            for s in c.sources:
                readable += [f'> **{s.document}**: '+s.quote.replace('\n','\n> '),'']
    print(json.dumps({'plan':parsed.model_dump(),'audit':audit,'readable':'\n'.join(readable)},ensure_ascii=False))


if __name__ == '__main__':
    main()
