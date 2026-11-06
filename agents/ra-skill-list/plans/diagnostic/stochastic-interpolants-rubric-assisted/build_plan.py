"""Human-authored, rubric-assisted diagnostic checklist. No LLM or experiment.

Print a deterministic JSON artifact bundle to stdout; this script writes nothing.
Run using ra-skill-list/.venv-wsl/bin/python. Read AUDIT.md before using the plan.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
LAB = ROOT.parent
PAPER = LAB / 'PaperBench-Official/project/paperbench/data/papers/stochastic-interpolants'
BASE = LAB / 'PaperBench-Experiments/stochastic-interpolants/2026-09-17T10-13-57Z/agent-run/research_agent/plan.json'
INSTRUCTIONS = LAB / 'PaperBench-Official/project/paperbench/paperbench/instructions/code_only_instructions.txt'
sys.path.insert(0, str(ROOT / 'src'))
from research_agent.plan import parse_plan
from research_agent.plan_enrichment import is_legacy_outline

DOCS = {n: (PAPER / n).read_text() for n in ('paper.md', 'addendum.md')}
DOCS['instructions.txt'] = INSTRUCTIONS.read_text()
SOURCES = {}


def quote(key, document, text):
    assert text in DOCS[document], (key, 'non-verbatim source')
    SOURCES[key] = {'document': document, 'quote': text}


def span(key, start, end):
    text = DOCS['paper.md']
    a = text.index(start)
    b = text.index(end, a + len(start))
    quote(key, 'paper.md', text[a:b].strip())


def para(key, document, marker):
    matches = [p.strip() for p in DOCS[document].split('\n\n') if marker in p]
    assert len(matches) == 1, (key, len(matches))
    quote(key, document, matches[0])


span('definition', 'Definition 3.1 (Stochastic interpolant', 'A simple instance')
span('linear', 'In this context, we can use the interpolant', '\\subsection*{3.3.')
span('train', 'Algorithm 1 Training', '```\n```')
span('euler', 'Algorithm 2 Sampling', '```\nperform')
span('mask', 'For simplicity, the mask takes', 'In our experiments,')
span('fixed', 'The model sees the mask;', 'Results. For implementation,')
span('mask_channels', 'Results. For implementation,', 'Additional specific experimental details')
span('sr', 'tion 4.1. We denote', 'Results. Similarly')
span('sr_channels', 'Results. Similarly to the previous experiment,', 'Samples are displayed')
span('table2', 'Table 2: FID for Inpainting Task.', 'Table 3:')
span('table3', 'Table 3: FID-50k', 'tion 4.1. We denote')
span('arch', 'Architecture For the velocity model', 'Image-shaped conditioning')
para('cond', 'paper.md', 'Image-shaped conditioning in the Unet.')
para('optim', 'paper.md', 'Optimization. We use Adam optimizer')
quote('dopri', 'paper.md', 'Integration for sampling We use the Dopri solver from the torchdiffeq library (Chen, 2018).')
para('fig3', 'paper.md', 'Figure 3: Image inpainting:')
para('fig4', 'paper.md', 'Figure 4: Super-resolution:')
para('fig5', 'paper.md', 'Figure 5: Additional examples')
para('fig6', 'paper.md', 'Figure 6: Super-resolution:')
quote('batch', 'addendum.md', 'All experiments should use a batch size of 32.')
quote('steps', 'addendum.md', 'All experiments should run for 200,000 gradient steps')
quote('groups', 'addendum.md', 'The "Resnet block groups" that is discussed in the appendix is the number of groups that goes into the GroupNorm in the ResNet layers (their ResNet blocks use GroupNorm)')
para('hf', 'addendum.md', 'You should download ImageNet using HuggingFace.')
para('excluded', 'addendum.md', 'You do not have to replicate the baseline results')
para('codedev', 'instructions.txt', 'The code will not be executed during grading.')
para('readme', 'instructions.txt', 'Finally, please also include a README.md')
para('submission', 'instructions.txt', 'The output of your work must be a git repository')
para('scope', 'instructions.txt', 'Any experiments that are only introduced in the Appendix')

original = json.loads(BASE.read_text())
plan = copy.deepcopy(original)
for task in plan['subtasks']:
    task['checklist'] = []

# Correct three misleading statements, keeping all task IDs, names and order.
amendments = []


def replace_description(number, old, new):
    task = plan['subtasks'][number - 1]
    assert old in task['description']
    task['description'] = task['description'].replace(old, new)
    amendments.append({'task_id': task['id'], 'old': old, 'new': new})


replace_description(5, 'gradient clipping as specified in the addendum',
                    'gradient clipping as specified in Appendix B')
replace_description(6, 'x0 equals masked x1 plus unmasked noise',
                    'x0 = xi*x1 + (1-xi)*zeta, where xi=1 denotes observed pixels and xi=0 denotes missing pixels')
replace_description(7, 'The low-resolution image is available at all interpolant times because it appears in both x0 and x1.',
                    'The fixed upsampled low-resolution condition xi=U(D(x1)) is explicitly appended to the evolving image at every model call; it is not part of the evolved ODE state.')

links = {}
origins = {}


def add(number, criterion, sources, rubric='', origin='paper_or_addendum'):
    task = plan['subtasks'][number - 1]
    check_id = f"{task['id']}_check_{len(task['checklist']) + 1:03d}"
    task['checklist'].append({'id': check_id, 'criterion': criterion,
                              'sources': [copy.deepcopy(SOURCES[x]) for x in sources.split()]})
    origins[check_id] = origin
    for prefix in rubric.split():
        links.setdefault(prefix, []).append(check_id)


# 001: contracts/configuration, not execution of expensive experiments.
add(1, 'Provide importable, version-compatible dependencies for the referenced U-Net, PyTorch, torchdiffeq, HuggingFace ImageNet access and FID. The actual velocity-model import must resolve to the intended implementation, not a similarly named placeholder or a silent tiny-network fallback.', 'arch dopri hf', 'fa71af11')
add(1, 'Provide explicit production configurations for dependent inpainting, uncoupled inpainting and dependent super-resolution. Every production run uses batch size 32 and 200000 optimizer updates; small/debug settings must be opt-in and must not replace production defaults. Check configuration consumption, not merely the presence of constants.', 'batch steps excluded', '7de84bdb 7c7ebd70')
add(1, 'Separate image state, observation mask, fixed low-resolution condition, class label and time in the data/model interfaces. Document their shapes and the xi=1 observed / xi=0 missing convention so training and sampling cannot silently swap mask meanings or evolve conditioning channels.', 'mask cond sr')
add(1, 'This is a rubric-assisted CodeDev diagnostic: implement the full production paths, but do not require executing training or 50000-image sampling to complete a task. Implementation checks must follow callable code and configuration; prose, stub functions and cached claimed metrics are not implementations.', 'codedev')

# 002: dataset and transformations.
add(2, 'Implement authenticated HuggingFace imagenet-1k loading with genuine train and validation split access and labels preserved. Follow the addendum remote-code handling when applicable to the selected compatible datasets version; a missing dataset must not silently select synthetic data or the wrong split.', 'hf', '46ca1cd4')
add(2, 'Provide RGB ImageNet preprocessing at both 256x256 and 512x512, with explicit consistent normalization and inverse conversion for visualization/FID. Keep the initial ImageNet resolution preparation distinct from the later task-specific low-resolution degradation.', 'fig3 fig4 fig6')
add(2, 'Training receives training images; held-out dependent sampling receives validation images with their actual observations and labels. Split, sample identity, mask and class must stay aligned through batching; validation must never silently alias the training set.', 'hf fig3', origin='rubric_assisted_protocol')

# 003: mathematical coherence, explicitly resolves source inconsistency.
add(3, 'Construct the general interpolant I(t)=alpha(t)*x0+beta(t)*x1+gamma(t)*z with analytic derivatives and Definition 3.1 endpoint conditions. Broadcast a separate time per sample without mixing batch and image axes; any general noise term must be independent of the coupled endpoints.', 'definition train')
add(3, 'For the image experiments use gamma=0 and a coherent forward convention: x0 is the base, x1 the target, I(t)=(1-t)*x0+t*x1, dI/dt=x1-x0, and integrate from t=0 to t=1. Section 4.1 instead prints alpha=t,beta=1-t, inconsistent with Definition 3.1 and Algorithm 2. A wholly consistent reversed-time reparameterization is acceptable, but mixing its state formula with the forward derivative/sampler is not. Document the conflict rather than enforcing contradictory formulas.', 'definition fixed euler linear', '8c2b4e77 3a9c64b6 6b23bcf6 2326b8d4 44d89bd8 5aa12bc7', 'source_conflict_resolution')
add(3, 'Uniformly sample t_i in [0,1] independently for every minibatch element in all three training modes. The actual mode-specific training calls must use this continuous-time draw, not one fixed time, a shared batch scalar or diffusion-index sampling.', 'train', 'ee1e4da3 d5402a39 d790efdd')
add(3, 'Image-experiment noise belongs to the construction of x0; do not also add an unrequested nonzero gamma(t)*z to the linear experimental bridge. Dependent and independent endpoint constructors remain distinct and are selected by the actual experiment mode.', 'linear train table2')

# 004: architecture and conditioning contract.
add(4, 'Use the Appendix B lucidrain denoising-diffusion-pytorch U-Net velocity architecture, or a faithful explicit implementation of that variant, with its multiscale residual/attention/skip structure. A generic smaller CNN, similarly named U-Net or diffusion noise-predictor wrapper is not a substitute for the stated velocity model.', 'arch', '748d82fe')
add(4, 'Instantiate the production U-Net with dim=256 and dim_mults=(1,1,2,3,4), not repository defaults. Verify the instantiated channel hierarchy, not only config declarations.', 'arch', 'ba327b00 0bd8d434')
add(4, 'Use eight GroupNorm groups in the ResNet blocks, not eight blocks or eight channels; preserve valid divisibility in all actual residual blocks.', 'arch groups')
add(4, 'Use learned sinusoidal time conditioning with learned_sinusoidal_cond=True, learned_sinusoidal_dim=32 and random_fourier_features=False. The real model forward must consume this trainable time representation.', 'arch', '9c00369e 33409f2f eaf72173')
add(4, 'Use attention head dimension 64 and four heads in the specified U-Net attention modules; configuration overrides must reach the modules.', 'arch', 'c4401ce0 f0329e30')
add(4, 'Append the inpainting observation mask and the super-resolution fixed upsampled low-resolution image as spatial input channels in their respective modes at every training and sampling call. Output only C image-velocity channels with the original spatial resolution, never velocities for mask, label or low-resolution channels.', 'cond sr_channels fixed', 'b3d613a2')
add(4, 'For this rubric-assisted diagnostic, retain class conditioning and also append one HxW channel uniformly filled with each sample\'s ImageNet class value in all class-conditioned training and sampling paths. A learned embedding alone does not meet this diagnostic channel contract. Appendix B supports embeddings; the additional constant-value channel is a rubric-specified refinement, not a verbatim requirement of Appendix B.', 'arch sr_channels', '75cc97d2 3fd598f3 f823b47a f118676d 3c8fd9d0', 'rubric_assisted_protocol')

# 005: actual training semantics.
add(5, 'The minibatch update must construct paired endpoints from the selected coupling, sample per-example times, evaluate the coherent interpolant and its derivative, call the actual velocity model with the matching conditions, backpropagate and update parameters. All three mode-specific training entry points must reach this update, not an unused correct helper.', 'train table2')
add(5, 'Compute the empirical quadratic velocity objective mean_i[sum_pixels,channels(b_hat^2)-2*sum_pixels,channels(dI_dt*b_hat)] over image channels only. Squared-error regression is acceptable only with the same parameter gradients/reduction (its target-norm term is parameter independent); do not substitute image reconstruction, score/noise prediction, or omit the cross term. Check every training mode.', 'train', 'cbe7cd9e 4730f7b6 ff8dbe4e')
add(5, 'Use Adam with initial learning rate 2e-4 and zero weight decay for dependent inpainting, independent inpainting and super-resolution; AdamW defaults or weight decay on any active parameter group do not satisfy this requirement.', 'optim', '8190541a 74e1ae29 75958462 cd04efc2')
add(5, 'StepLR must multiply the learning rate by 0.99 after every 1000 optimizer updates, across 200000 updates. Do not step it on every minibatch as if one step were 1000, on each epoch, or per accumulated microbatch; resume must preserve its update count.', 'optim steps', '67af86d8')
add(5, 'Clip the global gradient norm of the entire velocity model parameter vector at 10000 before optimizer.step; do not clip parameter values, each tensor independently, or use a different threshold.', 'optim', 'cb0f23e1')
add(5, 'The production loop must actually honor batch size 32 and 200000 gradient updates, cycle the dataset as needed, and save model/config/optimizer/scheduler/update-counter state so sampling loads the matching learned architecture and a resumed run does not restart its schedule.', 'train batch steps', '7de84bdb 7c7ebd70')

# 006: dependent inpainting.
add(6, 'At both training and dependent sampling, create exactly 64 equal tiles (an 8x8 grid for these square images), with independent Bernoulli choices per tile and per sample at p=0.3. Define xi as the observed-pixel indicator and document how selected tiles map to it. Never interpret 64 as tile width, which yields only 16 tiles on 256x256 images.', 'mask', 'e04c1c97 4b3e6602')
add(6, 'The spatial observation mask is identical across RGB channels for each sample, whereas missing-pixel Gaussian noise is independently drawn across channels and pixels. Apply the same convention in training and sampling.', 'mask', '036df75d c6ba9100')
add(6, 'Construct dependent training x0=xi*x1+(1-xi)*zeta from the actual ImageNet training image, with standard-normal zeta. Known pixels equal x1 exactly; missing pixels are noise. Do not use xi*xi, replace the observation with zeros, or use an entirely independent noisy image.', 'mask', '551ad22d')
add(6, 'Build the dependent image interpolant so observed pixels remain exactly fixed at every time and their analytic target velocity is zero. Mask the predicted image velocity with (1-xi) during loss computation AND sampling; masking only the regression target is insufficient. No replacement/MCMC correction may substitute for this structural velocity constraint.', 'fixed mask', '718205ed')
add(6, 'The dependent training model must receive the same sample\'s mask, class conditioning and diagnostic constant class channel, while predicting only image velocity. Mask and label channels are fixed conditions, not interpolated targets.', 'mask_channels arch', '75cc97d2', 'rubric_assisted_protocol')

# 007: SR. Do not falsely attribute crop/nearest/sigma choices to the paper.
add(7, 'For this rubric-assisted diagnostic profile, construct low-resolution training observations from the prepared 256x256 image by a genuine 64x64 crop, and from 512x512 by a 256x256 crop. Make crop placement deterministic/configured and documented (the rubric does not specify its location). This crop protocol refines the paper\'s unspecified D operator; resize, average pooling or crop after already resizing to the low resolution is not this diagnostic operation.', 'sr', '74ebafd7', 'rubric_assisted_protocol')
add(7, 'Use nearest-neighbor interpolation to upsample those low-resolution observations back to 256x256 or 512x512 respectively, in both preparation and sampling. For this rubric-assisted profile nearest is fixed explicitly; the paper itself denotes only U. Bilinear or bicubic paths must not silently be the production default.', 'sr', 'af9c0ad4 52dab287', 'rubric_assisted_protocol')
add(7, 'Construct super-resolution x0=U(D(x1))+sigma*zeta using independent standard Gaussian image noise. Use sigma=1 in the diagnostic production profile to match the rubric; retain its explicit value in configuration. The paper permits sigma>0 and does not set it to one. Do not initialize from pure noise or omit the nonzero noise.', 'sr', 'f701ec5c 3a59ebee', 'rubric_assisted_protocol')
add(7, 'Keep xi=U(D(x1)) as a fixed, clean upsampled low-resolution condition; append it to the evolving image I(t)/X(t) at every velocity evaluation, with class conditioning and the diagnostic constant class channel. Evolve only the image channels. The paper explicitly specifies clean xi; do not silently replace it by the noisy base or leak the unknown high-resolution target into sampling. The rubric\'s noisy-appending wording conflicts with this paper specification.', 'sr sr_channels cond', '36ae2f00 9d2ac48f b3d613a2', 'source_conflict_resolution')
add(7, 'Use identical degradation/upsampling, class encoding, image normalization and condition layout at training and inference for each resolution pair. Verify that the SR training call receives the constant class channel; implementing it only in a generic model adapter is insufficient.', 'sr_channels arch', 'f118676d', 'rubric_assisted_protocol')

# 008: inference and real entry-point wiring.
add(8, 'Implement Algorithm 2 Euler updates X[n+1]=X[n]+b(n/N,X[n])/N for exactly N steps, n=0,...,N-1, returning X[N]. The state starts at the selected base and reaches t=1 under the forward convention; time conditioning must follow n/N. Provide working dependent-inpainting, uncoupled and SR dispatches, not just a generic unused loop.', 'euler definition', '35183fae e9d03723 547af5aa')
add(8, 'Use the torchdiffeq Dopri solver (dopri5) for the Appendix B production sampling path, with the trained velocity RHS and the same state/condition semantics as Euler. Do not label Euler as Dopri, silently fall back on import failure, or introduce an unrequested stochastic/SDE sampler.', 'dopri euler', 'ebc3e8cf')
add(8, 'Dependent inpainting inference must load actual held-out ImageNet validation/test observations, create the specified 64-tile shared-channel mask, and initialize X0=xi*x_observed+(1-xi)*zeta. Trace the CLI through loader, constructor and sampler: a correct helper that is not called does not satisfy this item. Never use the binary mask itself as the image or substitute training images.', 'mask euler', '4b3e6602 c6ba9100 d9883334', 'rubric_assisted_protocol')
add(8, 'Inpainting sampling must pass the corresponding real class labels, diagnostic uniform class-value channel and mask at every velocity evaluation. Multiply image velocity by the missing-region mask so observed pixels remain fixed throughout integration, not merely pasted back after generation.', 'fixed mask_channels arch', '3fd598f3', 'rubric_assisted_protocol')
add(8, 'The held-out SR sampling entry point must obtain real validation/test images and derive the diagnostic 256-to-64 or 512-to-256 crop, nearest upsampling, noisy base and clean fixed condition consistently with training. The separate training-reference FID mode is explicitly named and must not replace this held-out path.', 'sr fig4 fig6', '9108d07c 52dab287 3a59ebee', 'rubric_assisted_protocol')
add(8, 'SR sampling receives the corresponding class labels and diagnostic constant class-value channel; the model sees evolving RGB state plus fixed low-resolution/class conditions, and outputs only RGB velocity. Unknown high-resolution pixels are not an extra inference input. Preserve conditions unchanged throughout all solver calls.', 'sr_channels cond', '3c8fd9d0 b3d613a2', 'rubric_assisted_protocol')
add(8, 'Load the selected trained checkpoint/config in evaluation mode and disable gradients for inference without detaching training-time predictions. Sampling outputs must be the solver endpoint, not the reference image, a nearest-upsample-only proxy or an unrelated cached tensor.', 'euler fig3 fig4')

# 009: evaluators, protocols, not demands for achieved scores.
add(9, 'Implement genuine Fréchet Inception Distance using a declared standard pretrained Inception feature extractor, empirical means/covariances and a numerically sound covariance square-root term. Correctly convert model image ranges to the extractor\'s expected preprocessing. A random/untrained feature network, pixel MSE or hard-coded scalar is not FID.', 'table2 table3', '190a2c3f')
add(9, 'Provide FID-50k evaluation paths for both independent and dependent inpainting against the held-out ImageNet validation reference. Support exactly 50000 generated/reference samples with streaming/batched features, including partial final batches, and distinguish model outputs from source/reference images.', 'table2', origin='rubric_assisted_protocol')
add(9, 'For 64-to-256 SR, implement separately labeled Train and Valid FID-50k reports; the diagnostic train reference uses 50000 selected training samples and validation uses its own reference statistics. Cache keys must distinguish split, resolution and feature-extractor settings. Do not reuse train statistics as validation or compare references with themselves.', 'table3', origin='rubric_assisted_protocol')
add(9, 'Evaluation must call the selected experiment\'s actual checkpoint/sampler with the matching coupling and resolution. Report measured values only when computed; paper values are labeled reference values. Achieving published FIDs or running expensive evaluation is not a CodeDev completion requirement.', 'codedev table2 table3')

# 010: uncoupled baseline must genuinely differ.
add(10, 'In independent-baseline training, draw x0~N(0,Id) independently of x1 over the entire RGB image. Do not reuse the dependent observed-pixel/noise constructor, initialize from U(D(x1)), or change only an experiment label while retaining dependent coupling.', 'table2 definition', '3f74c905')
add(10, 'The independent baseline uses per-sample uniform times, the same coherent image bridge/analytic derivative and quadratic image-velocity loss as task_003/task_005, optimized with Adam. The class label must reach both the class conditioning and the diagnostic uniform class channel, not remain a disconnected dataloader field.', 'train arch table2', 'f823b47a d5402a39 44d89bd8 4730f7b6 74e1ae29', 'rubric_assisted_protocol')
add(10, 'Independent sampling starts from pure full-image Gaussian noise, then integrates its own trained velocity forward using the implemented Euler/Dopri path. It must not inherit dependent initialization or a known-pixel zero-velocity constraint, which would freeze initially noisy pixels. Any additional observation-conditioning policy for this underspecified baseline must be explicit and consistent between training and sampling, not a disguised dependent base.', 'table2 euler', '5afccb74 e9d03723')
add(10, 'Maintain separately identified dependent and uncoupled checkpoints/results with the same stated backbone, optimizer schedule, batch size and training budget. Implement the baseline of Table 2, but do not require retraining Improved DDPM, SR3, ADM, Cascaded Diffusion or I2SB, which the addendum excludes.', 'table2 excluded arch optim batch steps')

# 011: whole-pipeline reachability and reproducibility.
add(11, 'Wire actual runnable experiment entry points for dependent inpainting at 256 and 512, uncoupled inpainting baseline, and dependent SR at 64-to-256 and 256-to-512. The main configurations must select the correct coupling, loader, model channels, trainer, sampler and evaluator; detached demonstration helpers do not constitute an end-to-end path.', 'fig3 fig4 fig6 table2')
add(11, 'Resolve CLI/config choices once and propagate them consistently across training, checkpoint save/load, sampling and evaluation: architecture, class encoding, resolution, mask convention, interpolation orientation and low-resolution degradation. Production must not silently replace these with hard-coded sampling defaults.', 'train euler cond')
add(11, 'Allow separate training/sampling/evaluation commands and checkpoint resume, with required inputs and failures handled explicitly. Do not automatically run real training or large-image inference merely to satisfy CodeDev checks; keep production implementation intact without presenting synthetic smoke output as experimental evidence.', 'codedev readme')

# 012: faithful output representation, not pixel-identical targets.
add(12, 'Generate inpainting triplets from the same example: corrupted/base input, actual integrated model sample, and original reference, for 256 and 512. Missing-region output need only be conditionally plausible, not identical to the reference; do not reject the method solely for not copying the target.', 'fig3')
add(12, 'Generate SR triplets at both main-text resolution pairs with true low-resolution input, model output and high-resolution reference in the correct order; preserve their sample correspondence and show real generated outputs rather than just nearest/bicubic enlargement.', 'fig4 fig6')
add(12, 'Provide temporal-slice plotting from states of one genuine inpainting trajectory as in Figure 5, not independent random images or a hand-drawn interpolation. This is a visualization utility, not a requirement to train a new appendix-only experiment.', 'fig5 scope')
add(12, 'Populate Tables 2/3-style reports from measured evaluation artifacts with split/coupling labels; distinguish paper reference numbers from measurements and unavailable values. The paper reports dependent inpainting 1.13 and SR train/valid 2.13/2.05; do not silently reinterpret the rubric\'s approximate 1.15/2.15 as new measured results.', 'table2 table3', origin='source_conflict_resolution')

# 013: final submission and explicit scientific decisions.
add(13, 'README must map implemented files and commands to the two image tasks, baseline, Algorithms 1/2, Appendix B architecture/optimization, and Tables 2/3. Explicitly document the forward-time inconsistency resolution and which crop/nearest/class-channel/sigma choices belong to this rubric-assisted diagnostic rather than being quoted paper facts.', 'readme definition fixed sr arch')
add(13, 'Provide a real reproduce.sh that connects data access, production configurations, training, checkpoint-based sampling and evaluation without embedding unrelated workspace paths. It must expose required data credentials/cache handling without containing secrets, and must not silently skip failed components or replace their results.', 'submission readme hf')
add(13, 'Package all necessary source/configuration files and declared dependencies under the submission repository, with README and entry scripts included in the tracked source; keep committed content below 1GB and exclude downloaded ImageNet, large model artifacts and credentials. CodeDev grading does not require running the packaged full experiments.', 'submission codedev')


def leaves(node):
    if not node.get('sub_tasks'):
        yield node
    for child in node.get('sub_tasks', []):
        yield from leaves(child)


def bundle():
    validated = parse_plan(plan)
    assert not is_legacy_outline(plan)
    assert len(validated.subtasks) == len(original['subtasks']) == 13
    for old, new in zip(original['subtasks'], plan['subtasks']):
        assert (old['id'], old['name']) == (new['id'], new['name'])
        assert new['checklist']
    for task in plan['subtasks']:
        for check in task['checklist']:
            for source in check['sources']:
                assert source['quote'] in DOCS[source['document']]
    all_leaves = list(leaves(json.loads((PAPER / 'rubric.json').read_text())))
    dev = [n for n in all_leaves if n.get('task_category') == 'Code Development']
    assert len({n['id'][:8] for n in all_leaves}) == len(all_leaves)
    assert set(links) == {n['id'][:8] for n in dev}, (set(links) - {n['id'][:8] for n in dev}, {n['id'][:8] for n in dev} - set(links))
    conflicts = {'8c2b4e77', '3a9c64b6', '6b23bcf6', '36ae2f00', '9d2ac48f'}
    rows = [{'rubric_id': n['id'], 'requirement': n['requirements'],
             'check_ids': links[n['id'][:8]],
             'status': 'mapped_but_conflicts_with_paper_or_mathematics' if n['id'][:8] in conflicts else 'mapped_for_review'} for n in dev]
    coverage = {
        'purpose': 'Human-authored rubric-assisted diagnostic, not rubric-blind evaluation.',
        'mapping_is_not_a_score_or_proof_of_semantic_recall': True,
        'task_count': len(plan['subtasks']), 'check_count': sum(len(t['checklist']) for t in plan['subtasks']),
        'codedev_leaf_count': len(dev), 'conflicted_leaf_count': len(conflicts),
        'base_plan': str(BASE.relative_to(LAB)),
        'source_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in {
            'base_plan': BASE, 'paper.md': PAPER / 'paper.md', 'addendum.md': PAPER / 'addendum.md',
            'rubric.json': PAPER / 'rubric.json', 'instructions.txt': INSTRUCTIONS}.items()},
        'description_amendments': amendments, 'check_origins': origins, 'codedev_mapping': rows,
        'non_codedev_leaves': [{'id': n['id'], 'category': n['task_category'], 'requirement': n['requirements'],
                               'handling': 'Provide code pathways where relevant; actual execution and achieved scores are not acceptance conditions.'}
                              for n in all_leaves if n['task_category'] != 'Code Development']}
    md = ['# Stochastic-interpolants：人工 rubric-assisted checklist', '',
          '> 用于诊断高质量清单的效果；使用了 rubric，不能作为 rubric-blind 方法的结果。完整来源/冲突说明见 AUDIT.md。', '',
          f"13 个任务，{coverage['check_count']} 个检查项；{len(dev)} 个 CodeDev 叶节点均有审阅位置，其中 5 项有实质来源冲突，不能称为完全满足 rubric。", '',
          '保留普通 RA 的任务 ID、名称、顺序；修正三处描述，逐字差异见 coverage.json。每项检查须追踪实际代码调用路径，不以注释或未接线 helper 代替实现。', '']
    for task in plan['subtasks']:
        md += [f"## {task['id']} — {task['name']}", '', task['description'], '']
        for check in task['checklist']:
            md += [f"### {check['id']}", '', check['criterion'], '', f"来源性质：`{origins[check['id']]}`", '']
            for source in check['sources']:
                md += [f"**{source['document']}**", '', '```text', source['quote'], '```', '']
    return {'plan.json': json.dumps(plan, ensure_ascii=False, indent=2) + '\n',
            'coverage.json': json.dumps(coverage, ensure_ascii=False, indent=2) + '\n',
            'CHECKLIST.md': '\n'.join(md)}


if __name__ == '__main__':
    print(json.dumps(bundle(), ensure_ascii=False))
