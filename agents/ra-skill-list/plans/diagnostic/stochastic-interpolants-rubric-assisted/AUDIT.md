# Stochastic-interpolants 人工高质量 checklist：设计与审计

## 用途和产物

这是与 mechanistic、all-in-one 手工清单同类的 **rubric-assisted 诊断产物**：用于研究向 agent 提供充分明确的检查条件后，最终代码质量如何变化。制作时查看了 rubric，不能将其运行结果作为“不看 rubric 自动生成清单”的证据。没有启动实验、训练、模型调用或评分。

- `plan.json`：可交给 ra-skill-list“指定已有 checklist 大纲”分支的完整计划。
- `CHECKLIST.md`：便于逐项审阅的版本，包括完整原文引文。
- `coverage.json`：CodeDev 叶节点到检查项的映射、来源性质、基础大纲改动、源文件 SHA-256。
- `build_plan.py`：人工编写的确定性构建与校验程序。只输出 JSON bundle，不写文件、不调用模型。

本版有 **13 个子任务、59 个检查项**。全部 **58 个 Code Development 叶节点**都有明确的审阅位置，其中 **5 个叶节点存在与论文或数学定义的实质冲突**。映射不是评分结果，不是经过独立标注的精确率/召回率，更不等于 58 项均可同时满足。其余 53 项也只是建立了覆盖关系，而非证明未来实现会通过评分。

## 阅读依据

以 `仓库根目录` 为根目录：

1. `PaperBench-Official/project/paperbench/data/papers/stochastic-interpolants/paper.md`：Definition 3.1、§3.1–3.2、Algorithms 1/2、§4.1–4.2、Tables 2/3、Figures 3–6 的说明、Appendix B。
2. 同目录 `addendum.md`、完整 `rubric.json`。
3. `PaperBench-Official/project/paperbench/paperbench/instructions/code_only_instructions.txt`。
4. 基础大纲：`PaperBench-Experiments/stochastic-interpolants/2026-09-17T10-13-57Z/agent-run/research_agent/plan.json`，为普通 RA 大纲，而非 RA-Skill 的再生成大纲。
5. `RA_SKILL_CODEDEV_LOSS_ANALYSIS.md` 中两次 stochastic-interpolants 提交的失分记录。
6. 对 `2026-09-20T13-31-36Z/workspace/submission` 的相关源码作了定点复核：`scripts/sample.py` 的 `tile_size=64`、`class_labels=None`；`training/sampler.py` 的 `mask*mask`；`utils/interpolants.py` 的 tile-width 实现及 bilinear/bicubic 默认值。没有把旧分析中的所有判断不加区分地当作论文事实。

本次引用依据是本地论文 Markdown，不声称另行完成了 PDF 排版/公式图像逐页核对。没有读取被 blacklist 禁止的作者实现。所有 `sources.quote` 都经程序确认是对应本地文档的**逐字连续子串**，不是大纲文字或人工改写。

## 为什么不是简单复制 rubric

### 1. 时间方向：必须消解真正的数学矛盾

Definition 3.1 要求 `I(0)=x0`、`I(1)=x1`；Algorithm 2 从 base `X0` 出发，正向 Euler 积分到数据分布。但 §4.1 写 `alpha=t, beta=1-t`，与这个端点定义相反。rubric 同时要求：

- 三个训练分支的状态为 `I(t)=t*x0+(1-t)*x1`；
- 导数为 `x1-x0`；
- 从 `x0` 出发按正向 Euler 积分。

然而第一式的真实导数是 `x0-x1`。这不是格式差异，也不能靠把导数函数重命名解决。

本版 `task_003_check_002` 明确采用一致的正向约定：

```text
x0 = base；x1 = target
I(t) = (1-t)*x0 + t*x1
dI/dt = x1-x0
X(0) = x0；dX/dt = b(t,X)；输出 X(1)
```

也允许完整一致的反向时间重参数化，但不允许只改其中一式。与 rubric 逐字不一致的三个节点标为冲突：`8c2b4e77`、`3a9c64b6`、`6b23bcf6`。没有添加只为字面评分而设置的错误默认路径或不执行的“兼容公式”。如果评分器坚持相互矛盾的字面要求，这三项仍可能失分；不能承诺满分。

此前失分分析将这类系数方向差异直接列为“非忠实实现”，应加上上述限定。数学自洽问题与 rubric 字面匹配问题必须分开分析。

### 2. SR 条件：固定干净低分辨率图，不泄露未知目标

§4.2 明确将条件定义为 `xi=U(D(x1))`，并把它拼在每次速度模型的图像输入上；加入噪声的是初始 base `x0=xi+sigma*zeta`。rubric 的两个节点 `36ae2f00`、`9d2ac48f` 却说将加噪后的低分辨率图拼到原始 ImageNet 图上，且把这段话用于采样。

如果把该文字机械解释成采样时输入完整未知高分辨率原图，会泄露答案；如果把 noisy base 当固定条件，又与正文的 clean xi 不同。因此 `task_007_check_004` 要求：

- 演化状态是 RGB 图像 `X(t)`；
- 拼接条件为固定 clean `xi`，以及类别信息；
- 输出/积分只处理 RGB 速度；
- 不把未知高分辨率真值作为模型的额外采样条件。

这两个 rubric 节点也标为冲突，不声称逐字覆盖成功。

### 3. rubric 额外规定：采用，但不伪称论文原文

以下细节用于这份诊断计划，但其精确形式并不能全部从论文文本推出：

| 本版约定 | 论文与 rubric 的关系 | 清单处理 |
|---|---|---|
| 空间常数类别值通道 | Appendix B 说使用 class embeddings；rubric 要求 uniform class-value channel | 保留类别条件，并明确额外通道是 rubric-assisted refinement；训练和采样都检查 |
| SR 256→64、512→256 的 crop | 正文只定义 D，没有指定 crop；rubric 指定 cropping | 诊断 profile 采用真实 crop，不把 resize/pooling 说成等价；裁剪位置可配置且须记录，不捏造论文规定的位置 |
| nearest 上采样 | 正文只定义 U；rubric 指定 nearest | 诊断 profile 明确 nearest，并检查训练/采样同一路径 |
| SR sigma=1 | 正文要求 sigma>0，未给固定值；rubric 公式直接加标准高斯 | 诊断 profile 固定 1，注明来源差异 |
| held-out split、FID train reference 的具体样本协议 | rubric 补充了输入 split 和训练参考样本细节 | 提供相应代码路径，不强制现场跑满评测 |

这些项的 `sources` 引用论文里相关的条件/算子描述，不假装该引文足以证明所有 rubric 细节；`criterion` 明说其诊断性质，`coverage.json.check_origins` 标为 `rubric_assisted_protocol`。由于已有结构的 `document` 只允许三个公开文档名，rubric 节点作为旁路审计记录放在 `coverage.json`，没有伪造一个允许的来源文档来装 rubric 原文，也没有为本任务修改任何数据契约。

### 4. 掩码与独立基线的未完全指定部分

明确 `xi=1` 为 observed、`xi=0` 为 missing；64 tiles 是数量而非像素边长；同一位置的 RGB 掩码相同，噪声独立。论文“selected to enter the mask”的文字容易与 observed/missing 两种命名混用，因此要求实现明确其 Bernoulli 选择与 xi 的对应，而不让两处接口隐式反转含义。

Table 2 明确独立基线为 Gaussian base，但没有完整展开该 baseline 的所有额外 observation-conditioning 细节。本版不虚构一个完整基线算法，只锁定独立 base、相同训练目标/预算、纯噪声初始化和一致训练/采样条件。特别禁止在纯噪声初值下继承 dependent 的 known-pixel 零速度约束，从而把噪声当已知像素冻结。

### 5. 数值指标不是实现验收条件

Table 2 的 dependent FID 是 1.13，而 rubric 的 approximate target 是 1.15；Table 3 的 train FID 是 2.13，rubric 为 2.15。报告参考值应准确区分，但 CodeDev checklist 不要求真的训练到这些值。保留非 CodeDev 节点记录，仅要求支持相应训练/评估代码，不把执行结果节点混入 checker 的通过条件。

## 对原大纲的改动

保持全部 **13 个任务的 ID、名称和顺序不变**，仅修正三处描述：

1. `task_005`：梯度裁剪的来源由“addendum”更正为“Appendix B”。
2. `task_006`：把模糊的“masked x1 plus unmasked noise”替换为明确公式和 observed/missing 定义。
3. `task_007`：删去“LR 条件因为出现在两端点所以一直可用”的不充分解释，改为每次显式拼接固定 `U(D(x1))`，且不作为 ODE 演化状态。

逐字替换前后文本保留在 `coverage.json.description_amendments`。没有重分任务、改编号、删除原任务或增加新的科研实验。这些修正意味着它不是完全相同 description 的纯增补；后续做对照时应披露。

## 设计重点

不是只说“实现插值/掩码/SR/FID”，而是给出能够识别替代实现的具体条件：

- 数学量：端点、导数、时间方向、噪声位置、损失归约、固定条件与演化状态。
- 架构：dim/mults、GroupNorm groups、learned sinusoidal、attention head 维度/数量。
- 数据：真实图像 vs mask 本身；train vs held-out；64 tiles vs tile width；同位置跨通道共享 mask。
- 优化：Adam、零 weight decay、1000 更新一次的 StepLR、全参数梯度范数裁剪、32/200000 的生产配置。
- 三条独立调用链：dependent inpainting、uncoupled inpainting、dependent SR，不允许正确 helper 孤立于实际入口。
- 两个阶段：训练与采样分别验收类别通道、条件传递、coupling、分辨率与数据来源。
- 报告：真实 FID 与硬编码值区分、生成输出与 GT 区分、Train/Valid 与缓存统计区分。

共享数学和架构要求集中在负责它们的任务；采样和编排任务再检查入口接线。前面的配置/组件任务不要求未来尚未实现的整条实验已可运行，避免由 checklist 人为引入任务顺序死锁。

## 离线验证

构建器执行以下断言：

1. `parse_plan` 能解析完整大纲；`is_legacy_outline` 为 false，不会被误当作无 checklist 大纲重新生成。
2. 13 个任务保留 ID、名称和顺序，每个任务 checklist 非空，检查 ID 合法且唯一。
3. 所有来源引文均逐字出现在指定文档。
4. rubric 的 58 个 CodeDev 叶节点均有映射，映射不包含不存在的节点；5 个冲突单独标记。
5. 源文件 SHA-256 可用于后续确认用的是同一版本。

复查命令（只验证/输出 artifact bundle，不启动实验）：

```bash
agents/ra-skill-list/.venv-wsl/bin/python agents/ra-skill-list/plans/diagnostic/stochastic-interpolants-rubric-assisted/build_plan.py
```

结构和来源校验不能证明自然语言判据绝对完备，也不能保证 checker 严格执行或保证分数。下一阶段若使用本版诊断，应保留本审计中的五项冲突并单独解读其评分，不能为了声称覆盖率而隐藏它们。
