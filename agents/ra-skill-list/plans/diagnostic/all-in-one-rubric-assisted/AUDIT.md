# All-in-one：人工高质量 checklist 的制作与审核记录

日期：2026-09-29。只制作输入文件并进行离线验证，未启动 agent、训练、采样或评分。

## 目的与性质

沿用 mechanistic 手工清单的流程，在普通 RA 原大纲上人工补充具体、可核查的实现要求，观察优质检查清单条件下现有 ra-skill-list 的能力。
制作中读取了完整 Code-Dev rubric、论文正文及相关附录、addendum、官方 Code-Dev 说明、实际图示，以及两次 RA-Skill 的历史失分分析。
这是 **rubric-assisted diagnostic** 输入；不能作为“不看 rubric 自动生成 checklist”的成果，也不能用于证明盲测机制性能。

保留原来的 17 个任务 ID、名称及顺序，增加 **93 个检查项**。8 个任务 description 有明确追加说明，见下表和 coverage.json。
因此这个诊断比较包含清单质量、范围补足和歧义澄清，不能称为严格的“只增加 checklist”单变量消融。

“高质量”在这里指经过来源、任务归属、实现语义、历史错误和 rubric 对照审核；最终得分还取决于执行 agent、checker 与评分器，不能由离线审核保证。

## 文件与输入

- `plan.json`：直接给 ra-skill-list 使用的完整大纲，只有 `subtasks`；任务字段为 `id/name/description/checklist`，检查项为 `id/criterion/sources`，来源为 `document/quote`。
- `CHECKLIST.md`：与 JSON 内容一致的可读版，逐项附上来源原文。
- `coverage.json`：全部 92 个 Code-Dev rubric 叶节点的人工对应关系、来源 SHA256、任务描述追加文本和验证结果。该文件不作为 agent 输入。
- `build_plan.py`：确定性离线构建和验证程序，打印 JSON bundle；不调用模型、不启动容器、不写文件。
- `AUDIT.md`：本记录，说明范围、来源冲突、诊断性选择和覆盖的含义。

原始大纲来自普通 RA 82.38% 那次运行：
`PaperBench-Experiments/all-in-one/2026-09-15T02-55-50Z/agent-run/research_agent/plan.json`。
工作区 `ra-skill-list/plans/all-in-one.plan.json` 与该基线文件字节相同。原大纲及历史实验未修改。

主要资料：

- `PaperBench-Official/project/paperbench/data/papers/all-in-one/paper.md`、`paper.pdf`、`addendum.md`、`rubric.json`。
- `PaperBench-Official/project/paperbench/paperbench/instructions/code_only_instructions.txt`，作为清单中的 `instructions.txt`。
- `assets/asset_4.jpg`：Figure 4/5，直接核查预算横轴、模型变体、任意条件任务和 LV 定量扫描。
- `assets/asset_6.jpg`：Figure 6，直接核查时间范围、参数观测和 fatalities 图注。
- `assets/asset_12.jpg`：Figure A4 与 SLCP 公式，核查实际维度和公式排版。
- `RA_SKILL_CODEDEV_LOSS_ANALYSIS.md` 中 all-in-one 85.77% 与 85.07% 两次失分记录。

对 SLCP 和 Two Moons 的公式歧义，额外查看了论文引用的公共基准的独立实现：
[sbibm SLCP](https://raw.githubusercontent.com/sbi-benchmark/sbibm/main/sbibm/tasks/slcp/task.py) 与
[sbibm Two Moons](https://raw.githubusercontent.com/sbi-benchmark/sbibm/main/sbibm/tasks/two_moons/task.py)。
没有读取 blacklist 中的目标论文实现 `mackelab/simformer`。
sbibm 的 SLCP 实现将 theta3²/theta4² 作为标准差，协方差对角因而是 theta3⁴/theta4⁴；Two Moons 的半径标准差是 0.01。
这些外部证据只用于公开记录的纠错；agent-facing sources 仍严格使用现有三种 document 契约。

## 大纲范围及追加说明

| 任务 | 追加内容 | 原因 |
|---|---|---|
| 002 | 明确 SDE 的量纲和一致性、时间编码归 003，不要求实际运行 | 避免两处独立实现时间编码和中间任务要求训练产物 |
| 003 | 显式记录 time Fourier 256 输出/128 频率、metadata Fourier 128 输出、每头 Q/K/V 10 | rubric 与正文的维数冲突；50 维 token 不能直接套要求整除 4 的 stock MHA |
| 007 | 增补 HMM、联合密度及任意条件接口、四种任务 N-chain 协议 | 原描述只列五种模拟器，但任务 013 已要求 HMM；不能把第六种任务仅偷偷加入 checklist |
| 013 | 明确 Figure 4(b) 四种任务、100 个条件和三个预算；实现而非立即执行 | 补全实验调用流程，并符合 Code-Dev 范围 |
| 014 | 10⁵ illustrative cases；Figure 5(c) 三预算/三 mask 扫描；0–15 网格 | 原描述只有 10⁵，但正文图中的定量实验还包括另外两个预算 |
| 015 | dense、8 层、0–40 网格；四个 beta 观测加一个 D 观测 | 明确应用配置并处理正文与 Figure 6 图注冲突 |
| 016 | dense、8 层、完整 guidance、两种预测路径、可配置阈值 | 防止通用 helper 存在却没有被实际应用脚本调用 |
| 017 | 只要求完整可运行代码与轻量检查，不要求完成训练/采样/图表 | 避免把 code execution/result analysis 转成静态审批的前提 |

所有改动均为追加文本，原始 description 保留；追加的诊断约定在冲突处具有明确解释作用。
基于任务划分反向修订大纲的研究猜想尚未实现；此处仍是人工制作的固定输入。

## 来源冲突与处理原则

不能把 rubric 原文机械复制成不可质疑的验收条款。本次逐项处理如下：

1. **VE perturbation kernel。** rubric 把条件均值写成 xt，还把 sigma 用作方差名称。采用 mean=x0、std=sigma_min*(sigma_max/sigma_min)^t、variance=std²，前向噪声、score target、terminal noise 和反向 drift 保持一致；没有要求实现自指的高斯均值。
2. **VP drift 缺 x。** 原 PDF p18 的公式也省略 x。VP 备选实现采用标准线性 drift=-0.5*beta(t)*x 并匹配其解析 kernel；正文实验仍选 VE，不额外要求复现 appendix-only VP 曲线。
3. **扩散时间 128 vs 256。** PDF p18 明确写 128-dimensional，rubric `5963d717…` 明确要求 256。本诊断选择 128 个频率、sin/cos 拼接后 256 维作为送入线性映射前的特征。这个实现约定满足 rubric，但不能证明作者原文的 128 指频率数。已同时写入任务描述和 criterion，不能伪称它仅由论文得出。
4. **metadata 128、learnable projection 与 C2ST 五折。** 这些细节由 rubric 明确提供，而公开论文/addendum 并未都完整规定。清单要求 metadata Fourier 输出 128 后做可学习映射，C2ST 用 100 棵树和五折验证。引用支持相应方法，但不宣称引用逐字包含所有诊断细化。
5. **mask 代码中的尺寸/方向错误。** addendum 给 Two Moons/Gaussian Mixture 写了 10 个 x，给 SLCP 写了 4 个 theta，Tree 写了 10×10 并采用与其他例子不同的方向。按正文实际维度修正为 2+2、5+8、3+4；保留其意图表达的依赖，并统一 query-row/key-column 约定。不会要求虚构变量来迁就矩阵尺寸。
6. **SLCP 非正定公式。** PDF p19 与 markdown 都将对角写成 theta3²/theta4²，而非对角为 tanh(theta5)*theta3²*theta4²。对 theta3=theta4=2、theta5=2，非对角约 15.42，大于对角 4，矩阵不正定。采用该文所引用 sbibm 的标准 SLCP 协方差：s1=theta3²、s2=theta4²，diagonal=s1²/s2²、off-diagonal=rho*s1*s2；不强迫错误 Cholesky，也不通过大幅裁剪协方差掩盖来源问题。
7. **Two Moons、Tree 与 HMM 记号。** 半径的 0.012 按原 benchmark 的 0.01² 解释，标准差 0.01；Tree 按三参数四观测的 generative factorization；HMM 十个状态只有九次转移，不能根据 i=0,...,9 的印刷索引创建第十一个状态。
8. **SIRD 参数名称。** ODE 用死亡率 mu，prior 段及 rubric 用 delta，统一为同一死亡率的别名；不创造第三个全局常数率。
9. **SIRD 噪声参数化。** 文本称 log-normal mean S(t)、sigma=0.05，但没有无歧义地规定所有库参数的含义。要求对所观测的 S/I/R/D 分量采用一致的 log-normal 模型，明确物理均值与 log-location、数据域标准差与 log-scale 的差别。清单给出“若 0.05 指数据域标准差”的正确换算，并允许清楚说明的另一种解释；不把尚未被证实的解释冒充论文唯一规定。禁止把原始正状态直接送入 log-location 导致均值变成 exp(state)。
10. **SIRD Figure 6(b)。** 正文句子说 infected，图注和 rubric 说 fatalities/death。采用四次 beta 测量加一次 D 测量，显式标注依据。0–40 网格根据 Figure 6 横轴和 rubric；99% 区间来自图注。五个观测时点应保留变量/时间身份，不伪造论文没有给出的精确数值。
11. **HH rate 的 efun 分支。** 原文写 x<1e-4，会把全部负数错误地送入仅在零附近有效的 Taylor 展开。采用 abs(x) 接近零时展开、其余用 x/expm1(x)，并保留六个门控公式的全部参数与移位。
12. **HH central vs standardized moments。** rubric 提及 standardized，但 addendum 明确列出第二至第四 central moments。因此默认保留 addendum 的三个原始中心矩；可额外提供标准化版本，不能替换。能量沿实际钠电流积分，并分别实现累计电荷差分/平滑的 trace 路径及 total charge 的 scalar 路径，避免把已按总时长归一化的每步增量再求平均。
13. **Guidance Algorithm 1 的符号和索引。** 正文 c<=0 配合 log sigmoid(-s*c)，伪代码写了相反号，并有时间索引、循环和末尾重加噪歧义。采用正文的可行域符号；在同一当前时刻计算去噪估计和梯度；完成 reverse 后仅在下一次 recurrence 前 forward；最终输出较低时间状态。这些数值一致性修正不等于允许将 guidance 改成 clipping 或 rejection。
14. **尚未给全的实现数值。** LV Normal prior 的 location/scale、初始种群，HH 完整参数 prior 范围、部分积分设置和 spike detection 约定，并非所提供文本都精确给出。要求合理、显式、可追踪且贯穿调用链的配置；不编造“论文规定”的固定值，也不按某种库函数名字验收。

## 针对历史失分的审核

- ID 对应科学变量，同一变量的多次观测共享可训练 ID embedding；time metadata 和 diffusion time 是两个模块。
- value 必须重复标量；True 是共享可学习条件向量，False 是零；各字段按规定顺序拼接。
- transformer 对每个 token 输出一个标量；禁止以采样时切片、batch/token diagonal 或广播 loss 修补错误输出。
- 4 个 head 每头 Q/K/V 为 10，FF=150，实际每个 block 都加 time projection。
- 时间权重必须在执行的 loss 中；conditioned entries 必须保持干净且不贡献 loss。
- 完整实现 moralization、候选集、min-fill、fill edges、父边方向、候选集更新和最终 union，不只是“图反转”函数名称。
- N 个 joint draws 初始化 N 条链；Two Moons 1000 slice+3000 MH/0.01，SLCP 600 slice+2000 MH/0.1，Tree/HMM 5000 次真正 HMC；每条链只取末样本。
- NPE/NLE 的 NSF、sbi train API、batch=1000、Adam、validation early stopping 必须进入实际代码。内部默认 Adam 经核实可接受；禁止要求库不支持的 validate_data/optimizer 参数。
- SIRD 和 HH 的应用入口强制 dense+8 层，不能仅依赖全局配置默认值。
- HH 阈值来自未加约束的同一观察条件下 energy posterior predictive 的 10% 分位数，同时可显式覆盖；区间约束不能用 exact-value clamping 代替。
- 正确 helper 必须接入实际调用链；科学完整性与“现在跑出结果”分开验收，避免 checker 诱发昂贵长实验。

## Code-Dev 覆盖与验证

人工逐项对应了 **92/92 个 Code Development 叶节点**，完整 requirement/UUID/checklist ID 见 coverage.json。
这表示每个评分要求都有审阅位置，不是经过独立标注计算出的 100% recall，更不意味着 100% 正确或预计满分。
含歧义的条目按上面的解释处理，尤其 256 time、128 metadata、5-fold、SIRD noise/observations、HH moments。

同时检查了正文实验调用流程：Figure 4(a,b)、Figure 5(a,b,c)、Figure 6(a,b)、Figure 7 的无约束和能量约束两条路径。
其他类别中涉及的预算、条件、图表和观测被转换为“提供相应代码与配置”，没有把完成训练或达到论文经验分数设成 checker 必须通过的条件。

| 任务 | 检查项数 |
|---|---:|
| 001 环境配置 | 2 |
| 002 SDE、loss、反向采样 | 7 |
| 003 Transformer | 6 |
| 004 Tokenizer | 7 |
| 005 Graph inversion/masks | 6 |
| 006 Guidance | 5 |
| 007 六种 benchmark 与参考采样 | 13 |
| 008 Lotka-Volterra simulator | 4 |
| 009 SIRD simulator | 5 |
| 010 HH simulator/features/energy | 7 |
| 011 SBI baselines | 4 |
| 012 Training/inference/C2ST | 6 |
| 013 Benchmark workflows | 4 |
| 014 LV application | 4 |
| 015 SIRD application | 4 |
| 016 HH application | 5 |
| 017 Final integration | 4 |
| 合计 | 93 |

离线复核命令（只输出构建 bundle）：

```bash
ra-skill-list/.venv-wsl/bin/python \
  ra-skill-list/plans/diagnostic/all-in-one-rubric-assisted/build_plan.py
```

构建器检查真实 parse_plan 契约、complete/legacy 分类、17 个任务 ID/名称/顺序、每条 description 的精确追加、
全部 checklist ID 的唯一性及任务内连续编号、全部引文在指定文档中的逐字连续匹配、92 个叶节点映射的存在及完整性。
交付时还将磁盘上的 plan.json、coverage.json、CHECKLIST.md 与构建输出逐一比较，避免构建程序正确但交付文件过期。

现有 `ra-skill-list/scripts/run_experiment.sh` 支持这种完整带 checklist 的 plan，后续启动时可直接使用 plan.json。
本次未运行该启动脚本。
