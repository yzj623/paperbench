# Mechanistic：高质量 checklist 诊断大纲审核说明

日期：2026-09-27。实验尚未启动。此目录只包含离线编写与验证的输入文件。

## 用途和实验性质

目标是考察在人工精修、充分覆盖的 checklist 条件下，现有 ra-skill-list 主 agent 和 checker 的表现。
本大纲编写时主动阅读了 rubric 和历史失分材料，属于 **rubric-assisted diagnostic**，不是 blind benchmark 结果，也不证明自动 planner 能生成同样质量的清单。
“高质量”指经过来源、覆盖、语义和历史错误针对性审核；不是已通过实际实验验证的性能结论。

本次保留普通 RA 原大纲的 12 个任务 ID、名称及顺序，增加 67 项 checklist，并对三个任务描述作范围补充。
因此后续比较同时包含“检查清单质量”和“明确遗漏工作的归属”两项变化，不能称为严格的仅新增 checklist 文本的单变量消融。

## 文件

- `plan.json`：agent 输入，严格使用现有 `subtasks/id/name/description/checklist` 契约；检查项只有 `id/criterion/sources`，来源只有 `document/quote`。
- `CHECKLIST.md`：同一大纲的人类可读完整版本，包含全部检查项及原文引用。
- `coverage.json`：36 个 Code-Dev rubric 叶节点与检查项的人工映射、输入 SHA256、范围补充、验证结果。它是研究审计文件，不是 agent 输入。
- `build_plan.py`：确定性离线构建与验证程序，只向 stdout 输出 JSON；不会启动模型、容器、训练或评分，不会写文件。

普通 RA 原始大纲：
`PaperBench-Experiments/mechanistic-understanding/2026-09-13T04-33-18Z/agent-run/research_agent/plan.json`。
原文件和历史提交区均未修改。

来源为本地 `PaperBench-Official/project/paperbench/data/papers/mechanistic-understanding/` 的 paper.md、addendum.md、rubric.json，
及官方 `paperbench/instructions/code_only_instructions.txt`。计划中的 instructions.txt 引文来自该公开任务说明。
输入版本的精确哈希见 coverage.json。

## 范围归属：为什么补充三个 description

现有 checker 明确以冻结的任务 description 判断范围。仅在 checklist 中附加范围外要求，容易被判为 checklist_error。
因此范围补充是显式的，而不是把实验偷偷塞进原有检查项。

| 任务 | 显式补充的职责 | 依赖处理 |
|---|---|---|
| task_006 | Section 6 / Table 4 的全局七向量选择、key ×10 干预、真实评估入口 | 可以接受后续 DPO checkpoint 和 metric 接口作为参数；当前不要求运行或已有训练产物 |
| task_010 | Table 3 的具体 prompts、next-token 排名及三种模型条件的续写 | 复用干预接口与 DPO checkpoint |
| task_012 | Section 5.1 的原始参数与 DPO 参数比较；整套流程和必要提交入口的最终集成 | 在前序功能齐备后检查调用链及输入输出关联 |

Table 2 的指标与行集成归 task_007，具体减法干预归 task_006。
Fig. 1 / Table 3 归 task_010；Fig. 2 / Fig. 4 归 task_011；Fig. 5 与 Section 5.1 分开列项后归 task_012。
同一机制被不同实验复用时，分别检查底层运算与对应实验入口，而不要求无关子任务提前执行后续实验。

## 质量审核中作出的关键区分

1. **公式矩阵与库权重存储不同。**
   Probe 允许 PyTorch Linear 的转置存储，但要求显式导出/解释 [d_model,2] 的 W_Toxic 和 class-column 对应。
   GPT2 Conv1D 的 c_fc/c_proj 则按实际 [输入,输出] 存储取列/取行，不能机械照搬论文矩阵注释。
   后者是工程实现推导，不伪称论文直接规定了 HuggingFace 属性名称。
2. **三个 global top-k 必须分别落实。**
   SVD 输入为 global 128；Fig. 2 为 original GPT2 global 5 并固定两模型的 neuron IDs；
   Table 4 为 global 7 对应 keys。检查实际传入调用链的集合，不能只声明可配置的 k。
3. **Section 5.1 与 Fig. 5 不是同一统计量。**
   前者计算 cos(theta_original,theta_DPO) 和对应参数差范数；
   后者计算 cos(delta_v[j,i],delta_x[19])。禁止以前者与 delta 的混合量替代。
4. **目标 token 明确。**
   rubric 将论文脱敏显示 sh*t 明确为 GPT2 的单 token `" shit"`（有前导空格）。
   这项知识来自 rubric 辅助，不声称 public paper 的脱敏拼写已充分规定 tokenizer 字节。
   sources 保留支持实验目的的 public paper 引文；这一诊断性澄清在此披露。
5. **两个 RTP 子集不能混用。**
   激活/毒性使用 1,199 challenge prompts；Fig. 1 按原模型目标 token argmax 筛选 295 prompts。
   论文没有说后者一定是 challenge 的子集，不增加该限制，也不能伪造/重复记录凑数。
6. **配对 forward 与真实激活。**
   original GPT2 greedy 生成的相同 token 轨迹用于两模型对比，保留位置和样本身份；
   GeLU 后系数来自实际 MLP 计算，不用未经模型归一化/偏置的 residual 点积冒充。
7. **Fig. 5 固定 layer 19 reference。**
   展示 source layers 0,2,...,18 的所有 neurons，而不是每层自比或只比较 toxic subset。
   橙色是各 neuron 平均激活的分布，不是按 cosine 分箱后的平均激活曲线。
   层列表对照实际 asset_5.jpg 和 rubric 核验。
8. **均值、PCA 等未完全确定的实现约定显式记录。**
   参数差输出逐 tensor L2/Frobenius norm 及明确标注的算术平均，避免拿 global RMS 冒充。
   PCA 要共享坐标系并记录中心化；不强制论文未交代的库版本、符号或经验阈值。
   不把这些操作性澄清伪装成来源逐字规定。
9. **真实方法与经验结论分开。**
   要求真实 PPLM 梯度、KL、融合、DPO loss、超参数、数据规模、checkpoint 关联和指标计算。
   94%、0.99、1e-5、毒性下降等论文结果不成为必须达到的代码审批阈值。
   post-DPO perplexity 可作为明确披露的 alpha 校准参考，但不能当作已经测出的结果。
10. **范围和执行要求正确。**
    Llama2 和 appendix-only 新实验不设必过项；Appendix E 中正文方法的超参数保留。
    不要求现在进行训练、生成大规模样本或评分。reproduce.sh 是将来可执行的入口，当前仅检查其代码。
11. **单纯覆盖映射不等于语义正确性或高分保证。**
    36/36 表示每个叶节点都有明确检查归属；不代表评审模型必然正确读取代码，或不存在其它漏项。
    评分器的文件选择遗漏、输出解析异常不能通过 checklist 保证消失。

## 每个任务的检查项数

| 任务 | 原任务名称 | 数量 |
|---|---|---:|
| task_001 | Resource inspection and environment setup | 2 |
| task_002 | Data preparation - Toxicity probe dataset | 5 |
| task_003 | Data preparation - Toxicity probe model | 4 |
| task_004 | Extracting toxic MLP value vectors | 5 |
| task_005 | SVD decomposition of toxic vectors | 2 |
| task_006 | Intervention module - subtracting toxic vectors | 7 |
| task_007 | Metrics computation - toxicity, perplexity, F1 | 5 |
| task_008 | PPLM-based pairwise dataset generation | 7 |
| task_009 | DPO training implementation | 6 |
| task_010 | Logit lens visualization | 6 |
| task_011 | Residual stream shift analysis | 8 |
| task_012 | Value vector shift cosine similarity analysis | 10 |

## 离线验证

构建程序执行以下断言：现有 parse_plan 契约可解析；任务 ID/名称/顺序保留；
description 只追加上述三个已披露范围补充；checklist ID 唯一且归属正确；
每条 quote 均为对应文档中的**逐字连续子串**；全部 36 个 Code-Dev 叶节点映射存在且 ID 有效。
该验证不启动任何实验、评分、API 或 Docker。

可从 仓库根目录 离线复核（只打印结果）：

```bash
agents/ra-skill-list/.venv-wsl/bin/python agents/ra-skill-list/plans/diagnostic/mechanistic-rubric-assisted/build_plan.py
```

运行入口兼容性已经读代码确认：solver 的外部 --plan 分支对含 checklist 的大纲直接初始化，跳过补 checklist 的 planner。
run_experiment.sh 现在接受完整 plan，验证后将输入保存为 input_plan.json，然后调用 run_agent.sh 与评分器。
仍可用不含 checklist 的旧大纲，原来的补 checklist 分支保持有效。上述命令只是离线复核程序，不会启动实验。

## Code-Dev 逐叶覆盖映射

此表仅供研究审计，不需要提供给 agent。正文及其他类别中规定的实验规模、调用流程也已纳入对应检查项，
但没有把 Code Execution / Result Analysis 节点变成“必须现在执行实验/达到结果”的审批要求。

| Rubric 叶节点 | 原要求 | 对应 checklist |
|---|---|---|
| `cfcf4e45-a955-4064-8c12-17bc3787bed7` | Code for splitting Jigsaw toxic comment classification dataset into 90:10 training and validation sets has been implemented. | `task_002_check_001`, `task_002_check_002` |
| `14994612-f5c9-4578-91ca-d9670a4b42c2` | The code for a binary classifier of the form softmax(Wx) has been implemented to classify tokens as toxic or non toxic. | `task_003_check_002` |
| `627f8d87-0aed-4cb5-a29e-8208c3bc9871` | The classifier takes as input, x, the residual stream in the last layer, averaged across all timesteps. | `task_003_check_001`, `task_002_check_005` |
| `5614c277-43cb-4f54-afef-2af137f8e4a2` | The dimensionality of W is Kx2, where if W[:,0] refers to the non-toxic toxic vectors, then W[:,1] refers to the toxic vectors, and vice-versa. | `task_003_check_003` |
| `b54834e0-a42b-4a51-84cf-8fb9d185dd07` | The code for training the linear probe model has been implemented. | `task_003_check_004` |
| `bbdb4b01-fdf3-4879-bc5c-c59bcc6c12c2` | The code for calculating the 128 value vectors with largest cosine similarity to the toxic vector of the matrix W have been implemented. | `task_004_check_001`, `task_004_check_002` |
| `1a8266f6-fe17-4638-8b69-d338648a3286` | The code for doing SVD decomposition on MLP.vToxic has been implemented. | `task_005_check_001`, `task_005_check_002` |
| `9249d0dc-05dc-4748-afc8-ef1948f4bece` | The code for projecting toxic vectors onto the vocabulary space and identifying the nearest neighbor has been implemented. | `task_004_check_004`, `task_004_check_005`, `task_005_check_002` |
| `a8c8d524-076c-4404-ae91-fd874154f06a` | The code for PPLM described in Section 4.2, has been implemented or imported. The hyper parameters have been set according to Table 9 in Appendix E. | `task_008_check_004`, `task_008_check_005`, `task_008_check_006` |
| `cca1bf0e-dcc8-4f93-91a0-6883f92d2254` | The code for generating positive non-toxic examples for each prompt has been generated. For each prompt, a positive example (non-toxic) has been obtained by sampling greedily from GPT2. | `task_008_check_002`, `task_008_check_001` |
| `3c36d4c4-eb3a-441c-9058-334af9dbdef7` | The code for generating negative toxic examples for each prompt from GPT-2 has been generated. For each prompt, a negative example (toxic) has been obtained by using PPLM and the toxic vector W as the attribute classifier. | `task_008_check_003`, `task_008_check_004` |
| `7bfa543c-8def-49f2-ad8d-adb32ed7e430` | The code for fine-tuning GPT2 using DPO has been implemented. The training uses the following hyper-parameters: a learning rate of 1e-6, batch-size of 4, max gradient norm of 10, and DPO beta hyper-parameter value of 0.1. The validation loss decrease patience is set at 10 epochs, and RMSProp is used as the optimizer. | `task_009_check_001`, `task_009_check_002`, `task_009_check_003`, `task_009_check_004`, `task_009_check_005`, `task_009_check_006` |
| `52557c05-7f73-4700-9d41-90100745c5ac` | The code for identifying prompts that output " shit" as next token has been implemented. | `task_010_check_001`, `task_010_check_002`, `task_010_check_003` |
| `e0533084-5045-4f76-8e6b-e51b210b2a3b` | The code for calculating the probability of " shit" as next token for each layer, including layers within transformer block, has been implemented for GPT2. | `task_010_check_004`, `task_010_check_005` |
| `31a3f55d-0024-44a4-940d-88bf0f0ab3eb` | The code for calculating the probability of " shit" as next token for each layer, including layers within transformer block, has been implemented for GPT2 fine-tuned with DPO. | `task_010_check_004`, `task_010_check_005` |
| `c36d4dc7-20c0-452f-88f2-3921a1fa2f69` | The code for measuring F1 in continuations has been implemented. Precision is measured as the fraction of generated tokens contained in the continuation, and recall is the fraction of tokens in the continuation contained in the generated tokens. | `task_007_check_004` |
| `c03156e4-8b92-45c4-a417-27c1b2b44f34` | The code for measuring perplexity has been implemented. | `task_007_check_003` |
| `9d74128a-30df-4544-aca2-6c7c8a52eadd` | The code for measuring toxicity has been implemented using unbiased-toxic-roberta. | `task_007_check_001`, `task_007_check_002` |
| `52cf0416-3bcb-4e08-a5ee-e6b8e7cdf6d5` | The code for substracting a toxic vector from GPT2 has been implemented as described in Section 3.3. | `task_006_check_001`, `task_006_check_002`, `task_006_check_003`, `task_006_check_004` |
| `6130cc08-3e82-4ca1-932c-a6c885fcba99` | The code for evaluating the most likely next token from a prompt using GPT2 has been implemented. | `task_010_check_001`, `task_010_check_006` |
| `a9267407-e676-4644-928b-16756a6586bb` | The code for substracting a vector in MLP.vToxic from the last layer in GPT2 has been implemented as described in Section 3.3. | `task_006_check_001`, `task_006_check_002`, `task_010_check_006` |
| `9bbf6a62-e20d-43f5-98b7-e3b5e49bdf99` | The code for computing cosine similarity between model parameters has been implemented. | `task_012_check_001`, `task_012_check_003` |
| `cac04bcb-f996-40a0-b31b-127ab52f1a07` | The code for computing the average norm difference between model parameters has been implemented. | `task_012_check_002`, `task_012_check_003` |
| `cfba0129-04b8-4c21-8729-1a716c32dd0f` | Code has been implemented such that the top 5 most toxic value vectors in GPT-2 can be identified by taking the 5 value vectors in GPT-2 with the highest cosine similarity to wToxic, i.e. the top 5 vectors in MLP.vToxic in terms of cosine similarity to wToxic. | `task_004_check_003`, `task_011_check_001` |
| `78e8425c-47fd-420c-af6e-2cf108aac318` | Code has been implemented for measuring the activations across GPT-2 at each step for 20 token generations for each of the 1199 prompts from the RealToxicityPrompts challenge set | `task_011_check_002`, `task_011_check_003` |
| `e258bd80-1a56-4b76-8c38-d642896ffc2a` | Code has been implemented such that the mean activation in GPT-2, measured across the 20 tokens generated for each of the 1199 prompts from the RealToxicityPrompts challenge set (so across 20*1199 points), can be computed for each of the top 5 most toxic value vectors in GPT-2 | `task_011_check_001`, `task_011_check_004` |
| `07c50dfc-6369-490d-a8db-d59716fa30bd` | Code has been implemented for measuring the activations across GPT-2 fine tuned with DPO at each step for 20 token generations for each of the 1199 prompts from the RealToxicityPrompts challenge set | `task_011_check_002`, `task_011_check_003` |
| `8b3007d3-c543-4a90-9c8d-b6b042b82c8d` | Code has been implemented such that the mean activation in GPT-2 fine tuned with DPO, measured across the 20 tokens generated for each of the 1199 prompts from the RealToxicityPrompts challenge set (so across 20*1199 points), can be computed for each of the top 5 most toxic value vectors in GPT (pre-DPO) | `task_011_check_001`, `task_011_check_004` |
| `e1a48a92-2abd-490e-b77b-6bb01093a1ad` | The code for computing the main principal component from the residual stream has been implemented. | `task_011_check_006` |
| `ae70d1b4-7039-4098-81dd-2f3938997824` | Code has been implemented such that the activation of the vector at MLP layer 19, idx 770 (the most toxic value vector) can be measured | `task_011_check_007` |
| `ca0752cf-4c1f-44da-9e51-5b0d36339ac0` | The code for calculating the difference in residual streams has been implemented. | `task_011_check_005`, `task_011_check_006` |
| `ae50c500-869c-4151-8506-34091e25f605` | The code has been implemented to calculate the difference between the residual stream of GPT2 and GPT2 fine-tuned with DPO. | `task_011_check_005`, `task_012_check_005` |
| `fe82a362-4850-480a-9ff8-2bab070d208b` | The code has been implemented to calculate the difference in parameters between the MLP block of GPT2 and GPT2 fine0tuned with DPO. | `task_012_check_004` |
| `91560109-8eb0-404a-8bd0-5900fc09c583` | The code for computing the cosine similarity between the difference in residual streams, delta_i, and the difference in parameters in MLP blocks, delta_mlp_i, has been implemented. | `task_012_check_005`, `task_012_check_006` |
| `89aa7b43-9107-482f-889d-46182f7f5262` | The code for computing the mean activation of the value vectors of the MLP layers has been implemented. | `task_012_check_007`, `task_012_check_008` |
| `2846e181-2992-4d1f-8565-594b90bfc816` | The code for un-aligning the GPT2 by scaling by 10 the 7 MLP vector with highest cosine similarity to the toxic matrix W has been implemented | `task_006_check_005`, `task_006_check_006`, `task_006_check_007` |
