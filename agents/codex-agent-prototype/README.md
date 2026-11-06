# Codex PaperBench 适配版

本目录独立实现原 Pi 适配版的 planner + 主 agent + 无状态 checker 控制协议。
Codex 官方源码固定在 `12cca49c81c8aa6a37a5cc4e39964ee3b70f9458`，修改仅位于本目录的 `codex/`。
不导入或运行其他 RA/Pi 目录的代码，也不读取那些目录的登录信息。

## 机制

1. 外置 Python 控制器维护不可变大纲、任务状态、拒绝次数、异步任务和最终提交门槛。
2. 可以指定已有大纲；未指定时启动只读 planner。planner 返回 JSON 后用本地契约校验，错误时在原线程追加修正要求。
3. 主 agent 使用持续的 Codex 线程，可调用原有 `submit_subtask`、`skip_subtask`、`final_submit` 和文件/命令/异步任务工具。
4. 每次提交子任务创建新的只读 checker 进程/线程，只接收当前子任务和本次提交说明。通过才能推进状态；有效拒绝达到三次才允许跳过。
5. 最终提交检查状态、必需文件、活动工作进程；有跳过任务时保留原来的再次确认逻辑。通过后冻结提交区，结束复现，容器外评分。
6. 普通 assistant 文本不能代替协议提交。合法 API 重试通知不会被外置控制器提前终止。

`protocol/` 是已经复制进本目录的原协议代码；状态机保持不变，提示词已适配 Codex 原生工具。
`provenance.json` 保存原始协议快照的 SHA-256 和当前提示词的 SHA-256，测试会核对。

## Codex 源码补丁

- 新增 `thread/prefix/set` RPC：控制器整体替换当前线程的前缀消息数组，`[]` 清除；Codex 不读取大纲文件，也不理解大纲语义。
- 消息数组按给定顺序加入每次模型请求的 `input` 顶部，原生 base instructions 仍保留。只接受文本 `user` / `developer` / `assistant` 消息，不接受工具调用项。
- 前缀仅存于线程内存，不写入永久历史、不被历史压缩；每次执行采样、工具继续和传输重试读取最新前缀。计入上下文预算，必要时调用原生压缩。
- 控制器负责大纲渲染与隐藏规则，并在返回工具结果前等待前缀更新成功；失败时终止，不允许携带旧前缀继续。进程重启或线程恢复后需重新设置。
- `ppblab_dynamic_tools_only=false`：保留 Codex 原生工具，并把任务提交／审批协议作为动态工具追加；工具执行能力受容器和各角色的 Codex sandbox 设置约束。
- `--ppblab-protocol --strict-config`：控制器要求补丁版程序和严格配置。误用未修改的官方二进制会在启动时失败。

没有修改账户认证、模型服务地址、收费/配额处理，也没有改用提取登录 token 的通用 API 调用。
任务状态仍由外置控制器决定，压缩、模型执行循环和 Skill 发现由 Codex 管理。

## 自包含资源

- `skills/`：完整 PaperBench Skill，包括脚本和参考资料；每次仅挂载当前论文的 Skill。
- 主 agent 原生扫描 `.agents/skills`，索引名称/描述后按需阅读；不预塞全量 Skill。无关 bundled Skills 关闭，planner/checker 不启用 Skills。
- `codex_eval/` 和 `scripts/evaluate_submission.sh`：本地评分包装器，默认 USTC 模型为 `deepseek-v4-flash-ascend`（可用 `USTC_JUDGE_MODEL` 覆盖），保持 `code_only=true`、`skip_reproduction=true`。
- `vendor/paperbench-official/`：PaperBench 和其相对路径公共包的本地副本，默认启动和评分均使用这一副本。
- `.grader.env` 是复制的本地评分配置，不挂载到 agent 容器，不放入镜像或 Git。

## 无登录测试

```bash
cd agents/codex-agent-prototype
bash scripts/test_offline.sh
bash scripts/test_rust_plan.sh
.venv/bin/python run_paperbench.py --paper-id mechanistic-understanding --dry-run
```

Python 测试包含实际 JSONL 子进程模拟服务：大纲修正、嵌套新 checker、任务反馈、冻结提交、进程门槛、异常和重试。
Rust 轻量测试直接包含真实大纲模块，但用小型类型替身隔离依赖；它不代替完整 Codex 编译和集成测试。
`--smoke-test` 检查公共输入和配置，不启动模型进程、不登录、不评分；不能据此认定模型服务链路已经正常。

## 构建与后续启动

```bash
bash build_codex_image.sh --jobs 1
# 构建步骤需要代理时：添加 --proxy http://<可从 Docker 访问的 Clash 地址>:7890
# 或只编译本地程序：bash build_codex.sh --jobs 1
```

镜像中包含补丁版 `codex-app-server` 和本目录控制器。所有角色在同一个 Docker 容器运行。
默认 root、可写工作区和普通 bridge 网络；仅挂载当前实验所需资源，不挂载 Docker socket 或工作区的宿主机同路径。资源限制默认不设置。
可用 `--memory`、`--cpus` 限制资源，用 `--proxy http://127.0.0.1:7890` 指定本机 Clash，Docker 路径自动转换为 host-gateway。

在新机器上先从仓库根目录运行 `bash scripts/login_codex.sh`，通过设备授权将凭据保存到本项目 `.codex-auth/`；也可显式提供 `--auth-dir`。脚本不会复制 IDE 的登录凭据。
三个角色复用这个显式目录的认证，且不自动切换到个人默认登录目录。

认证和镜像准备后，正式实验命令形式：

```bash
.venv/bin/python run_paperbench.py \
  --paper-id mechanistic-understanding \
  --plan /absolute/path/to/frozen-plan.json \
  --model gpt-6-luna
```

默认启用语义 Check Agent。若要关闭 checker，让 `submit_subtask` 在协议校验通过后直接标记完成：

```bash
.venv/bin/python run_paperbench.py \
  --paper-id mechanistic-understanding \
  --plan /absolute/path/to/frozen-plan.json \
  --model gpt-6-luna \
  --no-checker
```

可选地关闭与复现无关的 Codex 原生 developer 上下文，便于做消息消融实验：

```bash
.venv/bin/python run_paperbench.py \
  --paper-id mechanistic-understanding \
  --plan /absolute/path/to/frozen-plan.json \
  --model gpt-6-luna \
  --no-checker \
  --strip-irrelevant-developer-context
```

该开关会关闭多 agent、apps、plugins、tool suggestions、collaboration mode 和 apps instructions。
副作用是子 agent、插件/Apps 工具和 plugin skills 也会一并停用；permissions、论文 skills 和环境上下文保留。
默认关闭；用 `--no-strip-irrelevant-developer-context` 可显式恢复默认行为。

启动脚本与控制器默认模型已设为 `gpt-6-luna`，仍可用 `--model` 显式覆盖。补丁版 app-server 的构建版本元数据已设为最新正式发布版本 `0.162.1`；项目独立 Plus 登录下的实际模型调用、前缀更新与合成 planner/main/checker 流程已通过。
所有结果写入独立唯一 `runs/<paper-id>-<随机后缀>/`，不会覆盖既有实验。
`output/main.jsonl`、`planner.jsonl`、`checker/`、`protocol/` 保留运行与审批证据。`prefix_snapshots/` 保存控制器每次成功设置的消息数组，不声称是每次模型请求的快照。

## 可选 GPT-6 Luna 评分器

评分默认仍使用学校 DeepSeek 接口。仅为某次评分切换为 GPT-6 Luna：

```bash
PAPERBENCH_GRADER_API_CONFIG=openai-oauth \
  bash scripts/evaluate_submission.sh PAPER_ID /absolute/submission /absolute/new-evaluation-dir
```

同一环境变量也可用于 `run_paperbench.py`、`scripts/regrade_serial.py` 和批量重评脚本，
仅改变评分模型。选文件、判分及整数/浮点解析都显式使用 `gpt-6-luna`、`medium`。
文件选择预算仍为 128,000 token；解析调用使用严格 JSON schema，必需
`valid_score`、`score`、`explanation`，格式异常最多重新请求两次，不补造字段。
选文件、自然语言判分及辅助解析提示词均逐字保留；单条前置 system 消息原文映射到
Responses 的 instructions 字段，不增补或替换任何评分提示词。rubric 和加权方式不变。

使用统一凭据 `~/.config/ppblab/openai-oauth/credentials.json`，支持自动刷新和文件锁；
可用 `OPENAI_OAUTH_DIR` 或 `CODEX_GRADER_OPENAI_OAUTH_DIR` 指定目录。
传输、刷新、重试代码都在本项目 `codex_eval/` 内，不运行时导入 RA 的代码。
这份凭据与 Codex CLI 的 `.codex-auth` 是两种不同认证，不可相互替代。
代理默认继承 `HTTPS_PROXY`，也可用 `CODEX_GRADER_OPENAI_PROXY` 单独设置。

## 网页监控

运行 `bash monitor/start_monitor.sh`，浏览器打开 `http://127.0.0.1:8768`。
监控器位于本项目 `monitor/`，显示大纲、任务隐藏状态、主 agent 轨迹、checker
逐次审批和评分状态。默认优先跟踪活跃正式实验；用 `CODEX_RUN_DIR=/绝对路径/实验目录`
固定目标。详情见 [monitor/README.md](monitor/README.md)。

## 验证边界

详细状态见 `IMPLEMENTATION_STATUS.md`。新接口版 `codex-paperbench:latest` 已完成构建，并通过实际 Plus 模型的合成协议测试；尚未验证真实论文实验与评分。

实现参考 OpenAI Docs 的 [app-server](https://learn.chatgpt.com/docs/app-server) 与 [Skill 文档](https://learn.chatgpt.com/docs/build-skills)，协议按本地固定版本源码核对。

2026-10-13 真实测试记录：项目独立 ChatGPT 登录已完成，容器账户检查确认 Plus。当前 WSL 下应使用实际 Windows Clash 地址 `--proxy http://172.22.224.1:7890`；本次 `host.docker.internal` 代理路径的 app-server 账户路由查询失败，而实际 IP 成功（WSL 重启后需核对 IP）。旧版 `0.0.0` 构建的失败记录保留在 `runs/live-codex-check.5nXylH/summary.json`；`0.162.1` 构建通过记录见 `runs/live-codex-version-lf7ZRtDg/summary.json`。未运行真实论文复现或评分。
