# PaperBench 论文复现 Agent 实验仓库

本仓库保存几个 Research Agent 版本、Codex 适配版、PaperBench 评分代码和监控器的源码。旧实验提交区、评分结果、凭据、虚拟环境、Docker 镜像和大部分论文 PDF 没有随仓库迁移。以下流程面向 **Linux x86_64 / WSL2**，实验结束后进行 PaperBench CodeDev 自动评分。

## 目录

| 目录 | 内容 |
| --- | --- |
| `agents/ResearchAgent` | Planner、主 Agent、Checker |
| `agents/ResearchAgent-NoChecker` | Planner、主 Agent；无 Checker |
| `agents/ResearchAgent-Skill` | 带 Skill 选择的 RA |
| `agents/ResearchAgent-Skill-ErrorMemory` | 带错误记忆的 Skill 版本 |
| `agents/ra-skill-list` | Checklist 规划与逐项审查版本 |
| `agents/newra` | 初稿后进行整体审查的版本 |
| `agents/codex-agent-prototype` | Codex 控制器、补丁版 Codex 源码、评分适配与监控 |
| `codex-source` | 另一份 Codex 源码快照 |
| `PaperBench-Official` | PaperBench 基准源码、论文材料与 rubric |
| `ResearchAgent-Monitor` | RA 网页监控 |
| `scripts` | 跨版本的准备、运行与 Codex 登录脚本 |

RA 实验默认写入根目录 `experiments/`；Codex 实验默认写入 `agents/codex-agent-prototype/runs/`。两者均被 Git 忽略。

## 1. 克隆与基础工具

在 Linux/WSL 中安装 Git、Git LFS、Python 3（包含 `venv` 和 `pip`）、`uv`、Docker，并确保 Docker daemon 正在运行。Codex 登录还需要 Codex CLI，或者 Node.js 的 `npx`。

```bash
git clone <仓库的 GitHub 地址>
cd paperbench-agent-study
git lfs version
uv --version
docker info
```

准备期间需要访问软件包源、Docker 镜像源和 [PaperBench 上游仓库](https://github.com/openai/frontier-evals)。Codex 镜像会编译 Rust 源码，可能占用较多时间和空间。其 Dockerfile 当前下载 x86_64 辅助程序；其他架构尚未适配。

## 2. 准备论文材料与环境

`AGENT` 使用上表中 `agents/` 下的目录名；`PAPER_ID` 使用 PaperBench 的论文 ID。示例：

```bash
bash scripts/prepare.sh ResearchAgent mechanistic-understanding
```

准备脚本补齐所选论文的 PDF、Markdown、补充材料及资源文件，创建该版本的 Python 环境，并构建 `pb-env:latest`。换论文时需为新 ID 再运行一次；已有镜像与下载缓存可以复用。

准备 Codex 版：

```bash
bash scripts/prepare.sh codex-agent-prototype mechanistic-understanding
```

它还会安装 Codex 控制器依赖并构建补丁版 Codex 镜像。默认编译并行度为 1，可设置 `BUILD_JOBS`。论文下载缓存位于 `.runtime/frontier-evals/`，不提交到 Git。RA 版本可用 `PAPERBENCH_ROOT` 指向另一份材料完整、版本兼容的 PaperBench 目录；Codex 版默认使用其 `vendor/paperbench-official/` 副本。上游克隆和 Git LFS 下载默认各有 300 秒超时，可通过 `PAPERBENCH_CLONE_TIMEOUT`、`PAPERBENCH_LFS_TIMEOUT` 调整。

## 3. 配置 Agent 与评分器凭据

### Python Research Agent

每个 RA 版本的启动脚本会读取**该版本目录内**的 `.env`。以下示例让普通 RA 使用 `gpt-6-luna`、中等思考强度，并用官方 DeepSeek 接口评分：

```dotenv
# agents/ResearchAgent/.env；替换占位符，不要提交此文件
RESEARCH_AGENT_API_CONFIG=openai-oauth
RESEARCH_AGENT_MODEL=gpt-6-luna
RESEARCH_AGENT_REASONING_EFFORT=medium
PAPERBENCH_GRADER_API_CONFIG=deepseek-official-flash
DEEPSEEK_API_KEY='YOUR_DEEPSEEK_KEY_HERE'
```

准备环境后，单独完成 RA 的 OAuth 登录：

```bash
agents/ResearchAgent/.venv-wsl/bin/python \
  agents/ResearchAgent/scripts/openai_oauth.py login
```

登录程序提供浏览器授权链接，默认将凭据保存在当前用户的 `~/.config/ppblab/openai-oauth/`。该凭据供 **RA 模型**调用；示例中的 `DEEPSEEK_API_KEY` 供**评分器**调用。其他 RA 版本可使用同一 OAuth 凭据，但须分别配置自己的 `.env`，或从 shell 导出环境变量。使用 Qwen 等模型时，按对应版本 README 配置 API 模式与密钥。

### Codex

在新电脑上完成一次 Codex 设备授权：

```bash
bash scripts/login_codex.sh
bash scripts/login_codex.sh --status
```

脚本将凭据保存到 `agents/codex-agent-prototype/.codex-auth/`，即实验启动器默认读取的目录。若使用本机浏览器授权，可运行 `bash scripts/login_codex.sh --browser`。这份凭据**不供评分器使用**。

Codex 自动评分仍需单独提供评分器凭据。例如在 `agents/codex-agent-prototype/.grader.env` 中写入：

```dotenv
PAPERBENCH_GRADER_API_CONFIG=deepseek-official-flash
DEEPSEEK_API_KEY='YOUR_DEEPSEEK_KEY_HERE'
```

上述两个凭据目录和 `.env` 文件都被 Git 忽略。Codex 也支持用 `--no-grade` 跳过自动评分。

## 4. 运行实验

普通 RA、NoChecker、Skill、ErrorMemory 与 newra 版使用同一个入口，更换 `AGENT` 名称即可：

```bash
bash scripts/run.sh ResearchAgent mechanistic-understanding
bash scripts/run.sh ResearchAgent-NoChecker mechanistic-understanding
```

需要固定大纲时，可以设置 `RESEARCH_AGENT_PLAN=/绝对路径/plan.json`；否则由各版本的 Planner 生成。newra 的审查轮数用 `RESEARCH_AGENT_REVIEW_ROUNDS` 设置，默认 2。

Checklist 版**必须**传入大纲 JSON，可为其支持的无 Checklist 大纲或完整 Checklist 大纲：

```bash
bash scripts/run.sh ra-skill-list all-in-one /绝对路径/plan.json
```

Codex 版默认启动 Planner、主 Agent 与 Checker，并在成功提交后自动评分：

```bash
bash scripts/run.sh codex-agent-prototype mechanistic-understanding \
  --model gpt-6-luna
```

Codex 可选 `--plan /绝对路径/plan.json` 指定大纲，或用 `--no-checker`。其他参数见各版本 README 和脚本用法。

## 5. 网页监控与结果

另开终端启动相应监控器：

```bash
bash ResearchAgent-Monitor/start_monitor.sh
# Codex 版：
bash agents/codex-agent-prototype/monitor/start_monitor.sh
```

RA 监控默认读取 `experiments/`，Codex 监控默认读取其 `runs/`。端口、固定实验目录和其他参数见各监控器 README。`EXPERIMENTS_ROOT` 可以调整 RA 产物目录。

## 当前验证范围

已完成路径推导、Shell 语法、离线材料检查、Codex 启动命令 dry run、Codex 登录脚本模拟测试。**尚未在一台干净的新电脑上完成“克隆 → Git LFS 下载 → 构建 → 真实 Agent → 自动评分”全链路验证**；本机的上游 LFS 下载测试也未完成。README 给出当前源码对应的准备步骤，不能替代端到端验证。

`PaperBench-Official` 及 Codex 源码来自外部项目，上传时应保留各自的许可证与来源说明。官方目录包含 rubric，不能复制进 Agent 工作区；`agents/ra-skill-list/plans/diagnostic/` 中的历史诊断材料也不应作为不看 rubric 的正式实验输入。
