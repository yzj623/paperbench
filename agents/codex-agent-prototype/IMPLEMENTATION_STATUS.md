# 实现与验证状态

## 已完成

- 独立外置控制器、双向 JSONL app-server 客户端、动态协议工具。
- planner 只读规划与原线程 JSON 修正；指定旧大纲跳过 planner。
- 主 agent 持续线程；每次子任务审批创建新的无状态 checker。
- 原来的任务推进、拒绝计数、三次拒绝后跳过、最终确认与冻结门槛。
- 主 agent 普通回复不能结束实验；checker 故障经过原协议的有限重试后作为工具错误返回主 agent，不计为有效拒绝。
- 新版 Codex 提供通用 `thread/prefix/set`，由控制器提交前缀消息数组；每次执行采样、工具继续、传输重试和压缩后的继续请求读取最新数组。
- 仅在请求副本注入，不持续堆积旧大纲；原生压缩与模型执行循环保留。
- 只注册角色自己的动态工具；原生写文件/执行命令/子 agent 等不能绕过协议工具边界。
- 主 agent 使用 Codex 原生 Skill 发现，只提供当前论文完整 Skill，关闭无关 bundled Skills；planner/checker 不启用 Skills。
- 一个容器容纳控制器及所有角色；默认 Docker 设置与既有 Pi/RA 实验一致。
- 独立运行目录、事件轨迹、checker 证据日志、控制器前缀更新快照、容器外 Code-Dev 评分。
- PaperBench 及所需 common 包也已复制到本目录，默认不再依赖兄弟目录。
- 独立 `.codex-auth/` 已完成 ChatGPT 登录，容器账户检查返回 Plus；默认实验模型配置为用户指定的 `gpt-6-luna`。

## 已执行的验证

| 验证 | 结果 | 能证明什么 |
|---|---|---|
| Python 离线测试 | 53 项通过 | 控制器、实际 JSONL 模拟子进程、嵌套 checker、错误/重试、任务门槛、启动器、隔离性；新增前缀替换/清空/错误及时序测试 |
| 旧版 Docker 中 Python 测试 | 当时 48 项通过 | 前缀接口修改前的记录；新版未重复容器测试 |
| 轻量 Rust 前缀模块测试 | 3 项通过 | 真实新增模块的有序拼接、原历史不变、空数组、角色/文本校验、稳定 ID、非法更新不改变旧值及线程隔离；使用类型替身，不代表整个执行循环集成测试 |
| `cargo check --locked -p codex-config` | 通过 | 新配置字段和配置模块依赖的编译检查 |
| `cargo check --locked -p codex-app-server --bin codex-app-server` | 新版通过 | 通用前缀 RPC、app-server、core 和相关依赖通过 Rust 类型/编译检查 |
| 新版完整 Docker 编译与链接 | 通过 | 通用前缀接口版二进制已生成；单任务 Rust 编译用时 4 分 36 秒 |
| 版本元数据更新与重编 | 通过 | workspace 与锁文件中的本地包版本更新为 `0.162.1`；新镜像 app-server 报告 `0.162.1`，第三方依赖版本未变 |
| Plus 登录与 `gpt-6-luna` 实际模型测试 | 通过 | 新版本模型目录列出该模型；真实模型调用、动态工具和同一轮对话中的前缀替换成功 |
| 合成 planner/main/checker 流程 | 通过 | 实际模型完成规划、原生 Skill 读取、写入、审批、最终提交与冻结；仅用合成任务和测试提示词 |
| 新版镜像断网启动检查 | 通过 | 版本检查、控制器依赖导入及 JSONL initialize 成功；`thread/prefix/set` 已识别，并正确拒绝不存在的线程；未创建模型任务 |
| 旧文件注入版 Docker 编译与断网启动 | 曾通过，镜像现已删除 | 这是替换接口前的版本，不可作为新版验证证据 |
| 协议/提示词 SHA-256 | 通过 | 核心协议及提示词仍与复制时 Pi 版本逐字一致 |
| Shell 语法、Rustfmt、Git diff | 通过 | 脚本与源文件基本格式和差异检查 |
| 启动 dry-run 与无认证 host smoke | 通过 | 启动参数匹配实际控制器；不会自动登录或调用模型 |

首次容器测试使用未安装 jsonschema 的旧基础镜像，因此失败；在临时容器内安装 requirements 中的这一依赖后全部通过。
正式 Dockerfile 会安装完整 requirements，不修改现有 pb-env 镜像。

## 尚未执行

- Codex 完整原生集成测试套件；新增的原生 router / WebSocket / app-server 测试已有源码，但本轮未运行该完整套件。
- 真实论文复现或评分。

旧镜像 `codex-paperbench:latest`（ID `sha256:2566bd7b686e7c046c8a55b482002d6f1b573a8de3e5e7a4b23377fccc3501b6`）已按用户要求删除。
通用前缀接口版镜像已重新构建：`codex-paperbench:latest`，ID 为 `sha256:643630228cadcd4e1f051a35912221050733fde047d22b87ecf93eaf3b81c347`。
旧 `0.0.0` 构建的模型目录没有 `gpt-6-luna`，调用返回 HTTP 400（记录：`runs/live-codex-check.5nXylH/summary.json`）。更新构建版本元数据为 `0.162.1` 后，模型目录包含 `gpt-6-luna`，真实调用与合成完整流程通过（记录：`runs/live-codex-version-lf7ZRtDg/summary.json`）。本地源码提交时间晚于 `0.162.1` 正式发布标签，保留了本项目补丁；镜像是以该版本号构建的定制版，并非官方原封不动的发布二进制。
构建时修正了过宽的凭据排除规则，仅允许两个必须的 Rust 源文件进入构建上下文；真实凭据仍排除。

## 安全与资源

Rust 1.95 工具链、Cargo 缓存及本机编译所需的 OpenSSL 开发文件均放在本目录 `.build-tools/`；没有修改系统 PATH、安装全局开发包或改动 Docker daemon。
编译使用单任务，正式运行默认资源行为与既有实验一致，可自行通过启动参数设限。
镜像构建支持 `--proxy URL`，只作用于构建步骤，不修改 Docker daemon 或写入最终镜像环境；Cargo 下载与编译缓存由 BuildKit 保留。
此次构建通过 `--cgroup-parent pplabcodexbuild.slice` 在临时资源组内运行，所有 RUN 步骤合计限制 CPU 2 核、RAM 3 GiB、swap 2 GiB，并发数为 1。已验证构建进程位于该资源组；内存峰值达到 3 GiB，没有 OOM 或 OOM kill。构建后停止资源组并清除其临时配置，不改变实验运行资源设置。
`.grader.env` 仅供容器外评分，未进入 agent 容器或镜像。使用项目独立登录目录，不自动读取个人默认 Codex 登录目录；未输出凭据内容。
