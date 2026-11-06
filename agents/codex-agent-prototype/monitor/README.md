# Codex PaperBench 网页监控

复制 Pi 监控器的页面布局，在本项目内独立解析 Codex app-server JSONL 和本地控制协议。
无需安装额外 Python 包，不读取认证或评分配置，不启动、停止或修改实验。

从仓库根目录启动：

```bash
bash agents/codex-agent-prototype/monitor/start_monitor.sh
```

浏览器打开 `http://127.0.0.1:8768`。页面每两秒更新大纲与隐藏状态、主 agent
工具调用及返回、checker 每次审批的读取路径、提交说明、结论与拒绝问题，以及评分阶段和已有得分。
默认优先显示运行中的正式实验，否则显示最近实验。`launch.json` 中标记为
`smoke_test` 的运行和无正式启动元数据的调试目录不会显示。

固定某次实验：

```bash
CODEX_RUN_DIR=agents/codex-agent-prototype/runs/mechanistic-understanding-3v12w5g4 \
  bash agents/codex-agent-prototype/monitor/start_monitor.sh
```

`CODEX_RUNS_ROOT`、`WEB_HOST`、`WEB_PORT` 均可覆盖，或使用脚本的
`--runs-root`、`--run-dir`、`--host`、`--port` 参数。默认仅监听本机。

监控使用 `launch.json` 关联 `home/` 和 `output/`，读取 `output/protocol/`、
`output/main.jsonl` 和 `output/checker/`。`host_status.json` 用于区分 agent
运行、评分、评分失败和结束；分数读取 `evaluation/runs/*/*/grade.json`
中评分器写出的 `paperbench_result.judge_output.score`，不自行计算评分。
轨迹保留摘要，不把 `write_file` 的源代码正文送到浏览器。评分结果尚未产生时显示“尚无结果”。

接口：`/api/health`、`/api/runs`、`/api/state`、`/api/files`、
`/api/trace?role=main`、`/api/trace?role=checker`。

离线验证：

```bash
python3 -m unittest discover -s agents/codex-agent-prototype/monitor/tests -v
```

## 自动评分展示

页面会定期读取本次实验的 `evaluation/runs/*/*/grade.json` 或 Pi 的 `grade/runs/*/*/grade.json`。
兼容顶层 `score` 和 `paperbench_result.judge_output.score`，显示最新百分比分数、评分状态、评分记录数量和异常评分项数量。
评分中、评分失败、文件未写完或没有结果时不会把缺失分数显示为 0；真实 0 分会正常显示。
同一目录内有多份评分时按文件更新时间选择最新记录；不读取其他提交区的评分，也不会改写评分结果。
实验结束后保留原监控目标，避免自动切换到无关历史实验。可用 `--run-dir` 显式固定目标目录。
