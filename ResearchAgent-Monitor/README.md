# ResearchAgent Monitor

独立、只读、单进程的 PaperBench ResearchAgent 网页监控器。

直接启动：

```bash
cd ResearchAgent-Monitor
bash start_monitor.sh
```

tmux 后台启动（只终止同名旧监控会话）：

```bash
tmux kill-session -t research-agent-monitor 2>/dev/null || true
tmux new-session -d -s research-agent-monitor \
  'cd ResearchAgent-Monitor && exec bash start_monitor.sh'
```

环境变量：`WEB_HOST`、`WEB_PORT`、`EXPERIMENTS_ROOT`。默认监听
`http://0.0.0.0:8765`，Windows 浏览器访问 `http://127.0.0.1:8765`。

## 自动评分展示

页面会定期读取本次实验的 `evaluation/runs/*/*/grade.json` 或 Pi 的 `grade/runs/*/*/grade.json`。
兼容顶层 `score` 和 `paperbench_result.judge_output.score`，显示最新百分比分数、评分状态、评分记录数量和异常评分项数量。
评分中、评分失败、文件未写完或没有结果时不会把缺失分数显示为 0；真实 0 分会正常显示。
同一目录内有多份评分时按文件更新时间选择最新记录；不读取其他提交区的评分，也不会改写评分结果。
实验结束后保留原监控目标，避免自动切换到无关历史实验。可用 `--run-dir` 显式固定目标目录。

## 自动跟随串行实验

指定串行脚本产生的 `summary.json`，监控会跟随当前实验，并在下一个实验启动时自动切换。
评分期间继续显示该实验，批次结束后停留在最后一个实验；页面显示当前配置名称。

```bash
RA_BATCH_SUMMARY=/绝对路径/批次/summary.json bash start_monitor.sh
```

`RA_BATCH_SUMMARY` 与 `RA_RUN_DIR` 不能同时设置。命令行也支持 `server.py --batch-summary /绝对路径/summary.json`。
