# Codex 论文复现实验默认约定

用户指定 Codex 版论文复现实验时，除非另行说明：

- 模型使用 `gpt-6-luna`，思考强度使用 `medium`。
- 使用对应论文的普通 ResearchAgent 生成的大纲；不要将 Skill、Checklist 或其他版本的大纲误作普通 RA 大纲。
- 历史实验使用的 `mechanistic-understanding` 普通 RA 大纲来自旧实验目录；该实验产物未纳入此仓库。新机器运行时请用 `--plan` 指定本地可用的大纲，或让 planner 重新生成。

此文件记录实验偏好，不代表模型侧的跨会话记忆，也不自动修改启动脚本。每次启动仍须核对模型、思考强度与大纲来源。
