# 线程级前缀消息接口

本接口是本项目源码补丁，不是官方 app-server 现有接口。

```json
{
  "id": 7,
  "method": "thread/prefix/set",
  "params": {
    "threadId": "目标线程 id",
    "messages": [
      {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "控制器自定义的上下文"}]
      }
    ]
  }
}
```

成功返回 `{"id":7,"result":{}}`。发送 `messages: []` 清除；下一次调用整体替换，不追加。

- 接受按序排列的纯文本消息，角色为 `user`、`developer` 或 `assistant`。系统/base instructions 仍由原生接口管理。
- `assistant` 文本可采用 Responses 的 `output_text`，用户/开发者文本采用 `input_text`。拒绝工具项、图片、音频、空文本和响应元数据。
- 线程必须已加载，遵循原有直接输入权限检查。不同线程的前缀隔离。
- 全部验证成功后才原子替换；非法更新保留原前缀，返回 JSON-RPC 错误。
- 不启动新 turn、不写入会话历史。重启或恢复线程后由控制器重新设置。
- 每次执行模型请求前读取最新前缀；已经发出的请求不追溯更改。工具等待期间可以更新，控制器必须先等成功响应，再发送工具结果。
- 原生历史压缩不删除前缀，压缩后的执行请求重新拼接。前缀计入上下文预算；前缀过大且压缩后仍超限会报错，不静默截断。
- 控制器拥有内容、文件读取、大纲隐藏与状态逻辑。Codex 引擎仅处理通用消息数组。

控制器中的 `plan_prefix_messages()` 将原有大纲渲染结果变成一条前缀消息；`agent_view.json` 仅为日志展示产物，不再是 Codex 的数据来源。
`prefix_snapshots/` 记录控制器成功提交的数组；app-server JSONL 也记录请求和响应。未实现每次模型请求的独立快照。
