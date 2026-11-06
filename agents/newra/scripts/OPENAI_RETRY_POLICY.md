# OpenAI OAuth 重试防御

实现仅在普通 RA 和 RA-Skill 的 OpenAI OAuth 适配层中生效。
两套目录保留各自的同版代码，共用凭据文件；不改变 Qwen、DeepSeek、工具协议和评分器。

## 分类与处理

| 情况 | 处理 |
| --- | --- |
| 连接失败、连接重置、流式分块未读完、读写错误、代理错误、提前 EOF | 丢弃未完成响应，退避重试 |
| 请求超时、连接/读取/写入/连接池超时 | 有限退避重试 |
| HTTP 408、409、临时 429、5xx（包括 500、502、503、504） | 检查错误码和服务端提示后重试 |
| SSE `response.failed` / `error` 中的临时服务错误、限流或过载 | 收集错误元数据，有限重试；也兼容 SDK 直接抛出的流式 `APIError` |
| `subscription_sharing_usage_unavailable`、`subscription_sharing_user_unavailable` | 临时不可用，保留凭据并重试 |
| `subscription_sharing_usage_limit_exceeded` | 保留当前请求与 agent 状态，按 `Retry-After` 或至少 300 秒等待；不限轮次，直到额度恢复或进程被取消 |
| 余额不足、项目/组织支出及用量上限 | 停止当前请求，需要处理账户或计费配置 |
| HTTP 401 | 每条逻辑请求最多恢复一次：刷新凭据，再请求；第二次 401 停止 |
| 400、403、404、422、无效能力/路由/授权/安全策略错误 | 不盲目重试，需要修正请求、授权或配置 |
| `response.incomplete` | 保留失败信息，不执行部分输出；长度/过滤等原因不靠重复同一请求解决 |
| 用户取消、代码错误、本地协议或 URL 构造错误 | 原样停止，不吞掉取消或掩盖程序错误 |

服务端错误码优先用于排除不可重试情况；不是把所有 429、5xx 都一律重试。
授权刷新接口自身的临时 HTTP/网络错误也由同一策略处理。
服务端确认刷新令牌失效时，在锁内清除不可用的访问/刷新令牌，保留账户和客户端信息，要求重新授权。
临时网络故障不会删除或覆盖旧凭据。

## 上限、延迟与日志

- 普通临时错误默认最多 12 轮请求尝试（含首次；401 恢复最多额外请求一次）。
- 普通临时错误的每轮逻辑调用最多 2 小时；每次流式请求最多 10 分钟。订阅额度等待发生在此计时之外。
- 默认指数退避加随机抖动，等待上限 300 秒。
- 遵守 `Retry-After` 秒数、HTTP 日期及 `retry-after-ms`；超过配置等待上限时停止重试、交由调用方稍后重启，绝不提前请求。
- 普通临时错误遵守 `x-should-retry: false`；订阅额度错误按上面的等待策略处理。
- SDK 设置 `max_retries=0`，仅使用适配器这一层重试，避免重试次数相乘。
- 日志记录重试轮次、等待时间、错误类型、HTTP 状态、错误码、request ID；不记录 bearer、刷新令牌或请求正文。

只向 RA 返回收到 `response.completed` 的完整响应。失败尝试产生的文本、函数参数和 reasoning 缓存不会提交给主循环。
已经确认完成的响应不会因关闭连接时出现网络错误而再次重放。
本地重试不会重复执行工具，但中断前服务端可能已经生成并计入用量；重试并不保证只消耗一次模型额度。

故障注入测试位于 `tests/test_openai_oauth_resilience.py`，不真实运行论文实验。

## 官方依据

- [Sign in with ChatGPT：Errors and recovery](https://developers.openai.com/siwc/token-sharing-open-source/errors-and-recovery)
- [API 错误类型](https://developers.openai.com/api/docs/guides/error-codes)
- [Rate limits：退避、Retry-After 和预算限制](https://developers.openai.com/api/docs/guides/rate-limits)
- [官方 SDK 的可重试状态说明](https://developers.openai.com/api/reference/typescript)
