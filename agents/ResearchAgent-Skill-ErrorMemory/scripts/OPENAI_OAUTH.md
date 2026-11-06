# 独立的 ChatGPT-plan OAuth 测试

授权脚本只完成授权、模型查询和极小的 Responses 请求。RA 已新增独立的
`openai-oauth` provider；原 `openai` API-key、Qwen、DeepSeek 调用链保持不变。

在 ResearchAgent 目录执行（可选添加全局 `--proxy http://代理地址:端口`）：

```bash
.venv-wsl/bin/python scripts/openai_oauth.py login --probe-model gpt-6-luna
.venv-wsl/bin/python scripts/openai_oauth.py status
.venv-wsl/bin/python scripts/openai_oauth.py models
.venv-wsl/bin/python scripts/openai_oauth.py probe --model gpt-6-luna
```

授权时在本机浏览器打开脚本打印的链接；回调监听 127.0.0.1:1455。
普通 RA 和 RA-Skill 共用 `/root/.config/ppblab/openai-oauth/credentials.json`
（一般为 `$XDG_CONFIG_HOME/ppblab/openai-oauth/` 或 `~/.config/ppblab/openai-oauth/`）。
目录权限 0700，文件权限 0600，位于代码仓库之外。不要分享、上传此目录。
刷新采用共享文件锁和原子替换，遵守服务器刷新时间。
脚本会验证 ID token 签名、issuer、audience、nonce 和订阅 API 权限。
账户必须实际获授所需权限；Plus 登录本身不等于已经成功完成 API 测试。
模型不存在时不会自动替换。测试请求须收到 `response.completed` 才认定完成。

## RA 使用方式

```bash
RESEARCH_AGENT_API_CONFIG=openai-oauth \
RESEARCH_AGENT_MODEL=gpt-6-luna \
RESEARCH_AGENT_OPENAI_PROXY=http://172.22.224.1:7890 \
bash scripts/run_experiment.sh mechanistic-understanding
```

代理地址请按当前 WSL/Clash 配置设置；不需要代理时省略代理变量。
模型名可自由指定；省略 `RESEARCH_AGENT_MODEL` 时默认 `gpt-6-luna`。
`gpt6luna` 也会转换为官方 slug `gpt-6-luna`，其他名称不替换。
可用 `RESEARCH_AGENT_OPENAI_OAUTH_DIR` 或 `OPENAI_OAUTH_DIR` 指定其他授权目录，
两套代码应指向同一个目录，而不是各保存一份令牌副本。
新环境需安装 `pip install -e '.[openai-oauth]'`，以及项目原有的 PaperBench 依赖。

planner、主 agent、checker 都使用该模型与共享凭据，工具列表仍各自独立。
每次调用前从文件读取凭据，距离过期不足 5 分钟时自动刷新。
文件锁跨进程串行刷新；成功后访问凭据、旋转刷新凭据和过期时间原子保存。
收到 401 时最多刷新并重试一次；如果其他角色已刷新，则直接复用其新凭据。
服务器不允许提前刷新时，继续使用仍有效的访问凭据。
临时刷新失败不会覆盖原凭据；授权失效需重新登录。

使用公共 `/v1/responses`、流式响应和不保存服务端会话；每次发送 RA 当前完整消息。
系统消息仅在接口转换层映射为 `developer`，内容、位置和 RA 的裁剪机制不变。
协议工具按 API 要求封装在 namespace 中，返回后转换回 RA 原工具名称和 call ID。
不发送订阅接口不支持的 temperature、top_p、max_output_tokens 参数。
默认沿用 RA 的 128k 本地裁剪预算（不是对所有 GPT 模型原生上限的声明）。
评分仍使用原评分器，不随 agent provider 切换。

网络、流式与服务端错误的防御范围和重试上限见 [OPENAI_RETRY_POLICY.md](OPENAI_RETRY_POLICY.md)。

只验证适配器的工具调用，不运行论文实验：

```bash
.venv-wsl/bin/python scripts/smoke_openai_oauth.py --proxy http://172.22.224.1:7890
```

官方依据：
- https://developers.openai.com/siwc/token-sharing-open-source/sign-in
- https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference
- https://developers.openai.com/siwc/token-sharing-open-source/token-reference
