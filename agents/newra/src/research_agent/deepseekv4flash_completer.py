"""Official DeepSeek V4 Flash adapter, with thinking disabled."""
from typing import Literal
from openai import NOT_GIVEN
from .qwen37flash_completer import Qwen37FlashTurnCompleter, Qwen37FlashTurnCompleterConfig
from .glm_completer import _to_chat_tool

class DeepSeekV4FlashTurnCompleter(Qwen37FlashTurnCompleter):
    def __init__(self, **kwargs):
        kwargs.setdefault("model", "deepseek-v4-flash")
        kwargs.setdefault("base_url", "https://api.deepseek.com")
        kwargs.setdefault("api_key_env", "DEEPSEEK_API_KEY")
        if kwargs.get("thinking", "disabled") != "disabled":
            raise ValueError("This adapter requires thinking=disabled")
        super().__init__(**kwargs)

    def _extra_body(self) -> dict:
        return {"thinking": {"type": "disabled"}}

class DeepSeekV4FlashTurnCompleterConfig(Qwen37FlashTurnCompleterConfig):
    provider: Literal["deepseekv4flash"] = "deepseekv4flash"
    model: str = "deepseek-v4-flash"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    thinking: Literal["disabled"] = "disabled"

    def build(self) -> DeepSeekV4FlashTurnCompleter:
        tools = NOT_GIVEN if self.basicagent_tools is None else [_to_chat_tool(t) for t in self.basicagent_tools]
        return DeepSeekV4FlashTurnCompleter(
            model=self.model, base_url=self.base_url, api_key_env=self.api_key_env,
            context_window=self.context_window, thinking=self.thinking,
            temperature=self.temperature, max_tokens=self.max_tokens, top_p=self.top_p,
            tools=tools, tool_choice=self.tool_choice, retry_config=self.retry_config,
        )
