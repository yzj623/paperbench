"""GLM-4.7-Flash Chat Completions adapter for the standalone agent."""

from __future__ import annotations

import functools
import os
from typing import Any, Literal, Unpack

import openai
import tiktoken
from openai import NOT_GIVEN, NotGiven
from openai.types.chat import ChatCompletion, ChatCompletionToolParam
from openai.types.completion_usage import CompletionUsage
from preparedness_turn_completer.turn_completer import TurnCompleter
from preparedness_turn_completer.utils import RetryConfig, warn_about_non_empty_params
from pydantic import ConfigDict, Field

from paperbench.solvers.basicagent.completer import (
    BasicAgentTurnCompleterConfig,
    TimeTrackingRetryConfig,
)
from paperbench.solvers.basicagent.tools.base import Tool


GLM47FLASH_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
GLM47FLASH_MODEL = "glm-4.7-flash"
GLM47FLASH_CONTEXT_WINDOW = 200_000


def _to_chat_tool(tool: Tool) -> ChatCompletionToolParam:
    """Convert PaperBench's Responses-style function schema to Chat Completions."""

    schema = dict(tool.get_oai_tool_call())
    function: dict[str, Any] = {
        "name": schema["name"],
        "description": schema.get("description", ""),
        "parameters": schema.get("parameters", {"type": "object", "properties": {}}),
    }
    if "strict" in schema:
        function["strict"] = schema["strict"]
    return {"type": "function", "function": function}


class GLM47FlashTurnCompleter(TurnCompleter):
    """OpenAI-compatible GLM client with PaperBench's retry accounting."""

    def __init__(
        self,
        *,
        model: str = GLM47FLASH_MODEL,
        base_url: str = GLM47FLASH_BASE_URL,
        api_key_env: str = "ZAI_API_KEY",
        context_window: int = GLM47FLASH_CONTEXT_WINDOW,
        temperature: float | None | NotGiven = NOT_GIVEN,
        max_tokens: int | None | NotGiven = NOT_GIVEN,
        top_p: float | None | NotGiven = NOT_GIVEN,
        tools: list[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        tool_choice: Any | NotGiven = NOT_GIVEN,
        thinking: Literal["enabled", "disabled"] = "enabled",
        retry_config: RetryConfig | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.n_ctx = context_window
        self.context_window = context_window
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.tools = tools
        self.tool_choice = tool_choice
        self.thinking = thinking
        self.retry_config = retry_config or TimeTrackingRetryConfig()
        self.encoding_name = "o200k_base"
        tiktoken.get_encoding(self.encoding_name)

    class Completion(TurnCompleter.Completion):
        usage: CompletionUsage | None = None

    @functools.cached_property
    def _client(self) -> openai.AsyncOpenAI:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} is required for the glm47flash API config")
        return openai.AsyncOpenAI(api_key=api_key, base_url=self.base_url)

    def completion(
        self,
        conversation: TurnCompleter.RuntimeConversation,
        **params: Unpack[TurnCompleter.Params],
    ) -> Completion:
        raise NotImplementedError("Not implemented, use async_completion instead")

    async def async_completion(
        self,
        conversation: TurnCompleter.RuntimeConversation,
        **params: Unpack[TurnCompleter.Params],
    ) -> Completion:
        warn_about_non_empty_params(self, **params)
        completion: ChatCompletion | None = None
        async for attempt in self.retry_config.build():
            with attempt:
                completion = await self._client.chat.completions.create(
                    model=self.model,
                    messages=conversation,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p,
                    tools=self.tools,
                    tool_choice=self.tool_choice,
                    extra_body={
                        "thinking": {
                            "type": self.thinking,
                            # Each Agent turn may include a tool call. Clearing old hidden
                            # reasoning avoids requiring provider-specific chain-of-thought
                            # fields to be replayed in the next request.
                            "clear_thinking": True,
                        }
                    },
                )
        if completion is None:
            raise RuntimeError("GLM completion returned no response")
        return self.Completion(
            input_conversation=conversation,
            output_messages=[completion.choices[0].message],
            usage=completion.usage,
        )


class GLM47FlashTurnCompleterConfig(BasicAgentTurnCompleterConfig):
    """Serializable root/role configuration for GLM-4.7-Flash."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={NotGiven: lambda _value: "NOT_GIVEN"},
    )

    provider: Literal["glm47flash"] = "glm47flash"
    model: str = GLM47FLASH_MODEL
    base_url: str = GLM47FLASH_BASE_URL
    api_key_env: str = "ZAI_API_KEY"
    context_window: int = GLM47FLASH_CONTEXT_WINDOW
    temperature: float | None | NotGiven = NOT_GIVEN
    max_tokens: int | None | NotGiven = NOT_GIVEN
    top_p: float | None | NotGiven = NOT_GIVEN
    thinking: Literal["enabled", "disabled"] = "enabled"
    tool_choice: Any | NotGiven = NOT_GIVEN
    retry_config: RetryConfig = Field(default_factory=TimeTrackingRetryConfig)

    def build(self) -> GLM47FlashTurnCompleter:
        tools: list[ChatCompletionToolParam] | NotGiven = NOT_GIVEN
        if self.basicagent_tools is not None:
            tools = [_to_chat_tool(tool) for tool in self.basicagent_tools]
        return GLM47FlashTurnCompleter(
            model=self.model,
            base_url=self.base_url,
            api_key_env=self.api_key_env,
            context_window=self.context_window,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            tools=tools,
            tool_choice=self.tool_choice,
            thinking=self.thinking,
            retry_config=self.retry_config,
        )
