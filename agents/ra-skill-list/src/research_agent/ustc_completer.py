"""USTC DeepSeek-Flash Chat Completions adapter for the standalone agent."""

from __future__ import annotations

import asyncio
import functools
import os
from typing import Any, Literal, Unpack

import openai
import tiktoken
from openai import NOT_GIVEN, LengthFinishReasonError, NotGiven
from openai.types.chat import ChatCompletion, ChatCompletionToolParam
from openai.types.completion_usage import CompletionUsage
from preparedness_turn_completer.turn_completer import TurnCompleter
from preparedness_turn_completer.utils import RetryConfig, warn_about_non_empty_params
from pydantic import ConfigDict, Field

from paperbench.solvers.basicagent.completer import (
    BasicAgentTurnCompleterConfig,
    TimeTrackingRetryConfig,
)

from .glm_completer import _to_chat_tool
from .ustc_judge import (
    USTC_BASE_URL,
    USTC_DEEPSEEK_FLASH_CONTEXT_WINDOW,
    USTC_DEEPSEEK_FLASH_MODEL,
)


class USTCDeepSeekFlashTurnCompleter(TurnCompleter):
    """OpenAI-compatible USTC client with Agent tools and retry accounting."""

    def __init__(
        self,
        *,
        model: str = USTC_DEEPSEEK_FLASH_MODEL,
        base_url: str = USTC_BASE_URL,
        api_key_env: str = "USTC_LLM_API_KEY",
        context_window: int = USTC_DEEPSEEK_FLASH_CONTEXT_WINDOW,
        temperature: float | None | NotGiven = NOT_GIVEN,
        max_tokens: int | None | NotGiven = NOT_GIVEN,
        top_p: float | None | NotGiven = NOT_GIVEN,
        tools: list[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        tool_choice: Any | NotGiven = NOT_GIVEN,
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
        self.retry_config = retry_config or TimeTrackingRetryConfig()
        self.encoding_name = "o200k_base"
        tiktoken.get_encoding(self.encoding_name)

    class Completion(TurnCompleter.Completion):
        usage: CompletionUsage | None = None

    @functools.cached_property
    def _client(self) -> openai.AsyncOpenAI:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{self.api_key_env} is required for the ustc-deepseek-flash agent config"
            )
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
        # Preserve the existing bounded workaround for transient USTC gateway
        # 403 responses while ordinary network/429/5xx retries use RetryConfig.
        for permission_attempt in range(3):
            try:
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
                        )
                break
            except openai.PermissionDeniedError:
                if permission_attempt == 2:
                    raise
                await asyncio.sleep(5 * (permission_attempt + 1))
        if completion is None or not completion.choices:
            raise RuntimeError("USTC agent completion returned no choices")
        if completion.choices[0].finish_reason == "length":
            raise LengthFinishReasonError(completion=completion)
        return self.Completion(
            input_conversation=conversation,
            output_messages=[completion.choices[0].message],
            usage=completion.usage,
        )


class USTCDeepSeekFlashTurnCompleterConfig(BasicAgentTurnCompleterConfig):
    """Serializable role-local USTC DeepSeek-Flash agent configuration."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        json_encoders={NotGiven: lambda _value: "NOT_GIVEN"},
    )

    provider: Literal["ustc-deepseek-flash"] = "ustc-deepseek-flash"
    model: str = USTC_DEEPSEEK_FLASH_MODEL
    base_url: str = USTC_BASE_URL
    api_key_env: str = "USTC_LLM_API_KEY"
    context_window: int = USTC_DEEPSEEK_FLASH_CONTEXT_WINDOW
    temperature: float | None | NotGiven = NOT_GIVEN
    max_tokens: int | None | NotGiven = NOT_GIVEN
    top_p: float | None | NotGiven = NOT_GIVEN
    tool_choice: Any | NotGiven = NOT_GIVEN
    retry_config: RetryConfig = Field(default_factory=TimeTrackingRetryConfig)

    def build(self) -> USTCDeepSeekFlashTurnCompleter:
        tools: list[ChatCompletionToolParam] | NotGiven = NOT_GIVEN
        if self.basicagent_tools is not None:
            tools = [_to_chat_tool(tool) for tool in self.basicagent_tools]
        return USTCDeepSeekFlashTurnCompleter(
            model=self.model,
            base_url=self.base_url,
            api_key_env=self.api_key_env,
            context_window=self.context_window,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            tools=tools,
            tool_choice=self.tool_choice,
            retry_config=self.retry_config,
        )
