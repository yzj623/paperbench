"""Optional USTC DeepSeek-Flash completer for PaperBench's official judge."""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
from typing import Any, Unpack

import openai
from openai import NOT_GIVEN, LengthFinishReasonError
from openai.types.chat import ChatCompletion
from openai.types.completion_usage import CompletionUsage
from preparedness_turn_completer.oai_completions_turn_completer import (
    OpenAICompletionsTurnCompleter,
)
from preparedness_turn_completer.turn_completer import TurnCompleter
from preparedness_turn_completer.utils import RetryConfig, warn_about_non_empty_params
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from paperbench.solvers.basicagent.completer import TimeTrackingRetryConfig


USTC_BASE_URL = "https://api.llm.ustc.edu.cn/v1"
USTC_DEEPSEEK_FLASH_MODEL = "deepseek-flash"
USTC_DEEPSEEK_FLASH_CONTEXT_WINDOW = 128_000
logger = logging.getLogger(__name__)


# PaperBench recognizes this interface when collecting completion token usage.
# Keep the USTC transport and configuration overrides below.
class USTCDeepSeekJudgeCompleter(OpenAICompletionsTurnCompleter):
    """Chat Completions client used only by the optional USTC judge config."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key_env: str,
        context_window: int,
        response_format: dict[str, Any] | None,
        temperature: float | None,
        thinking: dict[str, str] | None,
        max_tokens: int | None,
        top_p: float | None,
        retry_config: RetryConfig,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.n_ctx = context_window
        self.context_window = context_window
        self.response_format = response_format
        self.temperature = temperature
        self.thinking = thinking
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.retry_config = retry_config
        self.encoding_name = "o200k_base"

    class Completion(TurnCompleter.Completion):
        usage: CompletionUsage | None = None

    @functools.cached_property
    def _client(self) -> openai.AsyncOpenAI:
        api_key = os.getenv(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{self.api_key_env} is required for the ustc-deepseek-flash grader config"
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
        # USTC documents 403 as an authorization error, but its shared gateway
        # has historically also emitted transient 403 responses. Keep the
        # workaround bounded to three attempts and only in this optional config.
        for permission_attempt in range(3):
            try:
                async for attempt in self.retry_config.build():
                    with attempt:
                        completion = await self._client.chat.completions.create(
                            model=self.model,
                            messages=conversation,
                            # USTC DF2 JSON mode can return content=None and put
                            # the final JSON in reasoning_content. Request plain
                            # output; PaperBench still validates the JSON itself.
                            response_format=(
                                NOT_GIVEN
                                if self.model == "deepseek-flash-2"
                                else self.response_format or NOT_GIVEN
                            ),
                            temperature=(
                                self.temperature if self.temperature is not None else NOT_GIVEN
                            ),
                            max_tokens=self.max_tokens if self.max_tokens is not None else NOT_GIVEN,
                            top_p=self.top_p if self.top_p is not None else NOT_GIVEN,
                            **({"extra_body": {"thinking": self.thinking}}
                               if self.thinking is not None else {}),
                        )
                break
            except openai.PermissionDeniedError:
                if permission_attempt == 2:
                    raise
                await asyncio.sleep(5 * (permission_attempt + 1))
        if completion is None or not completion.choices:
            raise RuntimeError("USTC judge completion returned no choices")
        if completion.choices[0].finish_reason == "length":
            raise LengthFinishReasonError(completion=completion)
        message = completion.choices[0].message
        if self.model == "deepseek-flash-2" and self.response_format and message.content:
            # Plain mode may wrap structured JSON in a Markdown code fence.
            # Unwrap only a whole, valid JSON object; preserve all field values.
            lines = message.content.strip().splitlines()
            if (
                len(lines) >= 3
                and lines[0].strip().casefold() in ("\x60\x60\x60json", "\x60\x60\x60")
                and lines[-1].strip() == "\x60\x60\x60"
            ):
                candidate = "\n".join(lines[1:-1]).strip()
                try:
                    parsed = json.loads(candidate)
                except ValueError:
                    pass
                else:
                    if isinstance(parsed, dict):
                        message = message.model_copy(update={"content": candidate})
        return self.Completion(
            input_conversation=conversation,
            output_messages=[message],
            usage=completion.usage,
        )


class USTCDeepSeekStructuredJudgeCompleter(USTCDeepSeekJudgeCompleter):
    """Retry malformed PaperBench score-parser output, not the grading judgment."""

    parse_response_type: type[BaseModel] | None = None

    async def async_completion(self, conversation: Any, **params: Any) -> Any:
        if self.parse_response_type is None:
            return await super().async_completion(conversation, **params)

        messages = [dict(message) for message in conversation]
        messages[0]["content"] += (
            " Return exactly one valid JSON object with only these keys: "
            "valid_score (boolean), score (using the range specified above), "
            "and explanation (one short sentence). "
            "Do not include Markdown code fences or any other text."
        )
        usages = []

        def with_total_usage(result: Any) -> Any:
            if len(usages) < 2:
                return result
            total = usages[-1].model_copy(update={
                "prompt_tokens": sum(usage.prompt_tokens for usage in usages),
                "completion_tokens": sum(usage.completion_tokens for usage in usages),
                "total_tokens": sum(usage.total_tokens for usage in usages),
            })
            return result.model_copy(update={"usage": total})

        for attempt in range(3):
            result = await super().async_completion(messages, **params)
            if result.usage is not None:
                usages.append(result.usage)
            content = result.output_messages[0].content
            try:
                self.parse_response_type.model_validate_json(content)
            except ValidationError:
                if attempt == 2:
                    return with_total_usage(result)  # Preserve PaperBench's final failure behavior.
                logger.warning("USTC DF2 score parser returned invalid JSON; retrying (%d/3)", attempt + 1)
                messages.extend([
                    {"role": "assistant", "content": content or ""},
                    {"role": "user", "content": "That response could not be parsed as the required JSON object. "
                     "Return only the JSON object with valid_score, score, and explanation."},
                ])
            else:
                return with_total_usage(result)
        raise AssertionError("unreachable")


class USTCDeepSeekFlashJudgeCompleterConfig(TurnCompleter.Config):
    """Serializable PaperBench judge configuration for USTC DeepSeek-Flash."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = "deepseek-flash-2"
    base_url: str = USTC_BASE_URL
    api_key_env: str = "USTC_LLM_API_KEY"
    context_window: int = USTC_DEEPSEEK_FLASH_CONTEXT_WINDOW
    response_format: dict[str, Any] | None = None
    temperature: float | None = 0.0
    thinking: dict[str, str] | None = {"type": "disabled"}
    max_tokens: int | None = None
    top_p: float | None = None
    retry_config: RetryConfig = Field(default_factory=TimeTrackingRetryConfig)

    def build(self) -> USTCDeepSeekStructuredJudgeCompleter:
        return USTCDeepSeekStructuredJudgeCompleter(
            model=self.model,
            base_url=self.base_url,
            api_key_env=self.api_key_env,
            context_window=self.context_window,
            response_format=self.response_format,
            temperature=self.temperature,
            thinking=self.thinking,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            retry_config=self.retry_config,
        )


def install_ustc_structured_judge_config() -> None:
    """Make PaperBench's helper graders reuse the selected USTC endpoint.

    PaperBench normally creates separate GPT-4o clients for integer and float
    structured responses. This runtime-only hook changes no prompt or scoring
    behavior; it routes those two helper calls through USTC DF2, validating
    plain-text JSON and retrying malformed parser responses up to three times.
    """

    from paperbench.judge.simple import SimpleJudge

    if getattr(SimpleJudge, "_research_agent_ustc_installed", False):
        return
    original = SimpleJudge._init_structured_completer

    def _init_structured_completer(
        judge: Any,
        config: TurnCompleter.Config | None,
        response_format: type[Any],
    ) -> tuple[TurnCompleter.Config, TurnCompleter]:
        if config is None and isinstance(
            judge.completer_config, USTCDeepSeekFlashJudgeCompleterConfig
        ):
            selected = judge.completer_config.model_copy(update={"response_format": None})
            completer = selected.build()
            completer.parse_response_type = response_format
            return selected, completer
        return original(judge, config, response_format)

    SimpleJudge._init_structured_completer = _init_structured_completer
    SimpleJudge._research_agent_ustc_installed = True
