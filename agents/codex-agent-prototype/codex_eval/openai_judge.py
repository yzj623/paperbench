"""Self-contained OAuth Responses judge for the Codex PaperBench launcher."""
from __future__ import annotations

from typing import Any, Literal

from openai.types.chat import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
from preparedness_turn_completer.oai_completions_turn_completer import OpenAICompletionsTurnCompleter
from preparedness_turn_completer.turn_completer import TurnCompleter
from pydantic import Field, ValidationError

from .openai_oauth import OAuthError
from .openai_oauth_completer import (
    OpenAIOAuthTurnCompleter, default_credentials_dir, default_proxy,
)
from .openai_oauth_resilience import OpenAIOAuthRetryConfig


class InvalidJudgeOutput(OAuthError):
    """A completed response did not contain usable judge text or valid schema JSON."""


class OpenAIOAuthJudgeCompleter(OpenAIOAuthTurnCompleter, OpenAICompletionsTurnCompleter):
    # Retain PaperBench's existing Completion/usage interface for token accounting.
    Completion = OpenAICompletionsTurnCompleter.Completion

    def __init__(self, *, response_format=None, **kwargs):
        super().__init__(**kwargs)
        self.response_format = response_format

    def _request_body(self, conversation):
        # Map one leading system message to Responses' instructions field,
        # preserving its exact text. Never replace or append judge instructions.
        instructions = None
        if (conversation and conversation[0]["role"] == "system"
                and isinstance(conversation[0].get("content"), str)
                and sum(message["role"] == "system" for message in conversation) == 1):
            instructions = conversation[0]["content"]
            conversation = conversation[1:]
        body = super()._request_body(conversation)
        if instructions is not None:
            body["instructions"] = instructions
        if self.response_format is not None:
            body["text"] = {"format": self.response_format}
        return body

    async def async_completion(self, conversation, **params):
        # Network/auth retries are inherited; malformed completed responses get
        # at most two additional attempts, without fabricating missing fields.
        for attempt in range(3):
            try:
                return await super().async_completion(conversation, **params)
            except InvalidJudgeOutput:
                if attempt == 2:
                    raise
        raise AssertionError("unreachable")

    def _to_completion(self, conversation, response):
        texts, refused = [], False
        for item in response.output:
            if item.type == "reasoning":
                continue
            if item.type != "message":
                raise InvalidJudgeOutput("Unexpected non-message output in a tool-free judge request")
            for part in item.content:
                if part.type == "refusal":
                    refused = True
                elif part.type == "output_text":
                    texts.append(part.text)
        if refused:
            raise OAuthError("OpenAI judge refused the request; no score was synthesized")
        text = "\n".join(texts).strip()
        if not text:
            raise InvalidJudgeOutput("OpenAI judge completed without text")
        if self.response_format and self.response_format.get("type") == "json_schema":
            from paperbench.judge.simple import ParsedJudgeResponseInt, ParsedJudgeResponseFloat

            schema = self.response_format["schema"]
            model = (ParsedJudgeResponseInt if schema["properties"]["score"]["type"] == "integer"
                     else ParsedJudgeResponseFloat)
            def validate(candidate):
                value = model.model_validate_json(candidate)
                if not 0 <= value.score <= 1:
                    raise ValueError("Score outside PaperBench's range")
                return value
            try:
                validate(text)
            except (ValueError, ValidationError):
                # Responses can contain several message items. Accept only an
                # unambiguous, individually valid JSON message; never choose
                # between conflicting structured verdicts.
                valid = []
                for candidate in texts:
                    try:
                        validate(candidate)
                        valid.append(candidate)
                    except (ValueError, ValidationError):
                        pass
                if len(valid) != 1:
                    raise InvalidJudgeOutput("Judge output did not satisfy the required score schema") from None
                text = valid[0]
        usage = None
        if response.usage:
            source = response.usage
            usage = CompletionUsage(
                prompt_tokens=source.input_tokens, completion_tokens=source.output_tokens,
                total_tokens=source.total_tokens,
                prompt_tokens_details={"cached_tokens": getattr(source.input_tokens_details, "cached_tokens", 0)},
                completion_tokens_details={"reasoning_tokens": getattr(source.output_tokens_details, "reasoning_tokens", 0)},
            )
        return self.Completion(input_conversation=conversation,
                               output_messages=[ChatCompletionMessage(role="assistant", content=text)],
                               usage=usage)


class OpenAIOAuthJudgeCompleterConfig(TurnCompleter.Config):
    model: str = "gpt-6-luna"
    reasoning_effort: Literal["low", "medium"] = "medium"
    # Preserve the previous judge's code-selection budget for comparability.
    context_window: int = 128_000
    credentials_dir: str = Field(default_factory=default_credentials_dir)
    proxy: str | None = Field(default_factory=default_proxy)
    response_format: dict[str, Any] | None = None
    retry_config: OpenAIOAuthRetryConfig = Field(default_factory=OpenAIOAuthRetryConfig)

    def build(self):
        return OpenAIOAuthJudgeCompleter(
            model=self.model, credentials_dir=self.credentials_dir, proxy=self.proxy,
            context_window=self.context_window, reasoning={"effort": self.reasoning_effort},
            retry_config=self.retry_config, response_format=self.response_format,
        )


def install_openai_structured_judge_config():
    """Use the same OAuth model and strict schema for integer/float parsing."""
    from paperbench.judge.simple import SimpleJudge

    if getattr(SimpleJudge, "_codex_eval_openai_installed", False):
        return
    original = SimpleJudge._init_structured_completer

    def init(judge, config, response_format):
        if config is None and isinstance(judge.completer_config, OpenAIOAuthJudgeCompleterConfig):
            schema = response_format.model_json_schema()
            schema["additionalProperties"] = False
            config = judge.completer_config.model_copy(update={"response_format": {
                "type": "json_schema", "name": response_format.__name__,
                "schema": schema, "strict": True,
            }})
            return config, config.build()
        return original(judge, config, response_format)

    SimpleJudge._init_structured_completer = init
    SimpleJudge._codex_eval_openai_installed = True
