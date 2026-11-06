"""PaperBench judge configuration for DeepSeek's official V4.1 Flash API."""

import logging
from typing import Any

from pydantic import BaseModel, ValidationError
from preparedness_turn_completer.turn_completer import TurnCompleter

from .ustc_judge import USTCDeepSeekFlashJudgeCompleterConfig, USTCDeepSeekJudgeCompleter


logger = logging.getLogger(__name__)


class DeepSeekOfficialJudgeCompleter(USTCDeepSeekJudgeCompleter):
    """Retry only malformed responses from PaperBench's score parser."""

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
            total = usages[-1].model_copy(
                update={
                    "prompt_tokens": sum(usage.prompt_tokens for usage in usages),
                    "completion_tokens": sum(usage.completion_tokens for usage in usages),
                    "total_tokens": sum(usage.total_tokens for usage in usages),
                }
            )
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
                    return with_total_usage(result)  # Let PaperBench handle the final parse failure.
                logger.warning("DeepSeek score parser returned invalid JSON; retrying (%d/3)", attempt + 1)
                messages.extend(
                    [
                        {"role": "assistant", "content": content or ""},
                        {
                            "role": "user",
                            "content": "That response could not be parsed as the required JSON object. "
                            "Return only the JSON object with valid_score, score, and explanation.",
                        },
                    ]
                )
            else:
                return with_total_usage(result)
        raise AssertionError("unreachable")


class DeepSeekOfficialFlashJudgeCompleterConfig(USTCDeepSeekFlashJudgeCompleterConfig):
    # DeepSeek names the current V4.1 Flash release `deepseek-flash` on its API.
    model: str = "deepseek-flash"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    thinking: dict[str, str] | None = {"type": "disabled"}
    temperature: float | None = 0.0

    def build(self) -> DeepSeekOfficialJudgeCompleter:
        return DeepSeekOfficialJudgeCompleter(
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


def install_deepseek_official_structured_judge_config() -> None:
    """Use the official endpoint for PaperBench's integer and float parsers.

    The parser asks for JSON in ordinary text mode and validates its schema.
    """
    from paperbench.judge.simple import SimpleJudge

    if getattr(SimpleJudge, "_research_agent_deepseek_official_installed", False):
        return
    original = SimpleJudge._init_structured_completer

    def _init_structured_completer(
        judge: Any,
        config: TurnCompleter.Config | None,
        response_format: type[Any],
    ) -> tuple[TurnCompleter.Config, TurnCompleter]:
        if config is None and isinstance(
            judge.completer_config, DeepSeekOfficialFlashJudgeCompleterConfig
        ):
            selected = judge.completer_config.model_copy(update={"response_format": None})
            completer = selected.build()
            completer.parse_response_type = response_format
            return selected, completer
        return original(judge, config, response_format)

    SimpleJudge._init_structured_completer = _init_structured_completer
    SimpleJudge._research_agent_deepseek_official_installed = True
