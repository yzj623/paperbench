"""Immutable API parameters shared by ResearchAgent roles.

The OpenAI baseline uses PaperBench's unmodified official completer. Optional
providers create fresh role-local configs for the planner, main agent, and each
checker so tool lists cannot leak between roles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

from openai import NOT_GIVEN, NotGiven


@dataclass(frozen=True)
class APISettings:
    """Pure API parameters from which every role receives a new config."""

    model: str
    provider: Literal[
        "openai", "openai-oauth", "glm47flash", "ustc-deepseek-flash", "qwen37flash", "deepseekv4flash"
    ] = "openai"
    base_url: str | None = None
    api_key_env: str | None = None
    context_window: int | None = None
    thinking: Literal["enabled", "disabled"] | None = None
    reasoning: Any | None | NotGiven = NOT_GIVEN
    temperature: float | None | NotGiven = NOT_GIVEN
    max_output_tokens: int | None | NotGiven = NOT_GIVEN
    top_p: float | None | NotGiven = NOT_GIVEN
    credentials_dir: str | None = None
    proxy: str | None = None

    @classmethod
    def from_config(cls, config: Any) -> "APISettings":
        return cls(
            model=str(config.model),
            provider=getattr(config, "provider", "openai"),
            base_url=getattr(config, "base_url", None),
            api_key_env=getattr(config, "api_key_env", None),
            context_window=getattr(config, "context_window", None),
            thinking=getattr(config, "thinking", None),
            reasoning=getattr(config, "reasoning", NOT_GIVEN),
            temperature=getattr(config, "temperature", NOT_GIVEN),
            max_output_tokens=getattr(
                config, "max_output_tokens", getattr(config, "max_tokens", NOT_GIVEN)
            ),
            top_p=getattr(config, "top_p", NOT_GIVEN),
            credentials_dir=getattr(config, "credentials_dir", None),
            proxy=getattr(config, "proxy", None),
        )

    def new_config(
        self,
        *,
        basicagent_tools: Sequence[Any] | None,
        response_format: dict[str, Any] | None = None,
    ) -> Any:
        """Create one role-local provider configuration.

        ``response_format`` remains in the internal interface for compatibility.
        ResearchAgent uses a terminal protocol tool for the checker's structured
        output, so provider-specific JSON response formatting stays unset.
        """

        if response_format is not None:
            raise ValueError(
                "ResearchAgent uses protocol tools, not response_format"
            )
        if self.provider == "openai-oauth":
            from .openai_oauth_completer import OpenAIOAuthTurnCompleterConfig, default_credentials_dir
            return OpenAIOAuthTurnCompleterConfig(
                model=self.model, credentials_dir=self.credentials_dir or default_credentials_dir(),
                proxy=self.proxy, context_window=self.context_window or 128_000,
                reasoning=self.reasoning,
                basicagent_tools=list(basicagent_tools) if basicagent_tools is not None else None,
            )

        if self.provider == "glm47flash":
            from .glm_completer import GLM47FlashTurnCompleterConfig

            return GLM47FlashTurnCompleterConfig(
                model=self.model,
                base_url=self.base_url or "https://open.bigmodel.cn/api/paas/v4/",
                api_key_env=self.api_key_env or "ZAI_API_KEY",
                context_window=self.context_window or 200_000,
                thinking=self.thinking or "enabled",
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                top_p=self.top_p,
                basicagent_tools=(
                    list(basicagent_tools) if basicagent_tools is not None else None
                ),
            )

        if self.provider == "ustc-deepseek-flash":
            from .ustc_completer import USTCDeepSeekFlashTurnCompleterConfig

            return USTCDeepSeekFlashTurnCompleterConfig(
                model=self.model,
                base_url=self.base_url or "https://api.llm.ustc.edu.cn/v1",
                api_key_env=self.api_key_env or "USTC_LLM_API_KEY",
                context_window=self.context_window or 128_000,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                top_p=self.top_p,
                basicagent_tools=(
                    list(basicagent_tools) if basicagent_tools is not None else None
                ),
            )

        if self.provider == "deepseekv4flash":
            from .deepseekv4flash_completer import DeepSeekV4FlashTurnCompleterConfig
            return DeepSeekV4FlashTurnCompleterConfig(
                model=self.model, base_url=self.base_url or "https://api.deepseek.com",
                api_key_env=self.api_key_env or "DEEPSEEK_API_KEY",
                context_window=self.context_window or 128_000, thinking="disabled",
                temperature=self.temperature, max_tokens=self.max_output_tokens, top_p=self.top_p,
                basicagent_tools=list(basicagent_tools) if basicagent_tools is not None else None,
            )

        if self.provider == "qwen37flash":
            from .qwen37flash_completer import Qwen37FlashTurnCompleterConfig

            return Qwen37FlashTurnCompleterConfig(
                model=self.model,
                base_url=self.base_url
                or "https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key_env=self.api_key_env or "DASHSCOPE_API_KEY",
                context_window=self.context_window or 128_000,
                thinking=self.thinking or "disabled",
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                top_p=self.top_p,
                basicagent_tools=(
                    list(basicagent_tools) if basicagent_tools is not None else None
                ),
            )

        from paperbench.solvers.basicagent.completer import OpenAIResponsesTurnCompleterConfig

        return OpenAIResponsesTurnCompleterConfig(
            model=self.model,
            reasoning=self.reasoning,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
            top_p=self.top_p,
            tools=NOT_GIVEN,
            basicagent_tools=(
                list(basicagent_tools) if basicagent_tools is not None else None
            ),
        )


# Backwards-compatible internal name used by the copied protocol modules. It
# no longer denotes a third-party compatible endpoint.
CompatibleAPISettings = APISettings
