"""One-request GLM-4.7-Flash tool-calling smoke test.

Run with ``ZAI_API_KEY`` in the environment. The credential is never printed.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from openai.types.responses import FunctionToolParam

from paperbench.solvers.basicagent.api import make_completer_request
from paperbench.solvers.basicagent.completer import TimeTrackingRetryConfig
from paperbench.solvers.basicagent.tools.base import Tool
from research_agent.glm_completer import GLM47FlashTurnCompleterConfig


class EchoProbeTool(Tool):
    def name(self) -> str:
        return "echo_probe"

    async def execute(self, computer: Any, **kwargs: Any) -> str:
        return str(kwargs)

    def get_oai_tool_call(self) -> FunctionToolParam:
        return FunctionToolParam(
            type="function",
            name=self.name(),
            description="Return the supplied probe value.",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
            strict=False,
        )


async def main() -> None:
    config = GLM47FlashTurnCompleterConfig(
        basicagent_tools=[EchoProbeTool()],
        tool_choice={"type": "function", "function": {"name": "echo_probe"}},
        max_tokens=256,
        retry_config=TimeTrackingRetryConfig(wait_min=1, wait_max=5, stop_after=60),
    )
    started = time.monotonic()
    initial_messages = [
        {"role": "system", "content": "Follow the requested tool protocol exactly."},
        {"role": "user", "content": "Call echo_probe once with value GLM_SMOKE_OK."},
    ]
    response = await make_completer_request(
        completer=config.build(),
        messages=initial_messages,
    )
    if len(response.tool_calls) != 1:
        raise RuntimeError(f"expected one tool call, received {len(response.tool_calls)}")
    call = response.tool_calls[0]
    if call.name != "echo_probe" or call.arguments.get("value") != "GLM_SMOKE_OK":
        raise RuntimeError(f"unexpected tool call: {call.name} {call.arguments!r}")
    followup = await make_completer_request(
        completer=GLM47FlashTurnCompleterConfig(
            basicagent_tools=[EchoProbeTool()],
            max_tokens=256,
            retry_config=TimeTrackingRetryConfig(wait_min=1, wait_max=5, stop_after=60),
        ).build(),
        messages=initial_messages
        + response.response_messages
        + [
            {
                "role": "tool",
                "tool_call_id": call.call_id,
                "content": "GLM_SMOKE_OK. Finish with a short confirmation and no tool call.",
            }
        ],
    )
    if followup.tool_calls:
        raise RuntimeError("follow-up unexpectedly requested another tool")
    if not followup.response_messages[-1].get("content"):
        raise RuntimeError("follow-up returned no assistant content")
    print(f"GLM_MULTI_TURN_OK elapsed_seconds={time.monotonic() - started:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
