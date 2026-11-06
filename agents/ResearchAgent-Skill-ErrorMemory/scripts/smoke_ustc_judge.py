"""One-request smoke test for the optional USTC PaperBench judge config."""

from __future__ import annotations

import asyncio
import json
import time

from paperbench.solvers.basicagent.completer import TimeTrackingRetryConfig
from research_agent.ustc_judge import USTCDeepSeekFlashJudgeCompleterConfig


async def main() -> None:
    config = USTCDeepSeekFlashJudgeCompleterConfig(
        response_format={"type": "json_object"},
        max_tokens=128,
        retry_config=TimeTrackingRetryConfig(wait_min=1, wait_max=5, stop_after=60),
    )
    started = time.monotonic()
    completion = await config.build().async_completion(
        conversation=[
            {"role": "system", "content": "Return only a valid JSON object."},
            {"role": "user", "content": 'Return exactly {"score": 1}.'},
        ]
    )
    content = completion.output_messages[0].content
    if not isinstance(content, str) or json.loads(content).get("score") != 1:
        raise RuntimeError("USTC structured response was not the expected JSON object")
    print(f"USTC_JUDGE_JSON_OK elapsed_seconds={time.monotonic() - started:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
