#!/usr/bin/env python3
"""Tiny real tool-call round trip; no Docker, paper, training, or grading."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import asyncio
import json

from paperbench.solvers.basicagent.api import make_completer_request
from paperbench.solvers.basicagent.completer import TimeTrackingRetryConfig
from paperbench.solvers.basicagent.tools.base import Tool
from research_agent.openai_oauth_completer import OpenAIOAuthTurnCompleterConfig


class EchoProbeTool(Tool):
    def name(self):
        return "echo_probe"

    async def execute(self, computer, *, value):
        return value

    def get_oai_tool_call(self):
        return {"type": "function", "name": self.name(), "description": "Return the supplied probe value.",
                "strict": False, "parameters": {"type": "object", "properties": {"value": {"type": "string"}},
                                                  "required": ["value"], "additionalProperties": False}}


async def run(args):
    settings = {"model": args.model, "basicagent_tools": [EchoProbeTool()],
                "retry_config": TimeTrackingRetryConfig(wait_min=1, wait_max=3, stop_after=30)}
    if args.proxy:
        settings["proxy"] = args.proxy
    c = OpenAIOAuthTurnCompleterConfig(**settings).build()
    messages = [{"role": "system", "content": "Follow the user's tool protocol exactly."},
                {"role": "user", "content": "Call echo_probe exactly once with value OAUTH_SMOKE_OK. After its result, reply exactly OK without any more tools."}]
    try:
        first = await make_completer_request(completer=c, messages=messages)
        if len(first.tool_calls) != 1:
            raise RuntimeError("Expected exactly one echo probe tool call")
        call = first.tool_calls[0]
        if call.name != "echo_probe" or call.arguments != {"value": "OAUTH_SMOKE_OK"}:
            raise RuntimeError("Unexpected probe arguments")
        messages.extend(first.response_messages)
        messages.append({"role": "tool", "tool_call_id": call.call_id,
                         "content": await EchoProbeTool().execute(None, **call.arguments)})
        second = await make_completer_request(completer=c, messages=messages)
        text = "".join(m.get("content") or "" for m in second.response_messages)
        if second.tool_calls or text.strip() != "OK":
            raise RuntimeError("Tool round trip did not return the requested final OK")
        print(json.dumps({"passed": True, "model": c.model, "tool_round_trip": True, "response": text}))
    finally:
        await c._client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--proxy")
    asyncio.run(run(parser.parse_args()))
