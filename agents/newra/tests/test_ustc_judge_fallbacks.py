import asyncio

from openai.types.chat import ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
from paperbench.judge.simple import ParsedJudgeResponseInt

from research_agent.ustc_judge import (
    USTCDeepSeekFlashJudgeCompleterConfig,
    USTCDeepSeekJudgeCompleter,
)


def test_df2_structured_parser_repairs_once_and_counts_usage(monkeypatch):
    requests = []
    outputs = ["not json", '{"valid_score":true,"score":1,"explanation":"present"}']

    async def fake_completion(self, conversation, **params):
        requests.append(conversation)
        content = outputs.pop(0)
        return self.Completion(
            input_conversation=conversation,
            output_messages=[ChatCompletionMessage(role="assistant", content=content)],
            usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    monkeypatch.setattr(USTCDeepSeekJudgeCompleter, "async_completion", fake_completion)
    grader = USTCDeepSeekFlashJudgeCompleterConfig().build()
    grader.parse_response_type = ParsedJudgeResponseInt
    result = asyncio.run(grader.async_completion([
        {"role": "system", "content": "Parse a score."},
        {"role": "user", "content": "Score: 1"},
    ]))
    assert len(requests) == 2
    assert requests[0][0]["content"].startswith("Parse a score.")
    assert "JSON" in requests[0][0]["content"]
    assert requests[1][-1]["role"] == "user"
    assert ParsedJudgeResponseInt.model_validate_json(result.output_messages[0].content).score == 1
    assert result.usage.total_tokens == 30


def test_df2_parser_stops_after_three_invalid_responses(monkeypatch):
    calls = []

    async def fake_completion(self, conversation, **params):
        calls.append(conversation)
        return self.Completion(
            input_conversation=conversation,
            output_messages=[ChatCompletionMessage(role="assistant", content="invalid")],
            usage=None,
        )

    monkeypatch.setattr(USTCDeepSeekJudgeCompleter, "async_completion", fake_completion)
    grader = USTCDeepSeekFlashJudgeCompleterConfig().build()
    grader.parse_response_type = ParsedJudgeResponseInt
    result = asyncio.run(grader.async_completion([{"role": "system", "content": "Parse"}]))
    assert len(calls) == 3
    assert result.output_messages[0].content == "invalid"
