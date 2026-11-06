"""OAuth judge contracts; all model responses are local fixtures."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from openai.types.responses import ResponseUsage
from openai.types.chat import ChatCompletionMessage
from paperbench.judge.simple import SimpleJudge, ParsedJudgeResponseInt, ParsedJudgeResponseFloat

from codex_eval.openai_judge import (
    OpenAIOAuthJudgeCompleterConfig, InvalidJudgeOutput, install_openai_structured_judge_config,
)


def response(texts, usage=None):
    return SimpleNamespace(output=[SimpleNamespace(type="message", content=[
        SimpleNamespace(type="output_text", text=text)]) for text in texts], usage=usage)


class OpenAIJudgeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = OpenAIOAuthJudgeCompleterConfig()
        self.original = SimpleJudge._init_structured_completer
        self.was_installed = getattr(SimpleJudge, "_codex_eval_openai_installed", False)
        install_openai_structured_judge_config()
        self.addCleanup(self.restore)

    def restore(self):
        SimpleJudge._init_structured_completer = self.original
        if not self.was_installed:
            del SimpleJudge._codex_eval_openai_installed

    def structured(self, model=ParsedJudgeResponseInt):
        return SimpleJudge._init_structured_completer(
            SimpleNamespace(completer_config=self.config), None, model)[1]

    def test_medium_and_strict_required_fields(self):
        for schema in [ParsedJudgeResponseInt, ParsedJudgeResponseFloat]:
            client = self.structured(schema)
            body = client._request_body([{"role": "system", "content": "parse"},
                                         {"role": "user", "content": "Score: 1"}])
            self.assertEqual(body["model"], "gpt-6-luna")
            self.assertEqual(body["reasoning"], {"effort": "medium"})
            self.assertEqual(body["instructions"], "parse")
            self.assertNotIn("tools", body)
            fmt = body["text"]["format"]
            self.assertTrue(fmt["strict"])
            self.assertEqual(set(fmt["schema"]["required"]), {"valid_score", "score", "explanation"})
            self.assertFalse(fmt["schema"]["additionalProperties"])

    async def test_original_paperbench_parse_prompt_is_preserved_verbatim(self):
        client = self.structured()
        captured = []
        async def complete(body):
            captured.append(body)
            return response(['{"valid_score":true,"score":1,"explanation":"present"}'])
        client._complete_with_retries = complete
        original = SimpleJudge._parse_model_response
        # Capture PaperBench's original conversation independently of the adapter.
        async def capture(conversation):
            captured.append(conversation)
            return client.Completion(input_conversation=conversation, output_messages=[
                ChatCompletionMessage(
                    role="assistant", content='{"valid_score":true,"score":1,"explanation":"present"}')])
        proxy = SimpleNamespace(async_completion=capture)
        await original(SimpleNamespace(int_completer=proxy), "Score: 1.")
        conversation = captured.pop()
        await original(SimpleNamespace(int_completer=client), "Score: 1.")
        body = captured.pop()
        self.assertEqual(body["instructions"], conversation[0]["content"])
        self.assertEqual(body["input"][0]["content"], conversation[1]["content"])
        self.assertEqual(len(body["input"]), 1)

    def test_explicit_shell_selector_overrides_env_file(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="judge-selector-") as folder:
            base = Path(folder)
            submission = base / "submission"
            submission.mkdir()
            (submission / "README.md").write_text("fixture")
            credentials = base / "oauth"
            credentials.mkdir()
            (credentials / "credentials.json").write_text("{}")
            # Export a fake uv function; the real shell script runs its complete
            # preflight and argv construction without launching any evaluator.
            harness = ('uv() { printf "SELECTOR=%s\\n" "$PAPERBENCH_GRADER_API_CONFIG"; '
                       'printf "ARG=%s\\n" "$@"; }; export -f uv; exec bash "$@"')
            env = dict(os.environ, PAPERBENCH_GRADER_API_CONFIG="gpt6-luna",
                       CODEX_GRADER_OPENAI_OAUTH_DIR=str(credentials))
            result = subprocess.run(
                ["bash", "-c", harness, "selector-check", str(root / "scripts/evaluate_submission.sh"),
                 "mechanistic-understanding", str(submission), str(base / "evaluation")],
                env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            staged = base / "evaluation/direct-submissions/mechanistic-understanding/submission"
            self.assertTrue(staged.is_symlink())
            self.assertEqual(staged.resolve(), submission.resolve())
            self.assertIn("SELECTOR=openai-oauth", result.stdout)
            self.assertIn("codex_eval.openai_judge:OpenAIOAuthJudgeCompleterConfig", result.stdout)

    async def test_missing_field_retries_then_succeeds(self):
        client = self.structured()
        client._complete_with_retries = AsyncMock(side_effect=[
            response(['{"score":1,"explanation":"implemented"}']),
            response(['{"valid_score":true,"score":1,"explanation":"implemented"}'])])
        result = await client.async_completion([{"role": "user", "content": "parse"}])
        self.assertEqual(client._complete_with_retries.await_count, 2)
        self.assertEqual(json.loads(result.output_messages[0].content)["score"], 1)

    async def test_missing_field_never_synthesizes_a_verdict(self):
        client = self.structured()
        client._complete_with_retries = AsyncMock(return_value=response(['{"score":1}']))
        with self.assertRaises(InvalidJudgeOutput):
            await client.async_completion([{"role": "user", "content": "parse"}])
        self.assertEqual(client._complete_with_retries.await_count, 3)

    def test_multiple_message_output_and_usage(self):
        client = self.structured()
        usage = ResponseUsage(input_tokens=20, output_tokens=10, total_tokens=30,
                              input_tokens_details={"cached_tokens": 8},
                              output_tokens_details={"reasoning_tokens": 4})
        result = client._to_completion([], response([
            'Here is the verdict.', '{"valid_score":true,"score":0,"explanation":"missing"}'], usage))
        self.assertEqual(json.loads(result.output_messages[0].content)["score"], 0)
        self.assertEqual(result.usage.prompt_tokens, 20)
        self.assertEqual(result.usage.completion_tokens_details.reasoning_tokens, 4)
        with self.assertRaises(InvalidJudgeOutput):
            client._to_completion([], response([
                '{"valid_score":true,"score":0,"explanation":"missing"}',
                '{"valid_score":true,"score":1,"explanation":"present"}']))

    def test_deepseek_hook_still_uses_its_own_model(self):
        from codex_eval.ustc_judge import USTCDeepSeekFlashJudgeCompleterConfig, install_ustc_structured_judge_config
        installed = getattr(SimpleJudge, "_codex_eval_ustc_installed", False)
        install_ustc_structured_judge_config()
        try:
            conf, client = SimpleJudge._init_structured_completer(
                SimpleNamespace(completer_config=USTCDeepSeekFlashJudgeCompleterConfig()),
                None, ParsedJudgeResponseInt)
            self.assertEqual(client.model, "deepseek-v4-flash-ascend")
            self.assertEqual(conf.response_format, {"type": "json_object"})
            alternate_conf, alternate_client = SimpleJudge._init_structured_completer(
                SimpleNamespace(completer_config=USTCDeepSeekFlashJudgeCompleterConfig(model="deepseek-flash-2")),
                None, ParsedJudgeResponseInt)
            self.assertEqual(alternate_client.model, "deepseek-flash-2")
            self.assertEqual(alternate_conf.response_format, {"type": "json_object"})
        finally:
            if not installed:
                del SimpleJudge._codex_eval_ustc_installed


if __name__ == "__main__":
    unittest.main()
