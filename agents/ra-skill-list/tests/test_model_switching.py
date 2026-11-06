"""Regression coverage for non-thinking dual-model API requests (no network)."""
import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from openai.types.chat import ChatCompletion
from research_agent.api_settings import APISettings
from research_agent.deepseekv4flash_completer import DeepSeekV4FlashTurnCompleterConfig
from research_agent.qwen37flash_completer import Qwen37FlashTurnCompleterConfig


class ModelSwitchingTest(unittest.TestCase):
    def test_role_configs_and_request_payloads(self):
        async def check(config, expected):
            fresh = APISettings.from_config(config).new_config(basicagent_tools=[])
            self.assertEqual(fresh.provider, config.provider)
            completer = fresh.build()
            completion = ChatCompletion.model_validate({
                "id": "offline-test", "object": "chat.completion", "created": 0,
                "model": config.model, "choices": [
                    {"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "OK"}}
                ],
            })
            create = AsyncMock(return_value=completion)
            completer.__dict__["_client"] = SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
            result = await completer.async_completion(
                [{"role": "user", "content": "Return OK"}])
            kwargs = create.call_args.kwargs
            self.assertEqual(kwargs["extra_body"], expected)
            self.assertEqual(kwargs["model"], config.model)
            self.assertEqual(result.output_messages[0].content, "OK")
            return fresh
        qwen = asyncio.run(check(Qwen37FlashTurnCompleterConfig(), {"enable_thinking": False}))
        deepseek = asyncio.run(check(DeepSeekV4FlashTurnCompleterConfig(),
                                    {"thinking": {"type": "disabled"}}))
        self.assertEqual(qwen.api_key_env, "DASHSCOPE_API_KEY")
        self.assertEqual(deepseek.api_key_env, "DEEPSEEK_API_KEY")
        self.assertEqual(qwen.context_window, deepseek.context_window)
        with self.assertRaises(ValueError):
            DeepSeekV4FlashTurnCompleterConfig(thinking="enabled")


if __name__ == "__main__":
    unittest.main()
