#!/usr/bin/env python3
"""Measure live USTC chat model latency and streaming throughput."""

import argparse
import asyncio
import datetime as dt
import json
import os
import statistics
import time
from pathlib import Path

import openai
import tiktoken


DEFAULT_MODELS = [
    "deepseek-flash", "deepseek-flash-2", "deepseek-v4-flash-ascend",
    "deepseek-v4-flash-ascend1", "glm-5.3-flash", "qwen3.8-chat",
    "qwen3.8-flash-next", "qwen3.8-flash-next-2", "qwen3.8-reasoner",
    "smart/default", "smart/reasoning",
]
PROMPT = "Write the integers from 1 through 200, separated by commas. Output only the numbers and commas."


async def measure(client, model, run, encoder, semaphore):
    async with semaphore:
        started = time.perf_counter()
        first = None
        last = None
        pieces = []
        usage = None
        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": PROMPT}],
                max_tokens=1024,
                stream=True,
                stream_options={"include_usage": True},
            )
            async for chunk in stream:
                now = time.perf_counter()
                if chunk.usage is not None:
                    usage = chunk.usage
                for choice in chunk.choices:
                    content = choice.delta.content or ""
                    if content:
                        if first is None:
                            first = now
                        last = now
                        pieces.append(content)
            ended = time.perf_counter()
            content = "".join(pieces)
            visible_tokens = len(encoder.encode(content))
            generation_time = (last - first) if first is not None and last is not None else None
            return {
                "model": model, "run": run, "status": "ok" if content else "empty",
                "ttft_s": round(first - started, 3) if first else None,
                "total_s": round(ended - started, 3),
                "generation_s": round(generation_time, 3) if generation_time is not None else None,
                "visible_tokens_o200k": visible_tokens,
                "visible_tokens_per_s": round(visible_tokens / generation_time, 2) if generation_time and generation_time > 0 else None,
                "reported_prompt_tokens": usage.prompt_tokens if usage else None,
                "reported_completion_tokens": usage.completion_tokens if usage else None,
                "reported_completion_tokens_per_s": round(usage.completion_tokens / generation_time, 2) if usage and generation_time and generation_time > 0 else None,
                "output_chars": len(content),
                "expected_end": content.strip().endswith("200"),
            }
        except Exception as exc:
            return {"model": model, "run": run, "status": "error",
                    "elapsed_s": round(time.perf_counter() - started, 3),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:300]}


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--all", action="store_true", help="Discover every model ID from the live USTC API")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.concurrency < 1:
        parser.error("repeats and concurrency must be positive")
    api_key = os.environ.get("USTC_LLM_API_KEY")
    if not api_key:
        parser.error("USTC_LLM_API_KEY is required")
    encoder = tiktoken.get_encoding("o200k_base")
    semaphore = asyncio.Semaphore(args.concurrency)
    async with openai.AsyncOpenAI(api_key=api_key, base_url="https://api.llm.ustc.edu.cn/v1", timeout=args.timeout, max_retries=0) as client:
        models = sorted(model.id for model in (await client.models.list()).data) if args.all else DEFAULT_MODELS
        tasks = [measure(client, model, run, encoder, semaphore)
                 for run in range(1, args.repeats + 1) for model in models]
        results = []
        for future in asyncio.as_completed(tasks):
            result = await future
            results.append(result)
            print(f"{result['model']} #{result['run']}: {result['status']} "
                  f"TTFT={result.get('ttft_s')}s visible={result.get('visible_tokens_per_s')} tok/s", flush=True)
    record = {"measured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "base_url": "https://api.llm.ustc.edu.cn/v1", "prompt": PROMPT,
              "max_tokens": 1024, "stream": True, "concurrency": args.concurrency,
              "repeats": args.repeats, "models": models, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(f"Results: {args.output}")
    for model in models:
        valid = [r for r in results if r["model"] == model and r["status"] == "ok"]
        if valid:
            ttft = statistics.median(r["ttft_s"] for r in valid)
            speed = statistics.median(r["visible_tokens_per_s"] for r in valid if r["visible_tokens_per_s"] is not None)
            print(f"MEDIAN {model}: n={len(valid)} TTFT={ttft:.2f}s visible={speed:.1f} tok/s")


if __name__ == "__main__":
    asyncio.run(main())
