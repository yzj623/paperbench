#!/usr/bin/env python3
"""Run the three fixed-plan Codex experiments and their grading sequentially."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BASELINES = (
    ("mechanistic-understanding", "2026-09-13T04-33-18Z"),
    ("stochastic-interpolants", "2026-09-17T10-13-57Z"),
    ("all-in-one", "2026-09-15T02-55-50Z"),
)


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-6-luna",
                        help="Uses the model's default reasoning level (medium for gpt-6-luna).")
    parser.add_argument("--proxy", default=os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy"),
                        help="Clash HTTP proxy; defaults to HTTPS_PROXY/https_proxy.")
    parser.add_argument("--image", default="codex-paperbench:latest")
    parser.add_argument("--batches-dir", type=Path, default=ROOT / "runs/serial-batches")
    parser.add_argument("--dry-run", action="store_true", help="Validate plans and show order without launching.")
    args = parser.parse_args()
    plans = []
    for paper, stamp in BASELINES:
        plan = ROOT.parent / "PaperBench-Experiments" / paper / stamp / "agent-run/research_agent/plan.json"
        if not plan.is_file():
            parser.error(f"Baseline plan missing: {plan}")
        json.loads(plan.read_text())
        plans.append((paper, plan))
    if args.dry_run:
        for paper, plan in plans:
            print(f"{paper}: {plan}")
        print(f"Model: {args.model}; auto-grade: yes; proxy configured: {bool(args.proxy)}")
        return 0
    args.batches_dir.mkdir(parents=True, exist_ok=True)
    batch = Path(tempfile.mkdtemp(prefix="codex-three-papers-", dir=args.batches_dir.resolve()))
    summary = {"state": "running", "created_at": now(), "model": args.model,
               "reasoning_effort": "model_default", "auto_grade": True, "results": []}

    def save():
        summary["updated_at"] = now()
        temporary = batch / "summary.tmp"
        temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(batch / "summary.json")

    print(f"Batch directory: {batch}", flush=True)
    save()
    for paper, source in plans:
        # Preserve the exact input plan and isolate each launcher's unique run directory.
        plan = batch / f"{paper}-plan.json"
        plan.write_bytes(source.read_bytes())
        runs = batch / paper
        command = [sys.executable, str(ROOT / "run_paperbench.py"), "--paper-id", paper,
                   "--plan", str(plan), "--model", args.model, "--runtime", "docker",
                   "--image", args.image, "--runs-dir", str(runs)]
        if args.proxy:
            command += ["--proxy", args.proxy]
        record = {"paper_id": paper, "state": "running", "started_at": now(),
                  "plan": str(plan), "log": str(batch / f"{paper}.log")}
        summary["results"].append(record)
        save()
        print(f"Starting {paper}", flush=True)
        process = None
        interrupted = False
        try:
            with Path(record["log"]).open("xb") as stream:
                process = subprocess.Popen(command, cwd=ROOT, stdout=stream,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                record["launcher_pid"] = process.pid
                save()
                code = process.wait()
            record.update(exit_code=code, state="complete" if code == 0 else "failed")
        except KeyboardInterrupt:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                process.wait()
            record["state"] = "interrupted"
            interrupted = True
        except OSError as exc:
            record.update(state="failed", error=str(exc))
        run_dirs = list(runs.glob(f"{paper}-*")) if runs.exists() else []
        if len(run_dirs) == 1:
            run = run_dirs[0]
            record["run_dir"] = str(run)
            status = run / "host_status.json"
            if status.is_file():
                try:
                    record["host_status"] = json.loads(status.read_text())
                except (OSError, ValueError) as exc:
                    record["status_read_error"] = str(exc)
            grades = list((run / "evaluation/runs").glob("*/*/grade.json"))
            if len(grades) == 1:
                try:
                    grade = json.loads(grades[0].read_text())
                    score = grade.get("score")
                    if score is None:
                        score = grade.get("paperbench_result", {}).get("judge_output", {}).get("score")
                    record.update(score=score, grade_file=str(grades[0]))
                except (OSError, ValueError, AttributeError) as exc:
                    record["grade_read_error"] = str(exc)
        record["finished_at"] = now()
        if interrupted:
            summary["state"] = "interrupted"
            save()
            return 130
        save()
        print(f"Finished {paper}: {record['state']}; score={record.get('score')}", flush=True)
    failed = any(record["state"] != "complete" for record in summary["results"])
    summary["state"] = "complete_with_failures" if failed else "complete"
    save()
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
