#!/usr/bin/env python3
"""Regrade each submission in a serial-experiment summary using regrade_serial.py."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    source = json.loads(args.summary.read_text())
    for entry in source["results"]:
        run = Path(entry["run_dir"])
        if not (run / "output/frozen/submission").is_dir():
            parser.error(f"Frozen submission missing: {run}")
        if json.loads((run / "launch.json").read_text())["paper_id"] != entry["paper_id"]:
            parser.error(f"Paper ID mismatch: {run}")
    batch = Path(tempfile.mkdtemp(prefix="regrade-three-papers-", dir=ROOT / "runs"))
    judge = ROOT / "vendor/paperbench-official/project/paperbench/paperbench/judge/simple.py"
    report = {"state": "running", "source_summary": str(args.summary.resolve()),
              "created_at": datetime.now(timezone.utc).isoformat(), "repeats": args.repeats,
              "judge_sha256": hashlib.sha256(judge.read_bytes()).hexdigest(), "results": []}

    def save():
        temporary = batch / "summary.tmp"
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(batch / "summary.json")

    print(f"Batch directory: {batch}", flush=True)
    save()
    for entry in source["results"]:
        paper = entry["paper_id"]
        record = {"paper_id": paper, "run_dir": entry["run_dir"], "state": "running",
                  "log": str(batch / f"{paper}.log")}
        report["results"].append(record)
        save()
        print(f"Starting {paper}: {args.repeats} regrades", flush=True)
        try:
            with Path(record["log"]).open("x") as log:
                with subprocess.Popen(
                    [sys.executable, str(ROOT / "scripts/regrade_serial.py"), paper,
                     entry["run_dir"], "--repeats", str(args.repeats)],
                    cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                ) as process:
                    for line in process.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                        if line.startswith("Batch directory: "):
                            record["regrade_dir"] = line.split(": ", 1)[1].strip()
                            save()
                    code = process.wait()
            record.update(exit_code=code, state="complete" if code == 0 else "failed")
            if record.get("regrade_dir"):
                status = Path(record["regrade_dir"]) / "status.json"
                record["regrades"] = json.loads(status.read_text())
        except (OSError, ValueError) as exc:
            record.update(state="failed", error=str(exc))
        save()
    failed = any(r["state"] != "complete" for r in report["results"])
    report["state"] = "complete_with_failures" if failed else "complete"
    save()
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
