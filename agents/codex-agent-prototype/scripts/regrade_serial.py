#!/usr/bin/env python3
"""Serially regrade frozen Codex submissions without overwriting existing results."""

import argparse
import datetime
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper_id")
    parser.add_argument("experiment_dirs", nargs="+", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    experiments = [p.resolve(strict=True) for p in args.experiment_dirs]
    for p in experiments:
        submission = p / "output/frozen/submission" if (p / "launch.json").is_file() else p / "frozen/submission"
        if not submission.is_dir():
            parser.error(f"Frozen submission missing: {p}")
        metadata = json.loads((p / ("launch.json" if (p / "launch.json").is_file() else "metadata.json")).read_text())
        if metadata["paper_id"] != args.paper_id:
            parser.error(f"Paper ID mismatch: {p}")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch = Path(tempfile.mkdtemp(prefix=f"regrade-{args.paper_id}-{stamp}-", dir=ROOT / "runs"))
    records = []

    def save(state):
        (batch / "status.json").write_text(json.dumps(
            {"state": state, "paper_id": args.paper_id,
             "repeats": args.repeats, "results": records}, ensure_ascii=False, indent=2
        ) + "\n")

    print(f"Batch directory: {batch}", flush=True)
    save("running")
    for index, experiment in enumerate(experiments, 1):
        for repeat in range(1, args.repeats + 1):
            name = f"submission-{index}-repeat-{repeat}"
            output = batch / name
            record = {"experiment": str(experiment), "repeat": repeat,
                      "output": str(output), "state": "running"}
            records.append(record)
            save("running")
            print(f"Starting {name}: {experiment.name}", flush=True)
            with (batch / f"{name}.log").open("x") as log:
                result = subprocess.run([
                    "bash", str(ROOT / "scripts/evaluate_submission.sh"), args.paper_id,
                    str(experiment / ("output/frozen/submission" if (experiment / "launch.json").is_file() else "frozen/submission")), str(output),
                ], stdout=log, stderr=subprocess.STDOUT, check=False)
            grades = list(output.glob("runs/*/*/grade.json"))
            record.update(exit_code=result.returncode,
                          state="complete" if result.returncode == 0 and len(grades) == 1 else "failed")
            if len(grades) == 1:
                grade = json.loads(grades[0].read_text())
                record.update(score=grade.get("score"), grade_file=str(grades[0]))
            save("running")
            print(f"Finished {name}: {record['state']}, score={record.get('score')}", flush=True)
    failed = any(r["state"] != "complete" for r in records)
    save("failed" if failed else "complete")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
