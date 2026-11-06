"""Verify diagnostics persist without changing selected file contents."""

import json
import logging
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import tiktoken

from paperbench.grade import grade_submission
from paperbench.judge.simple import SimpleJudge


class JudgeLoggingTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_submission_prefix_is_retried_once(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "submission").mkdir()
            (root / "submission/good.py").write_text("print(123)\n")
            logs = root / "logs"
            logs.mkdir()
            raw = "good.py\nsubmission/good.py\nmissing.py"
            judge = SimpleNamespace(
                get_logger=lambda task: logging.getLogger("submission-prefix-test"),
                tree_structures={"Code Development": "submission/good.py"},
                paper_md="Paper text", joined_addendum="Addendum text",
                completer=SimpleNamespace(async_completion=AsyncMock(return_value=SimpleNamespace(
                    output_messages=[SimpleNamespace(content=raw)]))),
                avail_context_lens={"Code Development": 10000}, submission_dir=root,
                computer=None, token_encoder=tiktoken.get_encoding("o200k_base"), log_path=logs,
            )
            task = SimpleNamespace(id="criterion-1", requirements="Implement code", task_category="Code Development")
            result = await SimpleJudge._prepare_relevant_files(judge, task)
            self.assertIn("submission/good.py", result)
            self.assertEqual(result.count("print(123)"), 2)
            trace = json.loads((logs / "criterion-1_file_selection.json").read_text())
            self.assertEqual(trace["included_files"], 2)
            self.assertEqual(trace["attempted_files"][0]["fallback_path"], str(root / "submission/good.py"))
            self.assertNotIn("fallback_path", trace["attempted_files"][1])
            self.assertEqual(trace["attempted_files"][2]["status"], "error")
            self.assertEqual(judge.completer.async_completion.await_count, 1)

    async def test_selection_log_records_failed_paths_and_preserves_content(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "submission").mkdir()
            (root / "submission/good.py").write_text("print(123)\n")
            logs = root / "logs"
            logs.mkdir()
            raw = "submission/good.py\nsubmission/missing.py\n- submission/good.py"
            judge = SimpleNamespace(
                get_logger=lambda task: logging.getLogger("selection-log-test"),
                tree_structures={"Code Development": "submission/good.py"},
                paper_md="Paper text", joined_addendum="Addendum text",
                completer=SimpleNamespace(async_completion=AsyncMock(return_value=SimpleNamespace(
                    output_messages=[SimpleNamespace(content=raw)]))),
                avail_context_lens={"Code Development": 10000}, submission_dir=root,
                computer=None, token_encoder=tiktoken.get_encoding("o200k_base"), log_path=None,
            )
            task = SimpleNamespace(id="criterion-1", requirements="Implement code", task_category="Code Development")
            without_logs = await SimpleJudge._prepare_relevant_files(judge, task)
            judge.log_path = logs
            with_logs = await SimpleJudge._prepare_relevant_files(judge, task)
            self.assertEqual(with_logs, without_logs)
            trace = json.loads((logs / "criterion-1_file_selection.json").read_text())
            self.assertEqual(trace["raw_selection"], raw)
            self.assertEqual(trace["included_files"], 1)
            self.assertFalse(trace["empty"])
            self.assertEqual([f["status"] for f in trace["attempted_files"]], ["included", "error", "error"])
            self.assertEqual(trace["attempted_files"][1]["error_type"], "FileNotFoundError")
            request = json.loads((logs / "criterion-1_file_selection_request.json").read_text())
            self.assertEqual(request, judge.completer.async_completion.call_args.kwargs["conversation"])
            judge.completer.async_completion.return_value.output_messages[0].content = "submission/missing.py"
            self.assertEqual(await SimpleJudge._prepare_relevant_files(judge, task), "")
            trace = json.loads((logs / "criterion-1_file_selection.json").read_text())
            self.assertTrue(trace["empty"])
            self.assertEqual(trace["included_files"], 0)

    async def test_grade_entrypoint_passes_persistent_log_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / "submission.tar.gz"
            with tarfile.open(archive, "w:gz"):
                pass
            with patch.dict(os.environ, {"PAPERBENCH_JUDGE_LOG_DIR": str(root / "logs")}), patch(
                "paperbench.grade.run_judge", new_callable=AsyncMock,
                side_effect=RuntimeError("diagnostic test stops before grading"),
            ) as judge:
                await grade_submission(
                    submission_path=str(archive), grader_upload_path=str(root / "grade.json"),
                    paper_id="stochastic-interpolants", judge_type="simple", completer_config=None,
                    run_group_id="group", runs_dir=str(root), run_id="run-1", code_only=True,
                )
            self.assertEqual(judge.call_args.kwargs["out_dir"], root / "logs/run-1")


if __name__ == "__main__":
    unittest.main()
