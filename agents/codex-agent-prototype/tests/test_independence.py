import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class IndependenceTest(unittest.TestCase):
    def test_protocol_snapshot_and_codex_prompt_are_recorded(self):
        manifest = json.loads((ROOT / "provenance.json").read_text())
        for name, digest in manifest["protocol_sha256"].items():
            if name == "prompts.py":
                continue  # Original Pi prompt digest is retained for provenance.
            self.assertEqual(hashlib.sha256((ROOT / "protocol" / name).read_bytes()).hexdigest(),
                             digest, name)
        self.assertEqual(hashlib.sha256((ROOT / "protocol/prompts.py").read_bytes()).hexdigest(),
                         manifest["codex_prompt_sha256"])

    def test_complete_skills_are_owned_locally(self):
        manifest = json.loads((ROOT / "provenance.json").read_text())
        for paper, count in manifest["complete_skill_counts"].items():
            skills = ROOT / "skills" / paper / "skill"
            self.assertEqual(len(list(skills.rglob("SKILL.md"))), count)
            self.assertTrue(any(skills.rglob("scripts/*.py")))
            for path in skills.rglob("*"):
                self.assertTrue(path.resolve().is_relative_to(ROOT), path)

    def test_no_runtime_dependency_on_other_agent_projects(self):
        files = [*ROOT.glob("*.py"), *ROOT.glob("scripts/*"),
                 *ROOT.glob("protocol/*.py"), *ROOT.glob("codex_eval/*.py")]
        for path in files:
            if not path.is_file() or path.suffix not in {".py", ".sh"}:
                continue
            text = path.read_text()
            for forbidden in ("pi-checker-prototype/", "ResearchAgent/", "ResearchAgent-Skill/",
                              "external/AREX-Skill-full"):
                self.assertNotIn(forbidden, text, path)

    def test_grading_remains_codedev(self):
        script = (ROOT / "scripts/evaluate_submission.sh").read_text()
        self.assertIn("paperbench.judge.code_only=true", script)
        self.assertIn("paperbench.reproduction.skip_reproduction=true", script)
        self.assertIn("codex_eval.ustc_judge", script)
        self.assertNotIn("pi_eval", script)
        self.assertNotIn("ResearchAgent", script)

    def test_vendored_paperbench_split_resources(self):
        splits = ROOT / "vendor/paperbench-official/project/paperbench/experiments/splits"
        for name in ("all", "debug", "dev", "human", "lite", "testing"):
            path = splits / f"{name}.txt"
            self.assertTrue(path.is_file(), path)
            self.assertTrue(path.resolve().is_relative_to(ROOT), path)
            self.assertTrue(path.read_text().strip(), path)
        papers = (splits / "all.txt").read_text().splitlines()
        self.assertEqual(len(papers), 20)
        for paper in ("mechanistic-understanding", "all-in-one", "stochastic-interpolants"):
            self.assertIn(paper, papers)


if __name__ == "__main__":
    unittest.main()
