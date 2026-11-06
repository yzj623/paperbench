"""Single-paper wrapper around the unmodified PaperBench evaluation pipeline."""

from __future__ import annotations

import os
from typing import Any

import chz
from nanoeval.eval import EvalSpec
from nanoeval.evaluation import run
from nanoeval.library_config import LibraryConfig
from nanoeval.setup import nanoeval_entrypoint
from paperbench.nano.entrypoint import DefaultRunnerArgs
from paperbench.nano.eval import PaperBench
from paperbench.nano.logging import paperbench_library_config, setup_logging
from paperbench.utils import get_paperbench_data_dir


@chz.chz
class SinglePaperBench(PaperBench):
    """Select exactly one official paper without editing PaperBench split files."""

    paper_id: str
    def check_for_lfs(self) -> None:
        """Validate only the explicitly selected paper for this single-paper eval."""

        paper_path = get_paperbench_data_dir() / "papers" / self.paper_id / "paper.md"
        assert paper_path.exists(), f"Paper at {paper_path} is missing; hydrate via git lfs."
        with paper_path.open("r", encoding="utf-8") as handle:
            assert len(handle.readlines()) > 5, (
                f"Paper at {paper_path} should be pulled from git lfs"
            )


    async def get_instances(self) -> list[Any]:
        instances = await super().get_instances()
        selected = [task for task in instances if task.paper_id == self.paper_id]
        if not selected:
            raise ValueError(f"unknown or unavailable PaperBench paper id: {self.paper_id}")
        return selected

    async def get_full_summary(self, results: list[tuple[Any, Any]]) -> dict[str, Any]:
        report = await super().get_full_summary(results)
        per_paper = report.get("metrics", {}).get("mean_score_by_paper", {})
        selected = per_paper.get(self.paper_id)
        # PaperBench's stock aggregate expects every member of the named split.
        # This wrapper selects one task, so expose that task's official score as
        # the aggregate without changing any rubric or judge behavior.
        if isinstance(selected, (int, float)):
            report["metrics"]["mean_score"] = float(selected)
            report["metrics"]["std_err"] = 0.0
            report["metrics"]["n_complete_tries"] = 1
        return report


async def main(
    paperbench: SinglePaperBench,
    runner: DefaultRunnerArgs,
    library_config: LibraryConfig = paperbench_library_config,
) -> None:
    grader_api_config = os.getenv("PAPERBENCH_GRADER_API_CONFIG", "openai")
    if grader_api_config == "ustc-deepseek-flash":
        from .ustc_judge import install_ustc_structured_judge_config

        install_ustc_structured_judge_config()
    elif grader_api_config == "deepseek-official-flash":
        from .deepseek_official_judge import install_deepseek_official_structured_judge_config

        install_deepseek_official_structured_judge_config()
    setup_logging(library_config)
    await run(EvalSpec(eval=paperbench, runner=runner))


if __name__ == "__main__":
    nanoeval_entrypoint(chz.entrypoint(main))
