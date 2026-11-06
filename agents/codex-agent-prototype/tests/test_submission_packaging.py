"""Exercise real PaperBench packaging through a linked submission directory."""

import tarfile
import tempfile
import unittest
from pathlib import Path

from paperbench.solvers.direct_submission.solver import (
    _put_folder_submission_in_canonical_path,
)


class SubmissionPackagingTests(unittest.TestCase):
    def test_linked_input_matches_direct_input_and_preserves_internal_links(self):
        with tempfile.TemporaryDirectory(prefix="submission-packaging-") as folder:
            base = Path(folder)
            source = base / "original source"
            (source / "nested").mkdir(parents=True)
            (source / "nested" / "main.py").write_text("print(123)\n")
            (source / "alias.py").symlink_to("nested/main.py")
            linked = base / "staging" / "submission"
            linked.parent.mkdir()
            linked.symlink_to(source, target_is_directory=True)

            def archive_contents(input_path, run_dir):
                _put_folder_submission_in_canonical_path(str(input_path), str(run_dir))
                archives = list(run_dir.rglob("*.tar.gz"))
                self.assertEqual(len(archives), 1)
                with tarfile.open(archives[0]) as archive:
                    root = archive.getmember("submission")
                    self.assertTrue(root.isdir())
                    self.assertEqual(archive.extractfile("submission/nested/main.py").read(), b"print(123)\n")
                    alias = archive.getmember("submission/alias.py")
                    self.assertTrue(alias.issym())
                    self.assertEqual(alias.linkname, "nested/main.py")
                    return [(m.name, m.type, m.size, m.linkname) for m in archive.getmembers()]

            self.assertEqual(
                archive_contents(source, base / "direct-run"),
                archive_contents(linked, base / "linked-run"),
            )
            self.assertTrue(linked.is_symlink())
            self.assertEqual((source / "nested" / "main.py").read_text(), "print(123)\n")


if __name__ == "__main__":
    unittest.main()
