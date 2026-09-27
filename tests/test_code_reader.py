import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.code_reader import CodeReader, hash_text  # noqa: E402


class TestCodeReader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "main.py").write_text("print('main')\n", encoding="utf-8")
        (root / "config.py").write_text("LR = 0.1\n", encoding="utf-8")
        (root / "utils.py").write_text("x = 1\n", encoding="utf-8")
        (root / "sub").mkdir()
        (root / "sub" / "model.py").write_text("y = 2\n", encoding="utf-8")
        (root / "__pycache__").mkdir()
        (root / "__pycache__" / "junk.py").write_text("junk\n", encoding="utf-8")
        (root / ".nems").mkdir()
        (root / ".nems" / "gen.py").write_text("gen\n", encoding="utf-8")
        self.root = root

    def tearDown(self):
        self.tmp.cleanup()

    def test_scan_priority_and_excludes(self):
        rels = [f.rel_path for f in CodeReader(self.root).scan()]
        self.assertEqual(rels[0], "main.py")
        self.assertEqual(rels[1], "config.py")
        self.assertIn("sub/model.py", rels)
        self.assertNotIn("__pycache__/junk.py", rels)
        self.assertNotIn(".nems/gen.py", rels)

    def test_hashes_stable(self):
        reader = CodeReader(self.root)
        hashes = reader.hashes()
        self.assertEqual(hashes["main.py"], hash_text("print('main')\n"))
        self.assertEqual(hashes, reader.hashes())

    def test_read_matching(self):
        reader = CodeReader(self.root)
        data = reader.read_matching(["config.py", "missing.py"])
        self.assertIn("config.py", data)
        self.assertNotIn("missing.py", data)

    def test_max_files(self):
        files = CodeReader(self.root).scan(max_files=2)
        self.assertEqual(len(files), 2)


if __name__ == "__main__":
    unittest.main()
