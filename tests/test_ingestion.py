import json
import tempfile
import unittest
from pathlib import Path

from src.ingestion.parsers import ParserRegistry
from src.ingestion.preprocess_raw import preprocess_all


class IngestionTests(unittest.TestCase):
    def test_registry_supports_declared_mvp_formats(self):
        extensions = ParserRegistry().supported_extensions
        for extension in {".txt", ".md", ".json", ".csv", ".pdf", ".docx", ".xlsx", ".html"}:
            self.assertIn(extension, extensions)

    def test_incremental_ingestion_reuses_unchanged_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            processed = root / "processed"
            raw.mkdir()
            (raw / "notes.md").write_text(
                "# Retrieval\n\nGraphMind combines semantic and graph retrieval.", encoding="utf-8"
            )

            first = preprocess_all(raw, processed)
            second = preprocess_all(raw, processed)

            self.assertEqual(first["parsed_files"], 1)
            self.assertEqual(second["unchanged_files"], 1)
            rows = [
                json.loads(line)
                for line in (processed / "document_chunks.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["source_uri"], "notes.md")
            self.assertIn("content", rows[0])
            self.assertEqual(rows[0]["schema_version"], "1.0")

            (raw / "notes.md").write_text("Updated retrieval content.", encoding="utf-8")
            changed = preprocess_all(raw, processed)
            self.assertEqual(changed["parsed_files"], 1)
            self.assertIn(
                "Updated retrieval content",
                (processed / "document_chunks.jsonl").read_text(encoding="utf-8"),
            )

            (raw / "notes.md").unlink()
            removed = preprocess_all(raw, processed)
            self.assertEqual(removed["removed_files"], 1)
            self.assertEqual((processed / "document_chunks.jsonl").read_text(encoding="utf-8"), "")

    def test_html_parser_excludes_script_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "page.html"
            path.write_text(
                "<html><body><h1>Visible</h1><script>secretNoise()</script><p>Knowledge</p></body></html>",
                encoding="utf-8",
            )
            parsed = ParserRegistry().parse(path)
            self.assertIn("Visible", parsed[0].content)
            self.assertNotIn("secretNoise", parsed[0].content)


if __name__ == "__main__":
    unittest.main()
