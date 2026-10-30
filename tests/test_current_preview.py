"""A current preview must never silently substitute missing-panel artwork."""
from pathlib import Path
import tempfile
import unittest

from figures.setFinalvF.scripts.build_previews import require_manifest_panels


class CompletePreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "manifest.yaml"
        self.manifest.write_text("""figure_id: test
canvas: {width_mm: 20, height_mm: 20}
panels:
  - id: a
    file: panel.pdf
    preview_file: panel.png
    x_mm: 1
    y_mm: 1
    w_mm: 10
    h_mm: 10
""")
        (self.root / "panel.pdf").write_bytes(b"%PDF-test")
        (self.root / "panel.png").write_bytes(b"\x89PNG\r\n\x1a\ntest")

    def test_complete_panel_passes(self):
        require_manifest_panels(self.manifest)

    def test_missing_raster_is_rejected(self):
        (self.root / "panel.png").unlink()
        with self.assertRaises(FileNotFoundError):
            require_manifest_panels(self.manifest)

    def test_missing_vector_is_rejected(self):
        (self.root / "panel.pdf").unlink()
        with self.assertRaises(FileNotFoundError):
            require_manifest_panels(self.manifest)

    def test_wrong_vector_format_is_rejected(self):
        (self.root / "panel.pdf").write_bytes(b"not a PDF")
        with self.assertRaises(ValueError):
            require_manifest_panels(self.manifest)


if __name__ == "__main__":
    unittest.main()
