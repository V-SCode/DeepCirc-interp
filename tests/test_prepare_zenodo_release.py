"""Offline release-packaging tests with complete miniature figure fixtures."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import fitz
from PIL import Image
import yaml

MODULE = Path(__file__).resolve().parents[1] / "scripts/prepare_zenodo_release.py"
SPEC = importlib.util.spec_from_file_location("prepare_zenodo_release", MODULE)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def pdf_bytes(text="Complete source panel"):
    with fitz.open() as document:
        page = document.new_page(width=200, height=200)
        page.insert_text((20, 30), text)
        return document.tobytes()


def png_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), "white").save(buffer, format="PNG", dpi=(600, 600))
    return buffer.getvalue()


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix="zenodo-packaging-test-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.out = self.root / "bundle"
        self.git("init", "-q")
        self.git("config", "user.name", "Release Test")
        self.git("config", "user.email", "test@example.invalid")
        self.put(".gitignore", b"figures/**/panels/\nfigures/**/final/figure_*\ndist/\n")
        self.put("README.md", b"Fixture repository\n")
        self.put("scripts/analysis.py", b"value = 1\n")
        self.put("figures/setFinal/figS10/final/.gitkeep", b"")
        self.put("figures/setFinalvF/table_legends.md", b"Historical legend, intentionally deferred.\n")
        for name in release.REQUIRED_NORMALIZED_INPUTS:
            self.put(name, b"{}\n" if name.endswith(".json") else b"part,value\nP,1\n")
        for name in release.rendering_roster():
            suffix = Path(name).suffix
            data = {".pdf": pdf_bytes(), ".png": png_bytes(),
                    ".svg": b'<svg xmlns="http://www.w3.org/2000/svg" width="1" height="1"/>',
                    ".csv": b"part,value\nP,1\n", ".yaml": b"{}\n"}[suffix]
            self.put(name, data)
        for number, ids in ((10, "abcde"), (11, "ab")):
            panels = []
            for label in ids:
                source = label if number == 10 else {"a": "d", "b": "e"}[label]
                prepared = source in "de"
                prefix = f"../publication/panels/figS{number}" if prepared else "panels/vector"
                png_prefix = prefix if prepared else "panels/raster"
                panels.append({"id": label, "file": f"{prefix}/panel_{source}.pdf",
                               "preview_file": f"{png_prefix}/panel_{source}.png"})
            self.put(f"{release.VF}/figS{number}/manifest.yaml", yaml.safe_dump({"panels": panels}).encode())
        self.git("add", ".")
        self.git("commit", "-q", "-m", "Fixture source")

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.repo), *args], stderr=subprocess.STDOUT)

    def put(self, name, data):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def prepare(self, **kwargs):
        return release.prepare_bundle(self.repo, self.out, **({"working_tree": True, "version": "v1.1.1"} if not kwargs else kwargs))

    def test_working_snapshot_provenance_and_all_checksums(self):
        self.put("scripts/analysis.py", b"value = 2\n")
        self.put(".env", b"FAKE_TEST_SECRET=not-a-real-credential\n")
        self.put("scripts/zenodo_token", b"fake-test-only\n")
        self.put("figures/setFinalvF/FigS11vF.ai", b"old Illustrator master")
        self.prepare()
        self.assertEqual(len(list(self.out.iterdir())), 5)
        manifest = json.loads((self.out / "manifest.json").read_text())
        self.assertEqual(manifest["source_mode"], "working-tree-review")
        self.assertIsNone(manifest["source_tag"])
        self.assertIsNone(manifest["source_commit"])
        self.assertFalse(manifest["ready_for_upload"])
        self.assertEqual(manifest["base_commit"], self.git("rev-parse", "HEAD").decode().strip())
        self.assertEqual(manifest["research_source_commit"], release.RESEARCH_COMMIT)
        for artifact in manifest["artifacts"]:
            data = (self.out / artifact["filename"]).read_bytes()
            self.assertEqual(release.fingerprint(data), {key: artifact[key] for key in ("size_bytes", "sha256", "md5")})
        with zipfile.ZipFile(self.out / "DeepCirc-interp-1.1.1.zip") as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(archive.read("DeepCirc-interp-1.1.1/scripts/analysis.py"), b"value = 2\n")
            self.assertTrue(any(name.endswith("table_legends.md") for name in archive.namelist()))
            self.assertFalse(any(name.endswith((".ai", ".env", "zenodo_token")) for name in archive.namelist()))
            for item in manifest["source_contents"]:
                self.assertEqual(hashlib.sha256(archive.read("DeepCirc-interp-1.1.1/" + item["path"])).hexdigest(), item["sha256"])
        with zipfile.ZipFile(self.out / "DeepCirc-interp-1.1.1-setFinalvF-source-panels-and-tables.zip") as archive:
            self.assertIsNone(archive.testzip())
            self.assertEqual(len(manifest["rendering_contents"]), 47)
            self.assertFalse(any("legend" in name.lower() or name.endswith(".ai") for name in archive.namelist()))
            for item in manifest["rendering_contents"]:
                self.assertEqual(release.fingerprint(archive.read(item["archive_path"])),
                                 {key: item[key] for key in ("size_bytes", "sha256", "md5")})
        metadata = json.loads((self.out / "metadata.json").read_text())["metadata_patch"]
        self.assertEqual(metadata["title"], "DeepCirc-Interp: Data Files and Code")
        self.assertEqual(set(metadata), {"title", "version", "description", "notes"})
        self.assertIn("UNTAGGED", metadata["notes"])
        readme = (self.out / "README-ZENODO.md").read_text()
        self.assertIn("historical prose", readme)
        self.assertIn("23227964", readme)

    def test_tagged_source_is_exact_git_archive(self):
        self.git("tag", "v1.1.1")
        self.prepare(ref="v1.1.1")
        manifest = json.loads((self.out / "manifest.json").read_text())
        self.assertEqual(manifest["source_tag"], "v1.1.1")
        self.assertEqual(manifest["source_commit"], self.git("rev-parse", "HEAD").decode().strip())
        self.assertTrue(manifest["ready_for_upload"])
        expected = self.git("archive", "--format=zip", "--prefix=DeepCirc-interp-1.1.1/", "v1.1.1")
        self.assertEqual((self.out / "DeepCirc-interp-1.1.1.zip").read_bytes(), expected)

    def test_tagged_mode_rejects_uncommitted_science(self):
        self.git("tag", "v1.1.1")
        self.put("scripts/analysis.py", b"value = 3\n")
        with self.assertRaisesRegex(ValueError, "differs from the tag"):
            self.prepare(ref="v1.1.1")
        self.assertFalse(self.out.exists())

    def test_missing_source_panel_rejected_before_output(self):
        (self.repo / f"{release.VF}/figS10/panels/vector/panel_a.pdf").unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.prepare()
        self.assertFalse(self.out.exists())

    def test_incomplete_manifest_rejected(self):
        self.put(f"{release.VF}/figS10/manifest.yaml", yaml.safe_dump({"panels": []}).encode())
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.prepare()
        self.assertFalse(self.out.exists())

    def test_missing_preview_placeholder_rejected(self):
        self.put(f"{release.VF}/figS10/final/figure_S10.pdf", pdf_bytes("[missing preview]"))
        with self.assertRaisesRegex(ValueError, "missing-preview placeholder"):
            self.prepare()
        self.assertFalse(self.out.exists())

    def test_corrupt_image_rejected(self):
        self.put(f"{release.VF}/figS11/panels/raster/panel_e.png", b"not PNG")
        with self.assertRaisesRegex(ValueError, "Invalid artifact"):
            self.prepare()
        self.assertFalse(self.out.exists())

    def test_missing_normalized_data_rejected(self):
        (self.repo / next(iter(release.REQUIRED_NORMALIZED_INPUTS))).unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.prepare()
        self.assertFalse(self.out.exists())

    def test_existing_bundle_is_never_overwritten(self):
        self.out.mkdir()
        (self.out / "keep").write_text("reviewed")
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            self.prepare()
        self.assertEqual((self.out / "keep").read_text(), "reviewed")

    def test_concurrent_source_change_rejected(self):
        original = release.validate_panel_manifests
        def mutate(*args):
            original(*args)
            self.put("scripts/analysis.py", b"concurrent edit\n")
        with patch.object(release, "validate_panel_manifests", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "Source changed during packaging"):
                self.prepare()
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
