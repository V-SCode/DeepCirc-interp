"""Offline integration/regression checks using only the Python standard library."""
import contextlib
import hashlib
import importlib.util
import io
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "download_data.py"
SPEC = importlib.util.spec_from_file_location("download_data", MODULE)
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="download-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "data"
        self.payloads = {}
        self.downloads = []
        self.output = io.StringIO()
        self.errors = io.StringIO()

    def file(self, key, contents, algorithm="md5"):
        url = "https://example.invalid/files/" + str(len(self.payloads))
        self.payloads[url] = contents
        return {"key": key, "links": {"self": url},
                "checksum": algorithm + ":" + hashlib.new(algorithm, contents).hexdigest()}

    def archive(self, entries, key="DeepCirc-interp-v1.1.0.zip"):
        data = io.BytesIO()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(data, "w") as archive:
                for name, content in entries:
                    archive.writestr(name, content)
        return self.file(key, data.getvalue())

    def retrieve(self, url, filename):
        self.downloads.append(url)
        Path(filename).write_bytes(self.payloads[url])
        return str(filename), None

    def run_main(self, files, *args):
        with patch.object(download, "_fetch_manifest", return_value={"files": files}) as fetch, \
             patch.object(download.urllib.request, "urlretrieve", side_effect=self.retrieve), \
             contextlib.redirect_stdout(self.output), contextlib.redirect_stderr(self.errors):
            status = download.main(["--root", str(self.root), *args])
        self.last_fetch = fetch
        return status

    def test_pinned_default_doi_does_not_change(self):
        entry = self.file("figures__data__topology_g3__x.json", b"new")
        self.assertEqual(self.run_main([entry]), 0)
        self.last_fetch.assert_called_once_with("20576709")
        self.assertEqual(download.DEFAULT_DOI, "10.5281/zenodo.20576709")

    def test_flattened_stage_layout_and_md5(self):
        entry = self.file("figures__data__topology_g3__panel_c_shapley__x.json", b"new")
        self.assertEqual(self.run_main([entry]), 0)
        self.assertEqual((self.root / "topology_g3/panel_c_shapley/x.json").read_bytes(), b"new")
        self.assertFalse((self.root / "data").exists())
        self.assertFalse((self.root / "data__topology_g3__panel_c_shapley__x.json").exists())

    def test_sha256_download(self):
        entry = self.file("figures__data__interp_processed__x.json", b"abc", "sha256")
        self.assertEqual(self.run_main([entry]), 0)
        self.assertEqual((self.root / "interp_processed/x.json").read_bytes(), b"abc")

    def test_checksum_failure_preserves_existing_data(self):
        first = self.file("figures__data__topology_g3__new.json", b"new")
        entry = self.file("figures__data__topology_g3__old.json", b"expected")
        self.payloads[entry["links"]["self"]] = b"corrupt"
        self.root.mkdir()
        (self.root / "retained.json").write_bytes(b"original")
        self.assertEqual(self.run_main([first, entry], "--overwrite"), 1)
        self.assertIn("MD5 mismatch", self.errors.getvalue())
        self.assertFalse((self.root / "topology_g3").exists())
        self.assertEqual((self.root / "retained.json").read_bytes(), b"original")

    def test_unsupported_or_missing_checksum_fails_before_download(self):
        for checksum in [None, "sha1:abcd", "md5:nothex"]:
            with self.subTest(checksum=checksum):
                entry = self.file("figures__data__topology_g3__x.json", b"abc")
                entry["checksum"] = checksum
                self.assertEqual(self.run_main([entry]), 1)
        self.assertEqual(self.downloads, [])

    def test_zip_fallback_outside_cwd_installs_only_data(self):
        archive = self.archive([
            ("repo/README.md", b"not installed"),
            ("repo/scripts/script.py", b"not installed"),
            ("repo/data/topology_g3/x.json", b"new"),
            ("repo/data/interp_processed/y.json", b"corrected"),
        ])
        self.assertEqual(self.run_main([archive]), 0)
        self.assertEqual((self.root / "topology_g3/x.json").read_bytes(), b"new")
        self.assertEqual((self.root / "interp_processed/y.json").read_bytes(), b"corrected")
        self.assertFalse((self.root / "README.md").exists())
        self.assertFalse((self.root / "scripts").exists())

    def test_zip_without_wrapper_supported(self):
        archive = self.archive([("data/topology_g3/x.json", b"new")])
        self.assertEqual(self.run_main([archive]), 0)
        self.assertEqual((self.root / "topology_g3/x.json").read_bytes(), b"new")

    def test_multiple_archives_require_explicit_selection(self):
        one = self.archive([("repo/data/topology_g3/x.json", b"old")], "old.zip")
        two = self.archive([("repo/data/topology_g3/x.json", b"new")], "new.zip")
        self.assertEqual(self.run_main([one, two]), 1)
        self.assertEqual(self.downloads, [])
        self.assertIn("--archive", self.errors.getvalue())
        self.assertEqual(self.run_main([one, two], "--archive", "new.zip"), 0)
        self.assertEqual((self.root / "topology_g3/x.json").read_bytes(), b"new")

    def test_archive_key_cannot_write_outside_temporary_directory(self):
        archive = self.archive([("repo/data/topology_g3/x.json", b"new")], "../../outside.zip")
        self.assertEqual(self.run_main([archive]), 0)
        self.assertEqual((self.root / "topology_g3/x.json").read_bytes(), b"new")
        self.assertFalse((Path(self.temporary.name) / "outside.zip").exists())

    def test_zip_rejects_traversal_absolute_and_windows_paths(self):
        for name in ["repo/data/../../outside", "/absolute", "repo/../x", "repo\\data\\x", "C:/x"]:
            with self.subTest(name=name):
                archive = self.archive([("repo/data/topology_g3/x.json", b"new"), (name, b"evil")])
                self.assertEqual(self.run_main([archive]), 1)
                self.assertFalse(self.root.exists())

    def test_zip_symlink_and_special_member_rejected(self):
        for mode in [stat.S_IFLNK, stat.S_IFIFO]:
            with self.subTest(mode=mode):
                info = zipfile.ZipInfo("repo/data/link")
                info.create_system = 3
                info.external_attr = (mode | 0o777) << 16
                archive = self.archive([("repo/data/topology_g3/x.json", b"new"), (info, b"/tmp")])
                self.assertEqual(self.run_main([archive]), 1)
                self.assertFalse(self.root.exists())

    def test_zip_duplicate_data_paths_rejected(self):
        archive = self.archive([("repo/data/topology_g3/x.json", b"first"), ("repo/data/topology_g3/x.json", b"second")])
        self.assertEqual(self.run_main([archive]), 1)
        self.assertIn("Duplicate data path", self.errors.getvalue())
        self.assertFalse(self.root.exists())

    def test_multiple_data_roots_rejected(self):
        archive = self.archive([("one/data/topology_g3/x.json", b"one"), ("two/data/topology_g3/x.json", b"two")])
        self.assertEqual(self.run_main([archive]), 1)
        self.assertIn("exactly one", self.errors.getvalue())
        self.assertFalse(self.root.exists())

    def test_stale_file_blocks_entire_install_until_overwrite(self):
        existing = self.root / "topology_g3/x.json"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"old")
        for layout in ["tier", "zip"]:
            with self.subTest(layout=layout):
                if layout == "tier":
                    files = [self.file("figures__data__topology_g3__new.json", b"added"),
                             self.file("figures__data__topology_g3__x.json", b"corrected")]
                else:
                    files = [self.archive([("repo/data/topology_g3/new.json", b"added"),
                                           ("repo/data/topology_g3/x.json", b"corrected")])]
                self.assertEqual(self.run_main(files), 1)
                self.assertEqual(existing.read_bytes(), b"old")
                self.assertFalse((existing.parent / "new.json").exists())
                self.assertIn("--overwrite", self.errors.getvalue())
        self.assertEqual(self.run_main(files, "--overwrite"), 0)
        self.assertEqual(existing.read_bytes(), b"corrected")
        self.assertEqual((existing.parent / "new.json").read_bytes(), b"added")

    def test_identical_existing_file_is_not_replaced(self):
        existing = self.root / "topology_g3/x.json"
        existing.parent.mkdir(parents=True)
        existing.write_bytes(b"same")
        previous_stat = existing.stat()
        entry = self.file("figures__data__topology_g3__x.json", b"same")
        self.assertEqual(self.run_main([entry]), 0)
        self.assertEqual(existing.stat().st_ino, previous_stat.st_ino)
        self.assertEqual(existing.stat().st_mtime_ns, previous_stat.st_mtime_ns)

    def test_full_tier_rejects_figures_only_tagged_files(self):
        for prefix in ["figures", "full"]:
            entry = self.file(prefix + "__data__topology_g3__topology_graphs.json", b"{}")
            self.assertEqual(self.run_main([entry], "--tier", "full"), 1)
        self.assertEqual(self.downloads, [])
        self.assertIn("no full-tier back-end artifacts", self.errors.getvalue())

    def test_full_tier_rejects_figures_only_source_archive(self):
        archive = self.archive([("repo/data/topology_g3/x.json", b"new")])
        self.assertEqual(self.run_main([archive], "--tier", "full"), 1)
        self.assertFalse(self.root.exists())
        self.assertIn("no full-tier back-end artifacts", self.errors.getvalue())

    def test_full_tier_accepts_actual_artifacts_and_deduplicates_data(self):
        one = self.file("figures__data__topology_g3__x.json", b"same")
        two = self.file("full__data__topology_g3__x.json", b"same")
        three = self.file("full__data__topology_g3__mlp_checkpoints__model.pt", b"model")
        self.assertEqual(self.run_main([one, two, three], "--tier", "full"), 0)
        self.assertEqual(len(self.downloads), 2)
        self.assertEqual((self.root / "topology_g3/mlp_checkpoints/model.pt").read_bytes(), b"model")

    def test_full_tier_archive_with_artifacts(self):
        archive = self.archive([("repo/data/topology_g3/x.json", b"new"),
                                ("repo/data/exemplars/0x17_design/outputs.h5", b"model")])
        self.assertEqual(self.run_main([archive], "--tier", "full"), 0)
        self.assertEqual((self.root / "exemplars/0x17_design/outputs.h5").read_bytes(), b"model")

    def test_conflicting_tier_entries_rejected(self):
        one = self.file("figures__data__topology_g3__x.json", b"old")
        two = self.file("full__data__topology_g3__x.json", b"new")
        self.assertEqual(self.run_main([one, two], "--tier", "full"), 1)
        self.assertIn("Conflicting deposit files", self.errors.getvalue())
        self.assertEqual(self.downloads, [])

    def test_flattened_paths_cannot_escape_data_root(self):
        for key in ["figures__data__..__escaped", "figures__/etc/passwd", "figures__scripts__x.py"]:
            entry = self.file(key, b"evil")
            self.assertEqual(self.run_main([entry]), 1)
        self.assertEqual(self.downloads, [])
        self.assertFalse(self.root.exists())

    def test_existing_symlink_destination_rejected(self):
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        self.root.mkdir()
        (self.root / "topology_g3").symlink_to(outside, target_is_directory=True)
        entry = self.file("figures__data__topology_g3__x.json", b"evil")
        self.assertEqual(self.run_main([entry], "--overwrite"), 1)
        self.assertEqual(list(outside.iterdir()), [])

    def test_zip_file_directory_collision_fails_before_install(self):
        archive = self.archive([("repo/data/topology_g3/first.json", b"first"),
                                ("repo/data/topology_g3/node", b"file"),
                                ("repo/data/topology_g3/node/child", b"child")])
        self.assertEqual(self.run_main([archive]), 1)
        self.assertFalse(self.root.exists())

    def test_existing_parent_file_blocks_entire_install(self):
        self.root.mkdir()
        (self.root / "topology_g3").write_bytes(b"preserved")
        first = self.file("figures__data__interp_processed__x.json", b"new")
        second = self.file("figures__data__topology_g3__x.json", b"new")
        self.assertEqual(self.run_main([first, second], "--overwrite"), 1)
        self.assertFalse((self.root / "interp_processed").exists())
        self.assertEqual((self.root / "topology_g3").read_bytes(), b"preserved")

    def test_dry_run_never_downloads_or_installs(self):
        entry = self.file("figures__data__topology_g3__x.json", b"new")
        self.assertEqual(self.run_main([entry], "--dry-run"), 0)
        archive = self.archive([("repo/data/topology_g3/x.json", b"new")])
        self.assertEqual(self.run_main([archive], "--dry-run", "--tier", "full"), 0)
        self.assertIn("tier availability are not yet verified", self.output.getvalue())
        self.assertEqual(self.downloads, [])
        self.assertFalse(self.root.exists())


if __name__ == "__main__":
    unittest.main()
