"""Prepare a local source-panel correction bundle; never upload or publish.

Review an uncommitted candidate:
    python scripts/prepare_zenodo_release.py --working-tree --version v1.1.1 \
        --out dist/zenodo-v1.1.1-review

After review, commit and tag separately, then prepare the exact release:
    python scripts/prepare_zenodo_release.py --ref v1.1.1 --out dist/zenodo-v1.1.1

Working-tree mode explicitly records an untagged snapshot. All seven source
panels, four prepared placement pages, previews, tables and manifest assets
are required. No old Illustrator master or standalone legend is packaged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import zipfile
import xml.etree.ElementTree as ET

import fitz
from PIL import Image
import yaml

CONCEPT_RECORD = "20576708"
PREVIOUS_RECORD = "23227964"
PUBLIC_TITLE = "DeepCirc-Interp: Data Files and Code"
RESEARCH_COMMIT = "0556a8bb137b03f67951cdca93d0161aed4565f4"
VF = "figures/setFinalvF"
ZIP_TIMESTAMP = (2026, 10, 30, 0, 0, 0)
SOURCE_ROOTS = {"data", "figures", "interp", "topology", "scripts", "tests", "docs", ".github"}
ROOT_FILES = {".gitignore", ".gitattributes", ".gitmodules", ".zenodo.json", "CITATION.cff",
              "LICENSE", "LICENSE.md", "NOTICE", "README.md", "Makefile", "pyproject.toml",
              "requirements.txt", "environment.yml", "setup.py", "setup.cfg"}
REQUIRED_NORMALIZED_INPUTS = {
    f"{VF}/data/l2_presence_enrichment_v2.csv",
    f"{VF}/data/l2_presence_denominators_v2.csv",
    f"{VF}/data/l2_presence_summary_v2.json",
}


def git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True)
    if result.returncode:
        raise ValueError(result.stderr.decode(errors="replace"))
    return result.stdout


def fingerprint(data: bytes) -> dict:
    return {"size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "md5": hashlib.md5(data).hexdigest()}


def allowed_source_path(name: str) -> bool:
    """Explicit source locations; exclude runtime, credential and OS sidecars."""
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(p in ("", ".", "..") for p in path.parts):
        return False
    forbidden = {".git", ".aws", ".codex", ".agents", ".venv", "venv", "__pycache__",
                 "node_modules", "dist", "build", "logs", "runs", "checkpoints", "scratch", "tmp"}
    if any(p in forbidden or p.startswith("._") or p.startswith(".!") for p in path.parts):
        return False
    lower = path.name.lower()
    if (lower.startswith(".env") or lower in {".netrc", ".npmrc", "credentials", "auth.json"}
            or lower.endswith((".pem", ".key", ".p12", ".pfx", ".pyc", ".log", ".ai"))
            or lower.startswith(("zenodo_token", "zenodo-token", "access_token", "access-token"))):
        return False
    if len(path.parts) == 1:
        return name in ROOT_FILES
    if path.parts[0] not in SOURCE_ROOTS:
        return False
    if path.name == ".gitkeep":
        return True
    if path.parts[0] == "figures" and any(p in {"panels", "preview"} for p in path.parts):
        return False
    if name.startswith(VF + "/") and path.name.startswith(("FigS10vF.", "FigS11vF.")):
        return False
    if path.parts[0] == "figures" and "final" in path.parts and not any(p.startswith("tabS0") for p in path.parts):
        return False
    if name.startswith(VF + "/publication/") and path.name in {"config.json", "config.jsx", "padding_audit.json"}:
        return False
    return True


def read_local(repo: Path, name: str) -> bytes:
    path = repo / name
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(repo):
        raise ValueError(f"Missing, nonregular or external file: {name}")
    return path.read_bytes()


def working_source(repo: Path) -> dict[str, bytes]:
    names = git(repo, "ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0")
    return {name: read_local(repo, name) for name in sorted(set(names)) if name and allowed_source_path(name)}


def zip_bytes(entries: dict[str, bytes], prefix: str = "") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(entries.items()):
            info = zipfile.ZipInfo(prefix + name, ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    content = buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        if archive.testzip():
            raise ValueError("Generated ZIP failed CRC validation")
    return content


def tagged_source(repo: Path, ref: str, prefix: str) -> tuple[str, bytes, dict[str, bytes]]:
    commit = git(repo, "rev-parse", "--verify", f"refs/tags/{ref}^{{commit}}").decode().strip()
    content = git(repo, "archive", "--format=zip", f"--prefix={prefix}", commit)
    entries = {}
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        if archive.testzip():
            raise ValueError("Tagged source archive failed CRC validation")
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = info.filename.removeprefix(prefix)
            if not info.filename.startswith(prefix) or not allowed_source_path(name):
                raise ValueError(f"Tagged archive contains excluded content: {name}")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"Tagged archive contains a symlink: {name}")
            entries[name] = archive.read(info)
    current = working_source(repo)
    differences = sorted(name for name in entries.keys() | current.keys() if entries.get(name) != current.get(name))
    if differences:
        raise ValueError("Working source differs from the tag; rebuild from a clean tagged checkout:\n" + "\n".join(differences))
    return commit, content, entries


def rendering_roster() -> dict[str, str]:
    roster = {}
    for number, panels in ((10, "abcde"), (11, "de")):
        for panel in panels:
            for suffix in ("pdf", "svg", "png"):
                folder = "raster" if suffix == "png" else "vector"
                roster[f"{VF}/figS{number}/panels/{folder}/panel_{panel}.{suffix}"] = "source panel"
        for suffix in ("pdf", "png"):
            roster[f"{VF}/figS{number}/final/figure_S{number}.{suffix}"] = "composer preview; not final Illustrator artwork"
        roster[f"{VF}/figS{number}/manifest.yaml"] = "placement manifest"
        for panel in "de":
            for suffix in ("pdf", "png"):
                roster[f"{VF}/publication/panels/figS{number}/panel_{panel}.{suffix}"] = "prepared native-scale placement page"
    for number in (1, 2, 3):
        for suffix in ("csv", "pdf", "png", "svg"):
            roster[f"{VF}/tabS0{number}/final/table_S{number}.{suffix}"] = "table deliverable"
    return roster


def validate_artifact(name: str, data: bytes) -> None:
    suffix = PurePosixPath(name).suffix
    if not data:
        raise ValueError(f"Empty artifact: {name}")
    try:
        if suffix == ".pdf":
            with fitz.open(stream=data, filetype="pdf") as document:
                if len(document) != 1:
                    raise ValueError("expected one page")
                if "missing preview" in document[0].get_text().lower():
                    raise ValueError("contains a missing-preview placeholder")
        elif suffix == ".png":
            with Image.open(io.BytesIO(data)) as image:
                if image.format != "PNG" or min(image.size) <= 0:
                    raise ValueError("invalid PNG dimensions or format")
                image.verify()
        elif suffix == ".svg":
            if ET.fromstring(data).tag.split("}")[-1] != "svg":
                raise ValueError("not an SVG document")
        elif suffix == ".csv":
            rows = list(csv.reader(io.StringIO(data.decode())))
            if len(rows) < 2 or len(rows[0]) < 2:
                raise ValueError("missing table header or data")
        elif suffix == ".yaml":
            if not isinstance(yaml.safe_load(data), dict):
                raise ValueError("invalid YAML manifest")
    except Exception as exc:
        raise ValueError(f"Invalid artifact {name}: {exc}") from exc


def validate_panel_manifests(repo: Path, renderings: dict[str, bytes]) -> None:
    for number, labels in ((10, set("abcde")), (11, set("ab"))):
        name = f"{VF}/figS{number}/manifest.yaml"
        panels = yaml.safe_load(renderings[name]).get("panels", [])
        if len(panels) != len(labels) or {p.get("id") for p in panels} != labels:
            raise ValueError(f"{name}: incomplete or duplicate panel roster")
        for panel in panels:
            for field, suffix in (("file", ".pdf"), ("preview_file", ".png")):
                if field not in panel:
                    raise ValueError(f"{name}: panel {panel['id']} lacks {field}")
                resolved = (repo / name).parent.joinpath(panel[field]).resolve()
                if not resolved.is_relative_to(repo):
                    raise ValueError(f"{name}: panel path escapes repository")
                rel = resolved.relative_to(repo).as_posix()
                if rel not in renderings or resolved.suffix != suffix:
                    raise ValueError(f"{name}: required panel asset is not packaged: {rel}")


def prepare_bundle(repo: Path, out: Path, *, ref: str | None = None,
                   working_tree: bool = False, version: str | None = None) -> dict:
    repo, out = repo.resolve(), out.resolve()
    if bool(ref) == bool(working_tree):
        raise ValueError("Choose exactly one of --ref or --working-tree")
    version = ref if ref else version
    if not version or not re.fullmatch(r"v\d+\.\d+\.\d+", version):
        raise ValueError("Version must have the form v1.1.1")
    if out.exists():
        raise ValueError(f"Refusing to overwrite existing bundle: {out}")
    prefix = f"DeepCirc-interp-{version[1:]}/"
    base_commit = git(repo, "rev-parse", "HEAD").decode().strip()
    if working_tree:
        source_entries = working_source(repo)
        source_zip = zip_bytes(source_entries, prefix)
        source_commit = None
    else:
        source_commit, source_zip, source_entries = tagged_source(repo, ref, prefix)
    missing = REQUIRED_NORMALIZED_INPUTS.difference(source_entries)
    if missing:
        raise ValueError("Normalized source inputs are absent: " + ", ".join(sorted(missing)))
    roster = rendering_roster()
    renderings = {name: read_local(repo, name) for name in roster}
    for name, data in renderings.items():
        validate_artifact(name, data)
    validate_panel_manifests(repo, renderings)
    for name, role in roster.items():
        if role in {"table deliverable", "placement manifest"} and source_entries.get(name) != renderings[name]:
            raise ValueError(f"Artwork/source mismatch: {name}")
    source_contents = [{"path": name, **fingerprint(data)} for name, data in sorted(source_entries.items())]
    tree_sha = hashlib.sha256(json.dumps(source_contents, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    provenance = (f"UNTAGGED WORKING-TREE REVIEW SNAPSHOT. Base commit: {base_commit}.\n"
                  "No source tag or release commit is claimed; prepare again from the reviewed tag before upload."
                  if working_tree else f"Exact source tag: {ref}. Source commit: {source_commit}.")
    readme = f"""# {PUBLIC_TITLE}: {version} source-panel correction bundle

{provenance}
Source-tree fingerprint: {tree_sha}
Research source: V-SCode/DeepCircMI commit {RESEARCH_COMMIT}
Previous published version: https://doi.org/10.5281/zenodo.{PREVIOUS_RECORD} (v1.1.0)
All-versions DOI: https://doi.org/10.5281/zenodo.{CONCEPT_RECORD}

This revision supplies circuit-presence-normalized S10d/e and Table S3 data,
their analysis/build code, and scientifically corrected S11 source-panel
labels and rendering. S11 attribution values are preserved. Original inclusive
percentile tiers and cutoff ties are retained; enrichment uses the actual
circuit denominator in each tier, conditional on structural context.

The source archive includes small figure inputs, code, tests, manifests, and
tracked table deliverables. The artwork archive includes all seven source
panels (PDF/SVG/PNG), four native-scale prepared placement pages (PDF/PNG),
complete S10/S11 composer previews (PDF/PNG), manifests, and S1–S3 tables
(CSV/PDF/PNG/SVG). Composer outputs are previews, not new Illustrator masters
or final publication exports. The user will regenerate Illustrator artwork.

Manuscript text, Methods, figure legends, and table legends are deliberately
deferred. Unchanged legacy legend files retained in the source repository are
historical prose, not updated descriptions of this correction. They are not
included as standalone items in the artwork archive. Do not use them as the
current legends for corrected panels. No Word document is bundled.

Install requirements and the package, then run `make figures-vf tables-vf`.
Use the bundled data; do not overwrite it with an older Zenodo download.
Arial is required for the intended typography. The private upstream submodule,
full training checkpoints, and large exemplar HDF5 data are not bundled.

Five outer files: two ZIPs, README-ZENODO.md, metadata.json, manifest.json.
The manifest records per-file sizes, MD5 and SHA256 checksums, including source
and artwork members. It cannot checksum itself; hash the final manifest during
upload and compare its remote MD5 directly.

This command performs no network calls, uploads or publication. A working-tree
review snapshot must be replaced by a tagged bundle before upload. Resume the
existing unpublished new-version draft in concept {CONCEPT_RECORD}, or create
a new version from the latest published version record (previously
{PREVIOUS_RECORD}); do not create a separate series. Merge metadata_patch into
the draft metadata, preserving creators, ORCIDs, affiliations, license and other
bibliographic fields. Verify every remote size and MD5 before publication.
"""
    artwork_entries = {"setFinalvF/" + name.split("setFinalvF/", 1)[1]: data for name, data in renderings.items()}
    artwork_entries["README.md"] = readme.encode()
    metadata = {
        "title": PUBLIC_TITLE, "version": version,
        "description": "<p>Interpretability data files and code for DeepCirc. This correction revision adds circuit-presence-normalized enrichment inputs for setFinalvF S10d/e and Table S3, retaining the original inclusive percentile tiers and accounting for their actual circuit counts. S11 attribution values are unchanged; source-panel units, attribution terminology, and display of a small negative interaction are corrected.</p><p>The archives contain reproducible code and small intermediate inputs, seven vector/raster source panels, prepared placement pages, complete composer previews, and supplementary tables. Illustrator regeneration and manuscript, Methods, figure-legend, and table-legend revisions are deferred. Composer previews are not final publication artwork. Historical legends retained in the source are not current descriptions of the corrected panels. Private upstream code, large checkpoints, and exemplar HDF5 data are not bundled.</p>",
        "notes": (f"UNTAGGED REVIEW SNAPSHOT based on {base_commit}; not ready for upload. " if working_tree else f"Source tag {ref}, commit {source_commit}. ")
                 + f"Research source V-SCode/DeepCircMI commit {RESEARCH_COMMIT}. Source-tree SHA256 {tree_sha}. Continues concept DOI 10.5281/zenodo.{CONCEPT_RECORD}. See README-ZENODO.md and manifest.json for scope and checksums.",
    }
    files = {
        f"DeepCirc-interp-{version[1:]}.zip": source_zip,
        f"DeepCirc-interp-{version[1:]}-setFinalvF-source-panels-and-tables.zip": zip_bytes(artwork_entries),
        "README-ZENODO.md": readme.encode(),
        "metadata.json": (json.dumps({"metadata_patch": metadata}, indent=2) + "\n").encode(),
    }
    manifest = {
        "schema_version": 2, "version": version,
        "source_mode": "working-tree-review" if working_tree else "git-tag",
        "source_tag": None if working_tree else ref, "source_commit": source_commit,
        "base_commit": base_commit, "source_tree_sha256": tree_sha,
        "research_source_repository": "V-SCode/DeepCircMI", "research_source_commit": RESEARCH_COMMIT,
        "conceptrecid": CONCEPT_RECORD, "previous_record_id": PREVIOUS_RECORD,
        "status": "prepared locally; not uploaded or published", "ready_for_upload": not working_tree,
        "publication_scope": "Source-panel/data correction; Illustrator and all manuscript prose/legends deferred",
        "artifacts": [{"filename": name, **fingerprint(data)} for name, data in files.items()],
        "source_contents": source_contents,
        "rendering_contents": [{"path": name, "archive_path": "setFinalvF/" + name.split("setFinalvF/", 1)[1],
                                "role": roster[name], "origin": "verified working rendering" if roster[name] not in {"table deliverable", "placement manifest"} else "same bytes as source archive",
                                **fingerprint(data)} for name, data in sorted(renderings.items())],
        "deferred": ["Illustrator masters and final exports", "manuscript text", "Methods", "figure legends", "table legends"],
    }
    # Detect concurrent source/artwork edits before writing a reviewable bundle.
    if working_source(repo) != source_entries:
        raise ValueError("Source changed during packaging; nothing written")
    for name, data in renderings.items():
        if read_local(repo, name) != data:
            raise ValueError(f"Rendering changed during packaging: {name}; nothing written")
    files["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    out.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        (out / name).write_bytes(data)
    return {"directory": str(out), "source_mode": manifest["source_mode"],
            "source_tree_sha256": tree_sha, "artifacts": manifest["artifacts"]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ref", help="Existing reviewed version tag")
    group.add_argument("--working-tree", action="store_true", help="Prepare an explicitly untagged review snapshot")
    parser.add_argument("--version", help="Proposed version for --working-tree, e.g. v1.1.1")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.ref and args.version:
        parser.error("--version is only used with --working-tree")
    try:
        result = prepare_bundle(args.repo, args.out, ref=args.ref, working_tree=args.working_tree, version=args.version)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Release preparation failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
