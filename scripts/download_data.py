"""Fetch version-pinned DeepCirc-interp data from Zenodo.

The default DOI refers to the published v1.0.0 record. Pass --doi explicitly
for another published version. Two layouts are supported: individually staged
figures__/full__ files, or a source ZIP with one data/topology_g3 subtree.
Only data files are installed under --root (default ./data).

Downloads are checksum-verified before installation. Matching existing files
are retained; different files require --overwrite. If several ZIPs are present,
select the source archive with --archive KEY. --tier full requires actual
back-end artifacts; a figures-only source archive is not a full-tier deposit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

DEFAULT_DOI = "10.5281/zenodo.20576709"
ZENODO_API = "https://zenodo.org/api/records/{record_id}"


def _resolve_record_id(doi: str) -> str:
    match = re.fullmatch(r"(?:https://doi\.org/)?10\.5281/zenodo\.(\d+)", doi.strip())
    if not match:
        raise ValueError(f"Expected a Zenodo DOI, got {doi!r}.")
    return match.group(1)


def _fetch_manifest(record_id: str) -> dict:
    with urllib.request.urlopen(ZENODO_API.format(record_id=record_id)) as response:
        return json.load(response)


def _digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _checksum(file: dict) -> tuple[str, str]:
    checksum = file.get("checksum", "")
    if not isinstance(checksum, str) or ":" not in checksum:
        raise ValueError(f"Missing algorithm-prefixed checksum for {file.get('key')!r}.")
    algorithm, expected = checksum.lower().split(":", 1)
    if algorithm not in {"md5", "sha256"}:
        raise ValueError(f"Unsupported checksum algorithm {algorithm!r}.")
    if not re.fullmatch(r"[0-9a-f]{" + str(hashlib.new(algorithm).digest_size * 2) + "}", expected):
        raise ValueError(f"Malformed {algorithm} checksum for {file.get('key')!r}.")
    return algorithm, expected


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Temporary directories and a user-specified root may be outside cwd.
    print(f"  downloading: {dst.name}", flush=True)
    urllib.request.urlretrieve(url, dst)


def _download_verified(file: dict, dst: Path) -> None:
    algorithm, expected = _checksum(file)
    url = file.get("links", {}).get("self")
    if not isinstance(url, str) or not url:
        raise ValueError(f"Missing download link for {file.get('key')!r}.")
    _download(url, dst)
    if _digest(dst, algorithm) != expected:
        raise ValueError(f"{algorithm.upper()} mismatch for {file['key']!r}; no data installed.")


def _safe_parts(name: str) -> tuple[str, ...]:
    # Do not let PurePath normalize away traversal or empty components.
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise ValueError(f"Unsafe data path: {name!r}.")
    parts = tuple(name.split("/"))
    if any(part in {"", ".", ".."} or ":" in part for part in parts):
        raise ValueError(f"Unsafe data path: {name!r}.")
    return parts


def _tier_path(key: str) -> PurePosixPath:
    prefix, separator, flattened = key.partition("__")
    if separator != "__" or prefix not in {"figures", "full"}:
        raise ValueError(f"Invalid tier-tagged key {key!r}.")
    # stage_for_zenodo.py encodes data/topology_g3/x as data__topology_g3__x.
    parts = _safe_parts(flattened.replace("__", "/"))
    if len(parts) < 2 or parts[0] != "data":
        raise ValueError(f"Tier file must originate under data/: {key!r}.")
    return PurePosixPath(*parts[1:])


def _has_full_artifacts(paths) -> bool:
    # Match the back-end artifact classes staged by FULL_TIER_GLOBS.
    for path in paths:
        parts = path.parts
        if len(parts) == 2 and parts[0] == "topology_g3" and parts[1] in {"population.pkl", "qc_tiers.pkl"}:
            return True
        if len(parts) >= 3 and parts[0] == "topology_g3":
            if (parts[1] == "registries" and path.suffix == ".pkl") or (parts[1] == "mlp_checkpoints" and path.suffix == ".pt"):
                return True
        if len(parts) >= 3 and parts[0] == "exemplars" and re.fullmatch(r"0x.+_design", parts[1]) and path.suffix in {".h5", ".pt", ".csv"}:
            return True
    return False


def _require_full(paths, tier: str) -> None:
    if tier == "full" and not _has_full_artifacts(paths):
        raise ValueError("This deposit has no full-tier back-end artifacts. Use --tier figures, or select a published deposit containing the full data.")


def _destination(root: Path, relative: PurePosixPath) -> Path:
    current = root
    if current.is_symlink():
        raise ValueError(f"Refusing symlink data root: {root}.")
    for part in relative.parts:
        if current.exists() and not current.is_dir():
            raise ValueError(f"Expected a directory at {current}.")
        current = current / part
        if current.is_symlink():
            raise ValueError(f"Refusing symlink destination: {current}.")
    return current


def _install(staged: dict[PurePosixPath, Path], root: Path, overwrite: bool) -> None:
    pending = []
    conflicts = []
    for relative in staged:
        if any(parent in staged for parent in relative.parents):
            raise ValueError(f"Deposit uses the same path as a file and directory: {relative}.")
    # Preflight every destination so a stale file cannot leave a mixed version.
    for relative, source in staged.items():
        destination = _destination(root, relative)
        if destination.exists():
            if not destination.is_file():
                raise ValueError(f"Expected a file at {destination}.")
            if _digest(source) == _digest(destination):
                print(f"  unchanged: {relative}")
                continue
            if not overwrite:
                conflicts.append(str(relative))
        pending.append((relative, source, destination))
    if conflicts:
        raise ValueError("Existing files differ from this deposit: " + ", ".join(conflicts[:5]) + ". Nothing installed. Review your selected version and use --overwrite to replace them.")
    for relative, source, destination in pending:
        destination.parent.mkdir(parents=True, exist_ok=True)
        # A same-directory temporary file makes each replacement atomic.
        descriptor, temporary = tempfile.mkstemp(prefix=".zenodo-", dir=destination.parent)
        os.close(descriptor)
        temporary_path = Path(temporary)
        try:
            shutil.copyfile(source, temporary_path)
            os.replace(temporary_path, destination)
        finally:
            temporary_path.unlink(missing_ok=True)
        print(f"  installed: {relative}")


def _select_archive(files: list[dict], key: str | None) -> dict:
    archives = [file for file in files if file.get("key", "").lower().endswith(".zip")]
    if key is not None:
        archives = [file for file in archives if file["key"] == key]
        if len(archives) != 1:
            raise ValueError(f"--archive {key!r} does not select exactly one ZIP in the deposit.")
    if not archives:
        raise ValueError("No tier-tagged data or source ZIP found in this deposit.")
    if len(archives) > 1:
        raise ValueError("Multiple ZIPs found; select the source archive with --archive KEY: " + ", ".join(file["key"] for file in archives))
    return archives[0]


def _archive_members(archive: zipfile.ZipFile) -> dict[PurePosixPath, zipfile.ZipInfo]:
    members = []
    roots = set()
    for member in archive.infolist():
        name = member.filename[:-1] if member.is_dir() else member.filename
        parts = _safe_parts(name)
        kind = stat.S_IFMT(member.external_attr >> 16)
        if member.flag_bits & 1:
            raise ValueError(f"Refusing encrypted ZIP member: {member.filename!r}.")
        if kind not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise ValueError(f"Refusing non-regular ZIP member: {member.filename!r}.")
        if member.is_dir():
            continue
        members.append((parts, member))
        # Source archives may be rooted directly or have one wrapper directory.
        for index in (0, 1):
            if len(parts) > index + 2 and parts[index:index + 2] == ("data", "topology_g3"):
                roots.add(parts[:index + 1])
    if len(roots) != 1:
        raise ValueError("Source ZIP must contain exactly one data/topology_g3 subtree.")
    data_root = next(iter(roots))
    selected = {}
    for parts, member in members:
        if parts[:len(data_root)] != data_root:
            continue
        relative = PurePosixPath(*parts[len(data_root):])
        if relative in selected:
            raise ValueError(f"Duplicate data path in ZIP: {relative}.")
        selected[relative] = member
    return selected


def _fallback_source_archive(files: list[dict], root: Path, dry_run: bool,
                             *, tier: str = "figures", overwrite: bool = False,
                             archive_key: str | None = None) -> int:
    file = _select_archive(files, archive_key)
    _checksum(file)
    print(f"Source archive: {file['key']}")
    if dry_run:
        print(f"  [dry] download, verify and inspect data files for {root}; contents and tier availability are not yet verified")
        return 0
    with tempfile.TemporaryDirectory(prefix="deepcirc-data-") as temporary:
        temporary_root = Path(temporary)
        zip_path = temporary_root / "source.zip"  # Never trust a record key as a path.
        _download_verified(file, zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            members = _archive_members(archive)
            _require_full(members, tier)
            staged = {}
            for index, (relative, member) in enumerate(members.items()):
                destination = temporary_root / f"member-{index}"
                with archive.open(member) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)
                staged[relative] = destination
        _install(staged, root, overwrite)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("figures", "full"), default="figures")
    parser.add_argument("--doi", default=DEFAULT_DOI)
    parser.add_argument("--root", type=Path, default=Path("./data"))
    parser.add_argument("--dry-run", action="store_true", help="List selected files without downloading or installing.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing files whose contents differ from this selected deposit.")
    parser.add_argument("--archive", help="Select a source ZIP by its exact key, instead of individual tier files.")
    args = parser.parse_args(argv)
    try:
        print(f"Resolving Zenodo record for DOI {args.doi}...", flush=True)
        record = _fetch_manifest(_resolve_record_id(args.doi))
        files = record.get("files", [])
        prefixes = ("figures__",) if args.tier == "figures" else ("figures__", "full__")
        selected = [file for file in files if file.get("key", "").startswith(prefixes)]
        if args.archive or not selected:
            return _fallback_source_archive(files, args.root, args.dry_run, tier=args.tier,
                                            overwrite=args.overwrite, archive_key=args.archive)
        entries = {}
        for file in selected:
            relative = _tier_path(file["key"])
            checksum = _checksum(file)
            if relative in entries:
                # Figures and full staging can legitimately duplicate a file.
                if checksum != _checksum(entries[relative]):
                    raise ValueError(f"Conflicting deposit files map to {relative}.")
                continue
            entries[relative] = file
        _require_full(entries, args.tier)
        print(f"Selected {len(entries)} files for tier={args.tier}.")
        if args.dry_run:
            for relative in entries:
                print(f"  [dry] {args.root / relative}")
            return 0
        with tempfile.TemporaryDirectory(prefix="deepcirc-data-") as temporary:
            staged = {}
            for index, (relative, file) in enumerate(entries.items()):
                destination = Path(temporary) / f"download-{index}"
                _download_verified(file, destination)
                staged[relative] = destination
            _install(staged, args.root, args.overwrite)
        print("Done.", flush=True)
        return 0
    except (ValueError, OSError, urllib.error.URLError, zipfile.BadZipFile) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
