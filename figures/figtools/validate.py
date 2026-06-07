"""Validation logic for figure manifests + asset files.

Each `check_*` function takes a Manifest (and optionally extra context) and
returns a list of `ValidationFinding`s. Findings carry a severity:
- ERROR   : blocking — manifest cannot be assembled
- WARN    : the figure will assemble but may look wrong at print scale
- INFO    : observation worth surfacing to the user

`scripts/validate_figure.py` is the CLI wrapper that calls all checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

from .layout import Manifest, PanelSpec


# ---------------------------------------------------------------------------
# Datatypes
# ---------------------------------------------------------------------------

SEVERITY_LEVELS = ("ERROR", "WARN", "INFO")


@dataclass
class ValidationFinding:
    severity: str            # one of SEVERITY_LEVELS
    panel_id: str | None     # None for figure-level findings
    message: str

    def render(self) -> str:
        icon = {"ERROR": "[!]", "WARN": "[*]", "INFO": "[i]"}[self.severity]
        scope = f"panel {self.panel_id}" if self.panel_id else "figure"
        return f"  {icon} {self.severity:<5}  {scope:<12}  {self.message}"


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_panel_files_exist(manifest: Manifest) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    for p in manifest.panels:
        abs_path = (manifest.manifest_dir / p.file).resolve()
        if not abs_path.exists():
            findings.append(ValidationFinding(
                "ERROR", p.id, f"linked asset missing: {p.file}"
            ))
    return findings


def check_panels_inside_canvas(manifest: Manifest) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    cw, ch = manifest.canvas_width_mm, manifest.canvas_height_mm
    for p in manifest.panels:
        if p.x_mm < -0.01 or p.y_mm < -0.01:
            findings.append(ValidationFinding(
                "ERROR", p.id,
                f"panel origin outside canvas ({p.x_mm:.1f}, {p.y_mm:.1f})"
            ))
        if p.x1_mm > cw + 0.5 or p.y1_mm > ch + 0.5:
            findings.append(ValidationFinding(
                "WARN", p.id,
                f"panel extent ({p.x1_mm:.1f}, {p.y1_mm:.1f}) "
                f"exceeds canvas ({cw:.1f}, {ch:.1f})"
            ))
    return findings


def check_no_panel_overlap(manifest: Manifest) -> list[ValidationFinding]:
    """Pairwise overlap test (mm). Reports each overlapping pair once."""
    findings: list[ValidationFinding] = []
    panels = manifest.panels
    for i, a in enumerate(panels):
        for b in panels[i + 1:]:
            if _rect_overlap(a, b, slack=0.5):
                findings.append(ValidationFinding(
                    "WARN", None,
                    f"panels {a.id} and {b.id} overlap by >0.5 mm"
                ))
    return findings


def check_label_present(manifest: Manifest) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    for p in manifest.panels:
        if p.label is None or not str(p.label.text).strip():
            findings.append(ValidationFinding(
                "WARN", p.id, "no panel label set"
            ))
    return findings


def check_raster_dpi(
    manifest: Manifest, *, min_effective_dpi: float = 300.0
) -> list[ValidationFinding]:
    """Check that any raster preview has >= 300 dpi at its rendered size."""
    findings: list[ValidationFinding] = []
    for p in manifest.panels:
        preview = p.resolve_preview_path(manifest.manifest_dir)
        if preview is None:
            findings.append(ValidationFinding(
                "INFO", p.id, "no raster preview found (vector-only panel)"
            ))
            continue
        try:
            with Image.open(preview) as im:
                px_w, px_h = im.size
        except Exception as e:  # pragma: no cover
            findings.append(ValidationFinding(
                "ERROR", p.id, f"cannot read raster preview ({e})"
            ))
            continue
        width_in = p.w_mm / 25.4
        height_in = p.h_mm / 25.4
        if width_in <= 0 or height_in <= 0:
            continue
        eff_dpi_w = px_w / width_in
        eff_dpi_h = px_h / height_in
        eff_dpi = min(eff_dpi_w, eff_dpi_h)
        if eff_dpi < min_effective_dpi:
            findings.append(ValidationFinding(
                "WARN", p.id,
                f"raster preview ~{eff_dpi:.0f} dpi at print size "
                f"(want >= {min_effective_dpi:.0f})"
            ))
    return findings


def check_extensions(manifest: Manifest) -> list[ValidationFinding]:
    """Flag panels with non-vector primary asset (PNG used as primary file)."""
    findings: list[ValidationFinding] = []
    for p in manifest.panels:
        ext = p.file.suffix.lower()
        if ext == ".png":
            findings.append(ValidationFinding(
                "WARN", p.id,
                "primary asset is PNG; prefer PDF/SVG for vector panels"
            ))
        elif ext not in (".pdf", ".svg", ".tif", ".tiff"):
            findings.append(ValidationFinding(
                "INFO", p.id, f"uncommon primary asset extension `{ext}`"
            ))
    return findings


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

ALL_CHECKS = (
    check_panel_files_exist,
    check_panels_inside_canvas,
    check_no_panel_overlap,
    check_label_present,
    check_raster_dpi,
    check_extensions,
)


def run_all(manifest: Manifest) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    for check in ALL_CHECKS:
        findings.extend(check(manifest))
    return findings


def render_report(
    manifest: Manifest,
    findings: Iterable[ValidationFinding],
) -> str:
    findings = list(findings)
    lines = []
    lines.append(f"validation report for figure `{manifest.figure_id}`")
    lines.append(
        f"  canvas: {manifest.canvas_width_mm:.1f} mm x "
        f"{manifest.canvas_height_mm:.1f} mm"
    )
    lines.append(f"  panels: {len(manifest.panels)}")
    if not findings:
        lines.append("  -> no issues found.")
        return "\n".join(lines)
    counts = {sev: 0 for sev in SEVERITY_LEVELS}
    for f in findings:
        counts[f.severity] += 1
    summary = "  -> " + ", ".join(f"{counts[s]} {s.lower()}" for s in SEVERITY_LEVELS)
    lines.append(summary)
    for f in findings:
        lines.append(f.render())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _rect_overlap(a: PanelSpec, b: PanelSpec, *, slack: float = 0.0) -> bool:
    """Axis-aligned rectangle overlap with an optional slack (mm)."""
    return not (
        a.x1_mm <= b.x_mm + slack or
        b.x1_mm <= a.x_mm + slack or
        a.y1_mm <= b.y_mm + slack or
        b.y1_mm <= a.y_mm + slack
    )


__all__ = [
    "ValidationFinding", "SEVERITY_LEVELS",
    "check_panel_files_exist", "check_panels_inside_canvas",
    "check_no_panel_overlap", "check_label_present",
    "check_raster_dpi", "check_extensions",
    "ALL_CHECKS", "run_all", "render_report",
]
