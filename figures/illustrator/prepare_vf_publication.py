#!/usr/bin/env python3
"""Prepare FigS10vF/FigS11vF inputs without changing their source artwork.

Run from any directory with a Python containing pypdf, PyMuPDF, and NumPy:
    /opt/miniconda3/bin/python figures/illustrator/prepare_vf_publication.py

S10 a/b/c retain their exact native page sizes. S10 d/e and S11 d/e are
centered, without scaling, on the approved nominal panel pages. The original
decoded PDF content bytes are placed verbatim in a vector Form XObject.
That form preserves the source coordinate space, including hatch patterns.
Source PDFs, build scripts, data, and manifests are never written.

Outputs are confined to figures/setFinalvF/publication: four placement PDFs
and 150 ppi PNG previews, an audit, readable config.json, and config.jsx for
Illustrator ExtendScript.
Paths in the generated config are resolved from this script's repository.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import fitz
import numpy as np
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject,
    ContentStream,
    DecodedStreamObject,
    DictionaryObject,
    FloatObject,
    NameObject,
    NumberObject,
)

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "figures/setFinalvF"
PUBLICATION = BASE / "publication"
PT_PER_MM = 72.0 / 25.4

# Requested coordinates refer to the top-left of each panel's PDF page.
# Native sizes replace only rounded a/b/c size requests, never their positions.
FIGURES = [
    {
        "name": "FigS10vF", "source_figure": "figS10",
        "width_mm": 240.0, "height_mm": 296.0, "label_size_pt": 12.0,
        "panels": [
            {"label": "a", "source_panel": "a", "x_mm": 4.0, "y_mm": 4.0,
             "requested_width_mm": 94.1, "requested_height_mm": 81.8, "native": True},
            {"label": "b", "source_panel": "b", "x_mm": 4.0, "y_mm": 89.8,
             "requested_width_mm": 89.4, "requested_height_mm": 81.8, "native": True},
            {"label": "c", "source_panel": "c", "x_mm": 100.0, "y_mm": 4.0,
             "requested_width_mm": 123.1, "requested_height_mm": 180.0, "native": True,
             "label_x_offset_mm": -5.0},
            {"label": "d", "source_panel": "d", "x_mm": 4.0, "y_mm": 185.8,
             "requested_width_mm": 115.0, "requested_height_mm": 105.0, "native": False},
            {"label": "e", "source_panel": "e", "x_mm": 121.0, "y_mm": 185.8,
             "requested_width_mm": 115.0, "requested_height_mm": 105.0, "native": False},
        ],
    },
    {
        "name": "FigS11vF", "source_figure": "figS11",
        "width_mm": 370.0, "height_mm": 490.0, "label_size_pt": 20.0,
        "panels": [
            {"label": "a", "source_panel": "d", "x_mm": 5.0, "y_mm": 5.0,
             "requested_width_mm": 360.0, "requested_height_mm": 210.0, "native": False},
            {"label": "b", "source_panel": "e", "x_mm": 5.0, "y_mm": 225.0,
             "requested_width_mm": 360.0, "requested_height_mm": 260.0, "native": False},
        ],
    },
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def page_size_pt(path: Path) -> tuple[float, float]:
    reader = PdfReader(path)
    require(len(reader.pages) == 1, f"Expected one PDF page: {path}")
    page = reader.pages[0]
    require(page.rotation == 0, f"Rotated source page is unsupported: {path}")
    require(list(page.mediabox.lower_left) == [0, 0], f"Nonzero MediaBox origin: {path}")
    require(list(page.cropbox) == list(page.mediabox), f"CropBox differs from MediaBox: {path}")
    return float(page.mediabox.width), float(page.mediabox.height)


def inspect_pdf(path: Path) -> dict:
    width_pt, height_pt = page_size_pt(path)
    with fitz.open(path) as doc:
        page = doc[0]
        fonts = []
        for font in page.get_fonts(full=True):
            program = doc.extract_font(font[0])[3]
            fonts.append({"name": font[3], "type": font[2], "extension": font[1],
                          "bytes": len(program), "sha256": hashlib.sha256(program).hexdigest()})
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
        pixels = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3]
        ys, xs = np.nonzero(np.any(pixels < 240, axis=2))
        require(len(xs) > 0, f"Blank PDF: {path}")
        ink_margins = [float(xs.min()) / 2, float(ys.min()) / 2,
                       float(pix.width - 1 - xs.max()) / 2, float(pix.height - 1 - ys.max()) / 2]
        require(all(font["bytes"] > 0 for font in fonts), f"Unembedded font: {path}")
        require(len(page.get_images(full=True)) == 0, f"Unexpected raster image: {path}")
        return {"path": str(path), "relative_path": relative(path), "sha256": sha256(path),
                "width_pt": width_pt, "height_pt": height_pt,
                "width_mm": width_pt / PT_PER_MM, "height_mm": height_pt / PT_PER_MM,
                "fonts": fonts, "image_count": 0, "drawing_count": len(page.get_drawings()),
                "text": page.get_text(), "ink_margins_pt_ltrb": ink_margins}


def compare_rendering(source: Path, padded: Path, dx: float, dy: float) -> dict:
    """Compare at 144 dpi in matching device coordinates, including hatch phase."""
    with fitz.open(source) as original, fitz.open(padded) as result:
        rect = original[0].rect
        clip = fitz.Rect(dx, dy, dx + rect.width, dy + rect.height)
        before = original[0].get_pixmap(matrix=fitz.Matrix(2, 2).pretranslate(dx, dy), alpha=False)
        after = result[0].get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip, alpha=False)
        require((before.width, before.height) == (after.width, after.height), "Render dimensions differ")
        delta = np.abs(np.frombuffer(before.samples, dtype=np.uint8).astype(float)
                       - np.frombuffer(after.samples, dtype=np.uint8).astype(float))
        # A few antialiased edge pixels can differ through renderer float rounding.
        require(float(delta.mean()) < 0.01, f"Rendered artwork changed: {padded}")
        return {"dpi": 144, "pixel_identical": bool(np.all(delta == 0)),
                "mean_absolute_channel_difference_0_255": float(delta.mean()),
                "max_channel_difference_0_255": float(delta.max()),
                "different_channels": int(np.count_nonzero(delta)), "total_channels": int(delta.size)}


def pad_original(source: Path, output: Path, width_mm: float, height_mm: float) -> dict:
    """Add centered margins; preserve all source content operands and font programs."""
    original = inspect_pdf(source)
    reader = PdfReader(source)
    source_page = reader.pages[0]
    sw, sh = original["width_pt"], original["height_pt"]
    tw, th = width_mm * PT_PER_MM, height_mm * PT_PER_MM
    dx, dy = (tw - sw) / 2, (th - sh) / 2
    require(dx >= 0 and dy >= 0, f"Requested canvas would crop source: {source}")
    original_bytes = source_page.get_contents().get_data()
    original_ops = ContentStream(source_page.get_contents(), reader).operations

    writer = PdfWriter()
    page = writer.add_blank_page(width=tw, height=th)
    form = DecodedStreamObject()
    form.set_data(original_bytes)
    form[NameObject("/Type")] = NameObject("/XObject")
    form[NameObject("/Subtype")] = NameObject("/Form")
    form[NameObject("/FormType")] = NumberObject(1)
    form[NameObject("/BBox")] = ArrayObject([FloatObject(0), FloatObject(0), FloatObject(sw), FloatObject(sh)])
    form[NameObject("/Resources")] = source_page["/Resources"].clone(writer)
    if "/Group" in source_page:
        form[NameObject("/Group")] = source_page["/Group"].clone(writer)
    form_ref = writer._add_object(form)
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/XObject"): DictionaryObject({NameObject("/Original"): form_ref})})
    stream = DecodedStreamObject()
    stream.set_data(f"q\n1 0 0 1 {dx:.12f} {dy:.12f} cm\n/Original Do\nQ\n".encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    writer.add_metadata({"/Title": f"{source.parent.parent.parent.name} {source.stem}: original artwork at 100%",
                         "/Subject": "Centered margins only; original vector artwork and embedded fonts preserved",
                         "/Producer": "DeepCirc prepare_vf_publication.py (pypdf)"})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        writer.write(handle)

    check = PdfReader(output)
    saved_form = check.pages[0]["/Resources"]["/XObject"]["/Original"].get_object()
    require(saved_form.get_data() == original_bytes, "Source content bytes changed")
    require(ContentStream(saved_form, check).operations == original_ops, "Source PDF operands changed")
    wrapper = ContentStream(check.pages[0].get_contents(), check).operations
    require([op for args, op in wrapper] == [b"q", b"cm", b"Do", b"Q"], "Unexpected wrapper operators")
    matrix = wrapper[1][0]
    require(list(matrix[:4]) == [1, 0, 0, 1], "Artwork was scaled, rotated or sheared")
    require(abs(float(matrix[4]) - dx) < 1e-7 and abs(float(matrix[5]) - dy) < 1e-7,
            "Wrong artwork translation")
    padded = inspect_pdf(output)
    require(original["text"] == padded["text"], "PDF text changed")
    require(original["fonts"] == padded["fonts"], "Embedded font programs changed")
    require(original["drawing_count"] == padded["drawing_count"], "Vector drawing count changed")
    require(min(padded["ink_margins_pt_ltrb"]) > 0, "Padded output touches the page edge")
    require(abs(padded["width_mm"] - width_mm) < 1e-6 and abs(padded["height_mm"] - height_mm) < 1e-6,
            "Padded page size is incorrect")
    require(sha256(source) == original["sha256"], "Source PDF changed")
    preview = output.with_suffix(".png")
    with fitz.open(output) as doc:
        doc[0].get_pixmap(dpi=150, alpha=False).save(preview)
    return {"source": original, "placement_input": padded,
            "preview": {"path": str(preview), "relative_path": relative(preview),
                        "dpi": 150, "sha256": sha256(preview)},
            "padding_each_side_mm_xy": [dx / PT_PER_MM, dy / PT_PER_MM],
            "translation_pdf_pt_xy": [dx, dy], "original_content_operator_count": len(original_ops),
            "source_content_bytes_and_operands_preserved": True, "source_unchanged": True,
            "text_and_embedded_font_programs_preserved": True,
            "render_comparison": compare_rendering(source, output, dx, dy)}


def prepare() -> dict:
    PUBLICATION.mkdir(parents=True, exist_ok=True)
    source_paths = [BASE / figure["source_figure"] / "panels/vector" / f"panel_{panel['source_panel']}.pdf"
                    for figure in FIGURES for panel in figure["panels"]]
    before = {relative(path): sha256(path) for path in source_paths}
    config = {"schema_version": 1, "generator": relative(Path(__file__).resolve()),
              "output_dir": str(BASE), "audit_dir": str(PUBLICATION),
              "coordinate_origin": "artboard top-left", "placement_scale_percent": 100,
              "figures": []}
    audit = {"method": "Original vector artwork at 100%; approved centered margins on four panels",
             "sources_sha256_before": before, "panels": []}

    for spec in FIGURES:
        figure = {key: spec[key] for key in ("name", "width_mm", "height_mm", "label_size_pt")}
        figure["label_font"] = "Arial-BoldMT"
        figure["panels"] = []
        for request in spec["panels"]:
            source = BASE / spec["source_figure"] / "panels/vector" / f"panel_{request['source_panel']}.pdf"
            if request["native"]:
                original = inspect_pdf(source)
                placement = original
                record = {"source": original, "placement_input": placement,
                          "padding_each_side_mm_xy": [0.0, 0.0], "translation_pdf_pt_xy": [0.0, 0.0],
                          "source_unchanged": True, "mode": "native_source"}
                width_mm, height_mm = original["width_mm"], original["height_mm"]
                preview_path = source.parent.parent / "raster" / source.with_suffix(".png").name
            else:
                output = PUBLICATION / "panels" / spec["source_figure"] / source.name
                width_mm, height_mm = request["requested_width_mm"], request["requested_height_mm"]
                record = pad_original(source, output, width_mm, height_mm)
                original, placement = record["source"], record["placement_input"]
                record["mode"] = "centered_original_artwork"
                preview_path = Path(record["preview"]["path"])
            x, y = request["x_mm"], request["y_mm"]
            require(x >= 0 and y >= 0 and x + width_mm <= spec["width_mm"]
                    and y + height_mm <= spec["height_mm"], "Panel extends outside its artboard")
            panel = {"label": request["label"], "path": placement["path"],
                     "x_mm": x, "y_mm": y, "width_mm": width_mm, "height_mm": height_mm,
                     "label_x_mm": x + request.get("label_x_offset_mm", -2.5), "label_y_mm": y - 2.0,
                     "requested_width_mm": request["requested_width_mm"],
                     "requested_height_mm": request["requested_height_mm"],
                     "relative_path": placement["relative_path"], "sha256": placement["sha256"],
                     "preview_path": str(preview_path), "preview_relative_path": relative(preview_path),
                     "mode": record["mode"], "scale_percent": 100,
                     "padding_each_side_mm_xy": record["padding_each_side_mm_xy"],
                     "translation_pdf_pt_xy": record["translation_pdf_pt_xy"],
                     "source": {key: original[key] for key in
                                ("path", "relative_path", "sha256", "width_pt", "height_pt", "width_mm", "height_mm")}}
            figure["panels"].append(panel)
            record.update(figure=spec["name"], label=request["label"])
            audit["panels"].append(record)
        config["figures"].append(figure)

    after = {relative(path): sha256(path) for path in source_paths}
    require(before == after, "An original source PDF changed during preparation")
    audit.update(sources_sha256_after=after, all_original_sources_unchanged=True, all_checks_passed=True)
    text = json.dumps(config, indent=2, ensure_ascii=True)
    (PUBLICATION / "config.json").write_text(text + "\n", encoding="utf-8")
    (PUBLICATION / "config.jsx").write_text("// Generated by prepare_vf_publication.py; rerun after moving the repository.\n"
                                           + "var VFPUBLICATION = " + text + ";\n", encoding="utf-8")
    (PUBLICATION / "padding_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=True) + "\n",
                                                    encoding="utf-8")
    return config


if __name__ == "__main__":
    prepared = prepare()
    print(f"Prepared {sum(len(f['panels']) for f in prepared['figures'])} panels for two artboards.")
    print(f"Config: {PUBLICATION / 'config.json'}")
    print(f"ExtendScript config: {PUBLICATION / 'config.jsx'}")
    print("All original source PDFs unchanged; all vector, text, font and rendering checks passed.")
