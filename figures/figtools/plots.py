"""Reusable Matplotlib plotting primitives in DeepCirc paper style.

Every helper accepts data arrays / DataFrames and a Matplotlib Axes (or
creates one). They do NOT touch the data — only the formatting. Use them
as building blocks inside panel-building scripts.

Typical usage:
    from figtools import use_style, save_panel
    from figtools.plots import deepcirc_barplot
    import matplotlib.pyplot as plt

    use_style()
    fig, ax = plt.subplots(figsize=(2.4, 1.6))
    deepcirc_barplot(
        ax,
        categories=["computational", "experimental"],
        values=[3.21, 3.05],
        errors=[0.14, 0.12],
        highlight=[True, False],
    )
    save_panel(fig, "panels/vector/panel_f")
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from ..styles.colors import (
    ACCENT_BLUE, ACCENT_BLUE_LIGHT, ACCENT_BLUE_DARK,
    DARK_GRAY, MID_GRAY, LIGHT_GRAY, VERY_LIGHT_GRAY,
    apply_axis_style,
)


# ---------------------------------------------------------------------------
# Bar plot
# ---------------------------------------------------------------------------

def deepcirc_barplot(
    ax,
    categories: Sequence[str],
    values: Sequence[float],
    *,
    errors: Sequence[float] | None = None,
    highlight: Sequence[bool] | None = None,
    bar_width: float = 0.7,
    rotation: float = 0,
    ylabel: str | None = None,
    title: str | None = None,
):
    """Draw a clean bar plot with blue highlighted bars and gray others.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    categories : sequence of str
        Category labels (placed on x-axis).
    values : sequence of float
        Bar heights.
    errors : sequence of float, optional
        Symmetric error bar values (e.g., s.d. or s.e.m.).
    highlight : sequence of bool, optional
        Per-bar boolean mask. True bars are painted ACCENT_BLUE; False bars
        are painted LIGHT_GRAY. If omitted, all bars are highlighted.
    bar_width : float
    rotation : float
        x-tick rotation in degrees.
    """
    n = len(categories)
    x = np.arange(n)
    values = np.asarray(values, dtype=float)
    highlight = (
        np.ones(n, dtype=bool) if highlight is None
        else np.asarray(highlight, dtype=bool)
    )
    colors = np.where(highlight, ACCENT_BLUE, LIGHT_GRAY)

    bars = ax.bar(
        x, values,
        width=bar_width,
        color=colors,
        edgecolor=DARK_GRAY,
        linewidth=0.4,
    )
    if errors is not None:
        ax.errorbar(
            x, values,
            yerr=np.asarray(errors, dtype=float),
            fmt="none",
            ecolor=DARK_GRAY,
            elinewidth=0.6,
            capsize=2.0,
            capthick=0.6,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=rotation, ha="right" if rotation else "center")
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left")
    apply_axis_style(ax)
    ax.margins(x=0.02)
    return bars


# ---------------------------------------------------------------------------
# Scatter
# ---------------------------------------------------------------------------

def deepcirc_scatter(
    ax,
    x: Sequence[float],
    y: Sequence[float],
    *,
    background_x: Sequence[float] | None = None,
    background_y: Sequence[float] | None = None,
    diagonal: bool = False,
    diag_color: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    marker_size: float = 6,
    accent: str = ACCENT_BLUE,
):
    """Scatter with optional gray background cloud + optional diagonal y=x line."""
    if background_x is not None and background_y is not None:
        ax.scatter(
            background_x, background_y,
            s=marker_size * 0.6, c=LIGHT_GRAY, alpha=0.7,
            edgecolor="none", rasterized=True,
        )
    ax.scatter(
        x, y,
        s=marker_size, c=accent, alpha=0.9,
        edgecolor="none", rasterized=False,
    )
    if diagonal:
        lo = min(np.nanmin(x), np.nanmin(y))
        hi = max(np.nanmax(x), np.nanmax(y))
        ax.plot([lo, hi], [lo, hi],
                color=diag_color or MID_GRAY,
                linewidth=0.6, linestyle="--", zorder=0)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left")
    apply_axis_style(ax)
    return ax


# ---------------------------------------------------------------------------
# Line plot
# ---------------------------------------------------------------------------

def deepcirc_lineplot(
    ax,
    x: Sequence[float],
    y: Sequence[float],
    *,
    err: Sequence[float] | None = None,
    marker: str | None = "o",
    label: str | None = None,
    accent: str = ACCENT_BLUE,
    band: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
):
    """Single line with optional mean ± s.d. band or error bars."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    ax.plot(
        x_arr, y_arr,
        color=accent, linewidth=1.0,
        marker=marker, markersize=2.5,
        markerfacecolor=accent, markeredgecolor="none",
        label=label,
    )
    if err is not None:
        err_arr = np.asarray(err, dtype=float)
        if band:
            ax.fill_between(
                x_arr, y_arr - err_arr, y_arr + err_arr,
                color=ACCENT_BLUE_LIGHT, alpha=0.35, linewidth=0,
            )
        else:
            ax.errorbar(
                x_arr, y_arr, yerr=err_arr,
                ecolor=DARK_GRAY, elinewidth=0.5, capsize=1.5, capthick=0.4,
                fmt="none",
            )
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    apply_axis_style(ax)
    return ax


# ---------------------------------------------------------------------------
# Design-space scatter / pseudo-landscape
# ---------------------------------------------------------------------------

def deepcirc_design_space(
    ax,
    x: Sequence[float],
    y: Sequence[float],
    *,
    highlight_x: Sequence[float] | None = None,
    highlight_y: Sequence[float] | None = None,
    densify: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
    point_size: float = 1.8,
):
    """Dense pseudo-landscape scatter with optional highlighted top design(s).

    The cloud is rendered in LIGHT_GRAY with low alpha; the highlight points
    are larger ACCENT_BLUE markers with a thin dark outline so they read
    on top of the cloud.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    ax.scatter(
        x_arr, y_arr,
        s=point_size,
        c=LIGHT_GRAY if not densify else MID_GRAY,
        alpha=0.35 if densify else 0.7,
        edgecolor="none",
        rasterized=True,
    )
    if highlight_x is not None and highlight_y is not None:
        ax.scatter(
            np.asarray(highlight_x), np.asarray(highlight_y),
            s=point_size * 6.0,
            c=ACCENT_BLUE,
            edgecolor=DARK_GRAY,
            linewidths=0.5,
            zorder=10,
        )
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    apply_axis_style(ax)
    return ax


# ---------------------------------------------------------------------------
# Small multiples grid
# ---------------------------------------------------------------------------

def deepcirc_small_multiples(
    fig,
    grid: tuple[int, int],
    data_iter: Iterable,
    *,
    plotter,
    sharex: bool = True,
    sharey: bool = True,
    panel_labels: Sequence[str] | None = None,
    pad: float = 0.4,
):
    """Build a small-multiples grid where each subplot is drawn by `plotter`.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    grid : (rows, cols)
    data_iter : iterable
        Yields one item per subplot. Each item is passed to `plotter`.
    plotter : callable(ax, item) -> None
        User-defined draw function. Receives the prepared Axes and the
        next item from `data_iter`.
    sharex, sharey : bool
        Whether to share axes across the grid.
    panel_labels : sequence of str, optional
        Lowercase panel labels (a, b, c, ...) — one per subplot.
    """
    rows, cols = grid
    axes = fig.subplots(rows, cols, sharex=sharex, sharey=sharey)
    axes_flat = np.atleast_1d(axes).ravel()
    items = list(data_iter)
    for idx, ax in enumerate(axes_flat):
        if idx < len(items):
            plotter(ax, items[idx])
            apply_axis_style(ax)
        else:
            ax.axis("off")
    if panel_labels is not None:
        for label, ax in zip(panel_labels, axes_flat):
            ax.text(
                -0.18, 1.05, str(label).lower(),
                transform=ax.transAxes,
                fontsize=9, fontweight="bold",
                color=DARK_GRAY, ha="left", va="bottom",
            )
    fig.tight_layout(pad=pad)
    return axes_flat


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------

def deepcirc_heatmap(
    ax,
    matrix,
    *,
    row_labels: Sequence[str] | None = None,
    col_labels: Sequence[str] | None = None,
    cmap: str = "Blues",
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str | None = None,
    show_cbar: bool = True,
):
    """Clean heatmap with thin border and compact labels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    matrix : 2-D array-like
    row_labels / col_labels : sequence of str, optional
    cmap : str, default "Blues"
        Sequential map for unsigned data. Use "RdBu_r" for diverging.
    vmin / vmax : float, optional
    cbar_label : str, optional
    show_cbar : bool
        If False, the caller is responsible for the colorbar.
    """
    matrix = np.asarray(matrix, dtype=float)
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    if col_labels is not None:
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")
    else:
        ax.set_xticks([])
    if row_labels is not None:
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)
    else:
        ax.set_yticks([])

    # Thin border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(DARK_GRAY)
        spine.set_linewidth(0.4)

    ax.tick_params(which="both", length=0)

    if show_cbar:
        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        cbar.outline.set_linewidth(0.4)
        cbar.outline.set_edgecolor(DARK_GRAY)
        cbar.ax.tick_params(labelsize=6, width=0.4, length=2)
        if cbar_label:
            cbar.set_label(cbar_label, fontsize=6, color=DARK_GRAY)
    return im


__all__ = [
    "deepcirc_barplot", "deepcirc_scatter", "deepcirc_lineplot",
    "deepcirc_design_space", "deepcirc_small_multiples", "deepcirc_heatmap",
]
