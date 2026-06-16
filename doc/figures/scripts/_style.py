"""Shared seaborn style + helpers for thesis figures.

Import:
    from _style import setup, FIG_DIR, EXP_DIR, load_runs, save
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

REPO = Path(__file__).resolve().parents[3]
EXP_DIR = REPO / "exp"
FIG_DIR = Path(__file__).resolve().parents[1]


def setup(font_scale: float = 0.95) -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font_scale=font_scale,
        rc={
            "figure.dpi": 120,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.titleweight": "bold",
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )
    sns.set_palette("colorblind")


PALETTE = sns.color_palette("colorblind")
HIGHLIGHT = "#d62728"


def load_runs(exp: str) -> list[dict]:
    """Load all benchmark JSONs for an exp folder, merging by (embedding_type, limit_parameter).

    When the same (extractor, N) pair appears in multiple JSONs the newest file wins,
    so partial re-runs overwrite only the runs they touched.
    """
    out = EXP_DIR / exp / "output"
    files = sorted(out.glob("benchmark_*.json"))
    if not files:
        raise FileNotFoundError(f"no benchmark JSON in {out}")
    # key → (file_mtime, run_dict); later files (newer mtime) overwrite earlier ones
    merged: dict[tuple, dict] = {}
    for path in files:
        mtime = path.stat().st_mtime
        with path.open() as f:
            results = json.load(f)["results"]
        for r in results:
            rp = r.get("reduction_params")
            key = (r.get("embedding_type"), r.get("limit_parameter"), r.get("reduction_type"), json.dumps(rp, sort_keys=True) if isinstance(rp, dict) else rp, r.get("clustering_type"))
            existing = merged.get(key)
            if existing is None or mtime > existing[0]:
                merged[key] = (mtime, r)
    return [v for _, v in merged.values()]


def save(fig, name: str) -> Path:
    """Save figure as PDF to doc/figures/."""
    out = FIG_DIR / f"{name}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO)}")
    return out


# Short display labels reused across plots
EXTRACTOR_SHORT = {
    "mobilenet_v3": "mobilenet_v3",
    "mobilenet_v3_graffiti_author_head": "mob.+author",
    "mobilenet_v3_graffiti_style_head": "mob.+style",
    "dinov2_vits14": "dinov2",
    "dinov2_graffiti_author_head": "dinov2+author",
    "dinov2_graffiti_style_head": "dinov2+style",
    "resnet50": "resnet50",
    "vgg16": "vgg16",
    "inception_v3": "inception_v3",
    "clip_vit_b32": "clip_b32",
    "yolon": "yolon",
    "yolom": "yolom",
}
