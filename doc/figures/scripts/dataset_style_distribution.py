"""Style dataset class distribution (train vs eval crops)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from _style import REPO, save, setup

DATA_DIR = REPO / "data" / "style"
SPLITS = {"Entrenamiento": "train_crop", "Evaluación": "eval_crop"}
CLASS_LABEL = {
    "tag": "tag",
    "throw-up": "throw-up",
    "piece": "piece",
    "character": "character",
}


def load_counts() -> pd.DataFrame:
    rows = []
    for split_label, subdir in SPLITS.items():
        labels = pd.read_csv(DATA_DIR / subdir / "labels.csv")
        counts = labels["style"].value_counts()
        for style, n in counts.items():
            rows.append({"split": split_label, "style": style, "count": int(n)})
    return pd.DataFrame(rows)


def main() -> None:
    setup()
    df = load_counts()

    totals = df.groupby("style")["count"].sum().sort_values(ascending=False)
    order = totals.index.tolist()
    df["style"] = pd.Categorical(df["style"], categories=order, ordered=True)

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    sns.barplot(
        data=df,
        x="style",
        y="count",
        hue="split",
        order=order,
        ax=ax,
        edgecolor="white",
        linewidth=0.4,
    )

    for container in ax.containers:
        ax.bar_label(container, fontsize=7, padding=2)

    total = int(df["count"].sum())
    ax.set_xlabel("")
    ax.set_ylabel("Número de recortes")
    ax.set_title(f"Distribución por estilo en el conjunto etiquetado (N={total})")
    ax.legend(title="", loc="upper right")
    ax.margins(y=0.15)

    save(fig, "dataset_style_distribution")


if __name__ == "__main__":
    main()
