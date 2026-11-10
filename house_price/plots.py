"""Figures saved to reports/figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ACCENT = "#2a6f97"
MUTED = "#9aa5b1"
WARN = "#c0504d"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 10,
    }
)


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_eda(df: pd.DataFrame, path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    axes[0].hist(df["Price_Lakhs"], bins=40, color=ACCENT, edgecolor="white")
    axes[0].set_title("Price distribution (right-skewed)")
    axes[0].set_xlabel("Price (INR lakhs)")
    axes[0].set_ylabel("Houses")

    order = df.groupby("City")["Price_Lakhs"].median().sort_values().index
    axes[1].boxplot(
        [df.loc[df["City"] == c, "Price_Lakhs"] for c in order],
        tick_labels=order,
        patch_artist=True,
        boxprops={"facecolor": "#cfe3ee"},
        medianprops={"color": ACCENT},
    )
    axes[1].set_title("Price by city")
    axes[1].set_ylabel("Price (INR lakhs)")

    corr = df.select_dtypes("number").corr()
    im = axes[2].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[2].set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right", fontsize=8)
    axes[2].set_yticks(range(len(corr)), corr.columns, fontsize=8)
    axes[2].set_title("Correlation matrix")
    axes[2].grid(False)
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    _save(fig, path)


def plot_model_comparison(table: pd.DataFrame, path) -> None:
    t = table.sort_values("test_RMSE", ascending=False)
    colors = [MUTED if "Baseline" in m else ACCENT for m in t["model"]]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(t["model"], t["test_RMSE"], color=colors)
    for y, (v, r2) in enumerate(zip(t["test_RMSE"], t["test_R2"])):
        ax.text(v, y, f"  {v:,.1f}  (R² {r2:.3f})", va="center", fontsize=9)
    ax.set_xlabel("Test RMSE (INR lakhs), lower is better")
    ax.set_title("Model comparison on held-out test set")
    ax.set_xlim(0, t["test_RMSE"].max() * 1.35)
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def plot_actual_vs_predicted(y_true, y_pred, path, title="Actual vs predicted price") -> None:
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.scatter(y_true, y_pred, s=14, alpha=0.5, color=ACCENT)
    lo, hi = min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], "--", color=WARN, label="Perfect prediction")
    ax.set_xlabel("Actual price (INR lakhs)")
    ax.set_ylabel("Predicted price (INR lakhs)")
    ax.set_title(title)
    ax.legend(frameon=False)
    _save(fig, path)


def plot_residuals(y_true, y_pred, path) -> None:
    resid = np.asarray(y_true) - np.asarray(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].scatter(y_pred, resid, s=14, alpha=0.5, color=ACCENT)
    axes[0].axhline(0, ls="--", color=WARN)
    axes[0].set_xlabel("Predicted price (INR lakhs)")
    axes[0].set_ylabel("Residual (actual − predicted)")
    axes[0].set_title("Residuals vs predicted")
    axes[1].hist(resid, bins=35, color=ACCENT, edgecolor="white")
    axes[1].axvline(0, ls="--", color=WARN)
    axes[1].set_xlabel("Residual (INR lakhs)")
    axes[1].set_ylabel("Houses")
    axes[1].set_title("Residual distribution")
    _save(fig, path)


def plot_importance(names, means, stds, path) -> None:
    order = np.argsort(means)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.barh(np.array(names)[order], np.array(means)[order], xerr=np.array(stds)[order], color=ACCENT)
    ax.set_xlabel("Drop in R² when the feature is shuffled")
    ax.set_title("Permutation feature importance (test set)")
    ax.grid(axis="y", visible=False)
    _save(fig, path)
