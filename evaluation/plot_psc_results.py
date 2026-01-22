#!/usr/bin/env python3
"""
plot_psc_results.py

Creates plots from:
A) Initial single-modality runs (nestedcv_oof_predictions.csv with columns: y_true,y_oof_prob)
B) Unified multi-modal run (oof_predictions.csv with proba_* columns)
C) SHAP summaries:
   - Early: shap_multi_early.csv (feature, mean_abs_shap) -> top 20 bar plot
   - Late : shap_multi_late_meta_lr.csv (feature, mean_abs_shap) -> modality bar plot

Outputs (in --out-dir):
  - roc_single_only.png
  - roc_multi_and_single.png
  - shap_early_late_combined.png   (subplots: Early top20 + Late modality importance)

Defaults:
  --out-dir defaults to: /common/mcgoverndlab/usr/Miad/PSC/results_no_leakage/plots
  DPI defaults to 1000 per your request.

Usage:
  python plot_psc_results.py
  python plot_psc_results.py --also-pdf
  python plot_psc_results.py --out-dir /some/other/path

Notes:
- Uses matplotlib only (no seaborn).
- Does not set any custom colors unless you later request it.
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, roc_auc_score


# -------------------------
# Utilities
# -------------------------

DEFAULT_OUT_DIR = "/common/mcgoverndlab/usr/Miad/PSC/results_no_leakage/plots"
DEFAULT_DPI = 1000


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def savefig_all(fig, path_png: str, dpi: int, also_pdf: bool = False) -> None:
    fig.tight_layout()
    fig.savefig(path_png, dpi=dpi, bbox_inches="tight")
    if also_pdf:
        fig.savefig(path_png.replace(".png", ".pdf"), bbox_inches="tight")


def load_old_single_oof(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Old format: y_true,y_oof_prob
    """
    df = pd.read_csv(path)
    if not {"y_true", "y_oof_prob"}.issubset(df.columns):
        raise ValueError(f"{path} must contain columns y_true,y_oof_prob. Found: {df.columns.tolist()}")
    y_true = df["y_true"].to_numpy().astype(int)
    y_prob = df["y_oof_prob"].to_numpy().astype(float)
    return y_true, y_prob


def load_new_multi_oof(path: str) -> pd.DataFrame:
    """
    New unified format: expects y_true and proba_* columns
    """
    df = pd.read_csv(path)
    if "y_true" not in df.columns:
        raise ValueError(f"{path} must contain y_true. Found: {df.columns.tolist()}")
    return df


def plot_roc_curves(curves: Dict[str, Tuple[np.ndarray, np.ndarray]], title: str) -> plt.Figure:
    """
    curves: name -> (y_true, y_prob)
    """
    fig = plt.figure(figsize=(6.6, 5.4))
    ax = plt.gca()

    # Chance line
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Chance")

    for name, (y_true, y_prob) in curves.items():
        auc = roc_auc_score(y_true, y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, linewidth=2.0, label=f"{name} (AUC = {auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linewidth=0.5, alpha=0.4)
    return fig


def clean_feature_name(name: str) -> str:
    """
    Clean feature names for plotting:
      - remove num__ prefix
      - remove trailing " EU"
      - normalize spacing
    """
    name = re.sub(r"^num__", "", str(name))
    name = re.sub(r"\s+EU$", "", name)
    return name.strip()


def plot_shap_early_and_late(
    shap_early_path: str,
    shap_late_path: str,
    out_path_png: str,
    dpi: int,
    also_pdf: bool = False,
    top_k: int = 20,
) -> None:
    """
    Single figure with two subplots:
      (A) Early integration SHAP top-k features
      (B) Late integration meta-model SHAP (modality level)
    """
    shap_early = pd.read_csv(shap_early_path)
    shap_late = pd.read_csv(shap_late_path)

    if not {"feature", "mean_abs_shap"}.issubset(shap_early.columns):
        raise ValueError(f"Early SHAP must have feature, mean_abs_shap. Found: {shap_early.columns.tolist()}")
    if not {"feature", "mean_abs_shap"}.issubset(shap_late.columns):
        raise ValueError(f"Late SHAP must have feature, mean_abs_shap. Found: {shap_late.columns.tolist()}")

    shap_early = shap_early.copy()
    shap_early["feature"] = shap_early["feature"].apply(clean_feature_name)
    shap_early["mean_abs_shap"] = pd.to_numeric(shap_early["mean_abs_shap"], errors="coerce")
    shap_early = shap_early.dropna(subset=["mean_abs_shap"]).sort_values("mean_abs_shap", ascending=False).head(top_k)
    shap_early = shap_early.iloc[::-1]

    shap_late = shap_late.copy()
    shap_late["feature"] = (
        shap_late["feature"]
        .astype(str)
        .str.replace("^p_", "", regex=True)
        .str.title()
    )
    shap_late["mean_abs_shap"] = pd.to_numeric(shap_late["mean_abs_shap"], errors="coerce")
    shap_late = shap_late.dropna(subset=["mean_abs_shap"]).sort_values("mean_abs_shap", ascending=False)
    shap_late = shap_late.iloc[::-1]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(14, 6),
        gridspec_kw={"width_ratios": [3, 1]},
    )

    # (A) Early
    axes[0].barh(shap_early["feature"], shap_early["mean_abs_shap"])
    axes[0].set_title("(A) Early Integration: Top 20 Feature Importance (SHAP)")
    axes[0].set_xlabel("Mean(|SHAP value|)")
    axes[0].grid(True, axis="x", alpha=0.4)

    # (B) Late
    axes[1].barh(shap_late["feature"], shap_late["mean_abs_shap"])
    axes[1].set_title("(B) Late Integration")
    axes[1].set_xlabel("Mean(|SHAP value|)")
    axes[1].grid(True, axis="x", alpha=0.4)

    savefig_all(fig, out_path_png, dpi=dpi, also_pdf=also_pdf)
    plt.close(fig)


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()

    # Output
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        help=f"Output directory for plots (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"DPI for PNG outputs (default: {DEFAULT_DPI})",
    )
    parser.add_argument("--also-pdf", action="store_true", help="Also save PDF copies.")

    # Old single-modality OOFs
    parser.add_argument(
        "--single-clinical",
        default="/common/mcgoverndlab/usr/Miad/PSC/results_no_leakage/clinical/nestedcv_oof_predictions.csv",
    )
    parser.add_argument(
        "--single-genetics",
        default="/common/mcgoverndlab/usr/Miad/PSC/results_no_leakage/genetics/nestedcv_oof_predictions.csv",
    )
    parser.add_argument(
        "--single-lab",
        default="/common/mcgoverndlab/usr/Miad/PSC/results_no_leakage/lab/nestedcv_oof_predictions.csv",
    )
    parser.add_argument(
        "--single-serology",
        default="/common/mcgoverndlab/usr/Miad/PSC/results_no_leakage/serology/nestedcv_oof_predictions.csv",
    )

    # New unified run outputs
    parser.add_argument(
        "--multi-oof",
        default="/common/mcgoverndlab/usr/Miad/PSC/results_no_leakage/multi_modal_all_in_one/oof_predictions.csv",
    )
    parser.add_argument(
        "--shap-early",
        default="/common/mcgoverndlab/usr/Miad/PSC/results_no_leakage/multi_modal_all_in_one/shap_multi_early.csv",
    )
    parser.add_argument(
        "--shap-late",
        default="/common/mcgoverndlab/usr/Miad/PSC/results_no_leakage/multi_modal_all_in_one/shap_multi_late_meta_lr.csv",
    )

    args = parser.parse_args()
    ensure_dir(args.out_dir)

    # -------------------------
    # 1) ROC: old single modalities only
    # -------------------------
    old_curves = {}
    for name, path in [
        ("Lab", args.single_lab),
        ("Serology", args.single_serology),
        ("Clinical", args.single_clinical),
        ("Genetics", args.single_genetics),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file for {name}: {path}")
        y_true, y_prob = load_old_single_oof(path)
        old_curves[name] = (y_true, y_prob)

    fig1 = plot_roc_curves(old_curves, title="ROC Curves (Single Modalities)")
    savefig_all(fig1, os.path.join(args.out_dir, "roc_single_only.png"), dpi=args.dpi, also_pdf=args.also_pdf)
    plt.close(fig1)

    # -------------------------
    # 2) ROC: new multi + new singles together
    # -------------------------
    if not os.path.exists(args.multi_oof):
        raise FileNotFoundError(f"Missing multi OOF file: {args.multi_oof}")

    df_new = load_new_multi_oof(args.multi_oof)
    y_true_new = df_new["y_true"].to_numpy().astype(int)

    required_cols = {
        "Multi-modality (LI)": "proba_multi_late",
        "Multi-modality (EI)": "proba_multi_early",
        "Serology": "proba_single_serology",
        "Lab": "proba_single_lab",
        "Clinical": "proba_single_clinical",
        "Genetics": "proba_single_genetics",
    }
    missing = [col for col in required_cols.values() if col not in df_new.columns]
    if missing:
        raise ValueError(f"Missing required columns in multi OOF file: {missing}")

    new_curves = {
        name: (y_true_new, df_new[col].to_numpy().astype(float))
        for name, col in required_cols.items()
    }

    fig2 = plot_roc_curves(new_curves, title="ROC Curves (Multi + Single; Intersection OOF)")
    savefig_all(fig2, os.path.join(args.out_dir, "roc_multi_and_single.png"), dpi=args.dpi, also_pdf=args.also_pdf)
    plt.close(fig2)

    # -------------------------
    # 3) SHAP: combined subplot figure (Early top20 + Late modality)
    # -------------------------
    if not os.path.exists(args.shap_early):
        raise FileNotFoundError(f"Missing early SHAP file: {args.shap_early}")
    if not os.path.exists(args.shap_late):
        raise FileNotFoundError(f"Missing late SHAP file: {args.shap_late}")

    plot_shap_early_and_late(
        shap_early_path=args.shap_early,
        shap_late_path=args.shap_late,
        out_path_png=os.path.join(args.out_dir, "shap_early_late_combined.png"),
        dpi=args.dpi,
        also_pdf=args.also_pdf,
        top_k=20,
    )

    print("\nSaved plots to:", args.out_dir)
    print(" - roc_single_only.png")
    print(" - roc_multi_and_single.png")
    print(" - shap_early_late_combined.png")
    if args.also_pdf:
        print(" (+ PDF copies)")


if __name__ == "__main__":
    main()

