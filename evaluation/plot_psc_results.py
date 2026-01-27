#!/usr/bin/env python3
"""
plot_psc_results.py

GitHub-friendly plotting script for PSC results.

Creates plots from:
A) (Optional) Old single-modality runs (nestedcv_oof_predictions.csv with columns: y_true,y_oof_prob)
B) Unified multi-modal run (oof_predictions.csv with proba_* columns)
C) SHAP summaries (optional):
   - Early: shap_multi_early.csv (feature, mean_abs_shap) -> top 20 bar plot
   - Late : shap_multi_late_meta_lr.csv (feature, mean_abs_shap) -> modality bar plot
D) Precision-Recall curves
E) NEW: Calibration curves + Brier scores (MULTI-MODAL ONLY; calibrated + uncalibrated)

Outputs (in --out-dir):
  - roc_single_only.png                      (only if old single inputs exist)
  - roc_multi_and_single.png                 (requires unified oof_predictions.csv)
  - pr_multi_and_single.png                  (requires unified oof_predictions.csv)
  - calibration_multi_only_uncalibrated.png  (multi only)
  - calibration_multi_only_calibrated.png    (multi only; only if *_cal columns exist)
  - brier_scores_multi_only_uncalibrated.csv
  - brier_scores_multi_only_calibrated.csv
  - shap_early_late_combined.png             (only if early+late SHAP files exist)

Defaults:
  --out-dir defaults to: results/plots
  --run-dir defaults to: results/multi_modal/latest
  DPI defaults to 300 (GitHub-friendly). You can set --dpi 1000 if desired.

Notes:
- Uses matplotlib only (no seaborn).
- Does not set any custom colors.
- Skips plots gracefully if optional input files are missing.
- Legends are sorted:
    ROC: by AUC (desc)
    PR : by AP  (desc)
    Cal: by Brier (asc; lower is better)
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve


# -------------------------
# Defaults (GitHub-friendly)
# -------------------------

DEFAULT_OUT_DIR = "results/plots"
DEFAULT_RUN_DIR = "results/multi_modal/latest"
DEFAULT_DPI = 300


# -------------------------
# Utilities
# -------------------------

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
    Unified format: expects y_true and proba_* columns
    """
    df = pd.read_csv(path)
    if "y_true" not in df.columns:
        raise ValueError(f"{path} must contain y_true. Found: {df.columns.tolist()}")
    return df


def try_path(p: Optional[str]) -> Optional[str]:
    if p and os.path.exists(p):
        return p
    return None


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


def _prevalence_from_curves(curves: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> float:
    # safe prevalence calc from any curve
    if not curves:
        return 0.0
    y = next(iter(curves.values()))[0]
    return float(np.mean(y))


# -------------------------
# Plotters
# -------------------------

def plot_roc_curves_desc(curves: Dict[str, Tuple[np.ndarray, np.ndarray]], title: str) -> plt.Figure:
    """
    curves: name -> (y_true, y_prob)
    Legend sorted by descending AUC.
    """
    items = []
    for name, (y_true, y_prob) in curves.items():
        auc = roc_auc_score(y_true, y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        items.append((auc, name, fpr, tpr))

    items.sort(key=lambda x: x[0], reverse=True)

    fig = plt.figure(figsize=(6.6, 5.4))
    ax = plt.gca()
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Chance")

    for auc, name, fpr, tpr in items:
        ax.plot(fpr, tpr, linewidth=2.0, label=f"{name} (AUC = {auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, linewidth=0.5, alpha=0.4)
    return fig


def plot_pr_curves_desc(curves: Dict[str, Tuple[np.ndarray, np.ndarray]], title: str) -> plt.Figure:
    """
    curves: name -> (y_true, y_prob)
    Legend sorted by descending AP.
    """
    items = []
    for name, (y_true, y_prob) in curves.items():
        ap = average_precision_score(y_true, y_prob)
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        items.append((ap, name, rec, prec))

    items.sort(key=lambda x: x[0], reverse=True)

    fig = plt.figure(figsize=(6.6, 5.4))
    ax = plt.gca()

    for ap, name, rec, prec in items:
        ax.plot(rec, prec, linewidth=2.0, label=f"{name} (AP = {ap:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, linewidth=0.5, alpha=0.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    return fig


def plot_calibration_curves_desc(
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    title: str,
    n_bins: int,
) -> Tuple[plt.Figure, pd.DataFrame]:
    """
    curves: name -> (y_true, y_prob)
    Legend sorted by ascending Brier (lower is better).
    Returns (fig, brier_df).
    """
    rows = []
    items = []

    for name, (y_true, y_prob) in curves.items():
        y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
        bs = brier_score_loss(y_true, y_prob)
        frac_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="quantile"
        )
        items.append((bs, name, mean_pred, frac_pos))
        rows.append({
            "model": name,
            "brier": float(bs),
            "n": int(len(y_true)),
            "n_pos": int(np.sum(y_true)),
            "prevalence": float(np.mean(y_true)),
        })

    items.sort(key=lambda x: x[0], reverse=False)  # lower brier first

    fig = plt.figure(figsize=(6.6, 5.4))
    ax = plt.gca()

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Perfectly calibrated")
    y0 = _prevalence_from_curves(curves)
    ax.axhline(y0, linestyle=":", linewidth=1.5, label=f"Prevalence = {y0:.3f}")

    for bs, name, mean_pred, frac_pos in items:
        ax.plot(mean_pred, frac_pos, marker="o", linewidth=2.0, label=f"{name} (Brier = {bs:.3f})")

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linewidth=0.5, alpha=0.4)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    brier_df = pd.DataFrame(rows).sort_values("brier", ascending=True).reset_index(drop=True)
    return fig, brier_df


# -------------------------
# SHAP combined plot (unchanged)
# -------------------------

def plot_shap_early_and_late(
    shap_early_path: str,
    shap_late_path: str,
    out_path_png: str,
    dpi: int,
    also_pdf: bool = False,
    top_k: int = 20,
) -> None:
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

    axes[0].barh(shap_early["feature"], shap_early["mean_abs_shap"])
    axes[0].set_title(f"(A) Early Integration: Top {top_k} Feature Importance (SHAP)")
    axes[0].set_xlabel("Mean(|SHAP value|)")
    axes[0].grid(True, axis="x", alpha=0.4)

    axes[1].barh(shap_late["feature"], shap_late["mean_abs_shap"])
    axes[1].set_title("(B) Late Integration (Meta-model)")
    axes[1].set_xlabel("Mean(|SHAP value|)")
    axes[1].grid(True, axis="x", alpha=0.4)

    savefig_all(fig, out_path_png, dpi=dpi, also_pdf=also_pdf)
    plt.close(fig)


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help=f"DPI for PNG outputs (default: {DEFAULT_DPI})")
    parser.add_argument("--also-pdf", action="store_true", help="Also save PDF copies.")

    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR, help=f"Unified run directory (default: {DEFAULT_RUN_DIR})")

    # Old single-modality OOFs (optional)
    parser.add_argument("--single-clinical", default=None, help="Path to old clinical nestedcv_oof_predictions.csv")
    parser.add_argument("--single-genetics", default=None, help="Path to old genetics nestedcv_oof_predictions.csv")
    parser.add_argument("--single-lab", default=None, help="Path to old lab nestedcv_oof_predictions.csv")
    parser.add_argument("--single-serology", default=None, help="Path to old serology nestedcv_oof_predictions.csv")

    # Overrides
    parser.add_argument("--multi-oof", default=None, help="Override: path to unified oof_predictions.csv")
    parser.add_argument("--shap-early", default=None, help="Override: path to shap_multi_early.csv")
    parser.add_argument("--shap-late", default=None, help="Override: path to shap_multi_late_meta_lr.csv")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k features for early SHAP barplot (default: 20)")

    # Calibration
    parser.add_argument("--calib-bins", type=int, default=10, help="Number of bins for calibration curve (default: 10)")

    args = parser.parse_args()
    ensure_dir(args.out_dir)

    run_dir = args.run_dir
    multi_oof = args.multi_oof or os.path.join(run_dir, "oof_predictions.csv")
    shap_early = args.shap_early or os.path.join(run_dir, "shap_multi_early.csv")
    shap_late = args.shap_late or os.path.join(run_dir, "shap_multi_late_meta_lr.csv")

    # -------------------------
    # 1) ROC: old single modalities only (OPTIONAL)
    # -------------------------
    old_paths = {
        "Lab": args.single_lab,
        "Serology": args.single_serology,
        "Clinical": args.single_clinical,
        "Genetics": args.single_genetics,
    }
    old_paths_resolved = {k: try_path(v) for k, v in old_paths.items()}

    if all(old_paths_resolved.values()):
        old_curves = {}
        for name, path in old_paths_resolved.items():
            y_true, y_prob = load_old_single_oof(path)
            old_curves[name] = (y_true, y_prob)

        fig1 = plot_roc_curves_desc(old_curves, title="ROC Curves (Single Modalities; Old Nested CV)")
        savefig_all(fig1, os.path.join(args.out_dir, "roc_single_only.png"), dpi=args.dpi, also_pdf=args.also_pdf)
        plt.close(fig1)
        print(f"Saved: {os.path.join(args.out_dir, 'roc_single_only.png')}")
    else:
        if any(v is not None for v in old_paths.values()):
            missing = [k for k, v in old_paths_resolved.items() if v is None]
            print(f"Skipping roc_single_only.png (missing old single OOF paths for: {missing})")
        else:
            print("Skipping roc_single_only.png (no old single OOF paths provided).")

    # -------------------------
    # 2) Unified OOF required
    # -------------------------
    if not os.path.exists(multi_oof):
        raise FileNotFoundError(
            f"Missing unified OOF file: {multi_oof}\n"
            f"Tip: pass --run-dir /path/to/run_YYYYMMDD_HHMMSS or --multi-oof /path/to/oof_predictions.csv"
        )

    df_new = load_new_multi_oof(multi_oof)
    y_true_new = df_new["y_true"].to_numpy().astype(int)

    # required for ROC/PR (your original behavior)
    required_cols = {
        "Multi-modality (LI)": "proba_multi_late",
        "Multi-modality (EI)": "proba_multi_early",
        "Serology": "proba_single_serology",
        "Lab": "proba_single_lab",
        "Clinical": "proba_single_clinical",
        "Genetics": "proba_single_genetics",
    }
    missing_cols = [col for col in required_cols.values() if col not in df_new.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in unified OOF file: {missing_cols}\n"
            f"Found columns: {df_new.columns.tolist()}"
        )

    curves_all = {
        name: (y_true_new, df_new[col].to_numpy().astype(float))
        for name, col in required_cols.items()
    }

    # -------------------------
    # 3) ROC: unified multi + unified singles
    # -------------------------
    fig2 = plot_roc_curves_desc(curves_all, title="ROC Curves (Multi + Single; Intersection OOF)")
    savefig_all(fig2, os.path.join(args.out_dir, "roc_multi_and_single.png"), dpi=args.dpi, also_pdf=args.also_pdf)
    plt.close(fig2)
    print(f"Saved: {os.path.join(args.out_dir, 'roc_multi_and_single.png')}")

    # -------------------------
    # 4) PR: unified multi + unified singles
    # -------------------------
    fig_pr = plot_pr_curves_desc(curves_all, title="Precision-Recall Curves (Multi + Single; Intersection OOF)")
    savefig_all(fig_pr, os.path.join(args.out_dir, "pr_multi_and_single.png"), dpi=args.dpi, also_pdf=args.also_pdf)
    plt.close(fig_pr)
    print(f"Saved: {os.path.join(args.out_dir, 'pr_multi_and_single.png')}")

    # -------------------------
    # 5) Calibration (MULTI-MODAL ONLY): uncalibrated + calibrated
    # -------------------------
    multi_uncal = {
        "Multi-modality (LI)": (y_true_new, df_new["proba_multi_late"].to_numpy().astype(float)),
        "Multi-modality (EI)": (y_true_new, df_new["proba_multi_early"].to_numpy().astype(float)),
    }

    fig_cal_u, brier_u = plot_calibration_curves_desc(
        multi_uncal,
        title="Calibration Curves (Multi-modal ONLY; Uncalibrated; Intersection OOF)",
        n_bins=args.calib_bins,
    )
    out_png_u = os.path.join(args.out_dir, "calibration_multi_only_uncalibrated.png")
    savefig_all(fig_cal_u, out_png_u, dpi=args.dpi, also_pdf=args.also_pdf)
    plt.close(fig_cal_u)
    print(f"Saved: {out_png_u}")

    brier_u_path = os.path.join(args.out_dir, "brier_scores_multi_only_uncalibrated.csv")
    brier_u.to_csv(brier_u_path, index=False)
    print(f"Saved: {brier_u_path}")
    print("\nBrier (multi only, uncalibrated; lower is better):")
    print(brier_u.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # calibrated curves only if columns exist
    has_cal = ("proba_multi_late_cal" in df_new.columns) and ("proba_multi_early_cal" in df_new.columns)
    if has_cal:
        multi_cal = {
            "Multi-modality (LI) [cal]": (y_true_new, df_new["proba_multi_late_cal"].to_numpy().astype(float)),
            "Multi-modality (EI) [cal]": (y_true_new, df_new["proba_multi_early_cal"].to_numpy().astype(float)),
        }
        fig_cal_c, brier_c = plot_calibration_curves_desc(
            multi_cal,
            title="Calibration Curves (Multi-modal ONLY; Calibrated; Intersection OOF)",
            n_bins=args.calib_bins,
        )
        out_png_c = os.path.join(args.out_dir, "calibration_multi_only_calibrated.png")
        savefig_all(fig_cal_c, out_png_c, dpi=args.dpi, also_pdf=args.also_pdf)
        plt.close(fig_cal_c)
        print(f"Saved: {out_png_c}")

        brier_c_path = os.path.join(args.out_dir, "brier_scores_multi_only_calibrated.csv")
        brier_c.to_csv(brier_c_path, index=False)
        print(f"Saved: {brier_c_path}")
        print("\nBrier (multi only, calibrated; lower is better):")
        print(brier_c.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    else:
        print(
            "\nSkipping calibration_multi_only_calibrated.png "
            "(missing proba_multi_late_cal / proba_multi_early_cal in oof_predictions.csv)."
        )

    # -------------------------
    # 6) SHAP combined figure (OPTIONAL)
    # -------------------------
    if os.path.exists(shap_early) and os.path.exists(shap_late):
        plot_shap_early_and_late(
            shap_early_path=shap_early,
            shap_late_path=shap_late,
            out_path_png=os.path.join(args.out_dir, "shap_early_late_combined.png"),
            dpi=args.dpi,
            also_pdf=args.also_pdf,
            top_k=args.top_k,
        )
        print(f"Saved: {os.path.join(args.out_dir, 'shap_early_late_combined.png')}")
    else:
        missing = []
        if not os.path.exists(shap_early):
            missing.append(shap_early)
        if not os.path.exists(shap_late):
            missing.append(shap_late)
        print("Skipping shap_early_late_combined.png (missing SHAP files):")
        for m in missing:
            print(f"  - {m}")

    if args.also_pdf:
        print("Also saved PDF copies (where plots were generated).")


if __name__ == "__main__":
    main()

