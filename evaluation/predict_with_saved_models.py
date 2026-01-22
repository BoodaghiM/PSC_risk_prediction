#!/usr/bin/env python3
"""
predict_with_saved_models.py

Robust scoring for PSC risk using saved single-modality + multi-modality models.

Key robustness updates:
- Aligns prediction input columns to what the saved model expects (feature_names_in_)
  to prevent sklearn "feature names should match those that were passed during fit".
- Adds missing expected columns as NaN and orders columns as expected.
- Works best when saved models are sklearn Pipelines (ColumnTransformer + OneHotEncoder(handle_unknown="ignore")).
"""

from __future__ import annotations

import argparse
import os
import json
from typing import Dict, List

import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import roc_curve
from sklearn.base import BaseEstimator

# ----------------------------
# Constants / column config
# ----------------------------

MODALITY_FILES = {
    "lab": "data_lab_imputed.csv",
    "serology": "data_serology_imputed_MIRIAD.csv",
    "genetics": "data_genetics_only_significant_snps_imputed.csv",
    "clinical": "data_clinical_imputed.csv",
}

SEROL_COLS = ["ANCA EU", "CBir1 EU", "OmpC EU", "IgA ASCA EU", "IgG ASCA EU"]
CLIN_COLS = [
    "Gender_Self", "Jewish_Self", "Diagnosis", "AgeDx", "Smoking", "FamilyHistory",
    "extent_UC", "Surgery", "upper_GI", "disease_location_CD", "behavior_CD", "Perianal"
]

FID_COL = "FID"
LABEL_COL = "PSC"

MULTI_FILENAMES = {
    "early": "multi_early_rf.joblib",
    "late_meta": "multi_late_meta_lr.joblib",
    "late_base_lab": "multi_late_base_lab.joblib",
    "late_base_serology": "multi_late_base_serology.joblib",
    "late_base_clinical": "multi_late_base_clinical.joblib",
    "late_base_genetics": "multi_late_base_genetics.joblib",
}

# ----------------------------
# Utilities
# ----------------------------

def normalize_fid(fid) -> str:
    fid = str(fid).strip()
    return str(int(fid)) if fid.isdigit() else fid


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_csv_with_fid(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if FID_COL not in df.columns:
        raise ValueError(f"Missing '{FID_COL}' in {path}")
    df[FID_COL] = df[FID_COL].apply(normalize_fid)
    return df


def load_pheno(pheno_path: str) -> pd.DataFrame:
    ph = pd.read_csv(pheno_path)
    if FID_COL not in ph.columns:
        raise ValueError(f"Phenotype file missing '{FID_COL}': {pheno_path}")
    ph[FID_COL] = ph[FID_COL].apply(normalize_fid)

    # Keep training-consistent filters:
    if "Race_Admix" in ph.columns:
        ph = ph[ph["Race_Admix"].isin(["White"])].copy()
    if "Diagnosis" in ph.columns:
        ph = ph[ph["Diagnosis"].isin(["CD", "UC"])].copy()

    keep_cols = [FID_COL]
    if LABEL_COL in ph.columns:
        keep_cols.append(LABEL_COL)

    for c in ["Race_Admix", "Race_Self", "Diagnosis"]:
        if c in ph.columns and c not in keep_cols:
            keep_cols.append(c)

    return ph[keep_cols].copy()


def is_undiagnosed_row(ph_row: pd.Series) -> bool:
    if LABEL_COL not in ph_row.index:
        return True
    v = ph_row[LABEL_COL]
    if pd.isna(v):
        return True
    try:
        iv = int(v)
        return iv not in (0, 1)
    except Exception:
        return True


def pick_feature_columns_for_modality(df: pd.DataFrame, modality: str) -> List[str]:
    if modality == "serology":
        return [c for c in SEROL_COLS if c in df.columns]
    if modality == "clinical":
        return [c for c in CLIN_COLS if c in df.columns]
    if modality == "genetics":
        return [c for c in df.columns if c != FID_COL and ":" in c]
    if modality == "lab":
        return [c for c in df.columns if c != FID_COL]
    raise ValueError(modality)


def load_best_model_from_dir(model_dir: str) -> BaseEstimator:
    path = os.path.join(model_dir, "best_model.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected best_model.joblib in: {model_dir}")
    return joblib.load(path)


def load_multi_models(multi_models_dir: str) -> Dict[str, BaseEstimator]:
    missing = []
    for k, fn in MULTI_FILENAMES.items():
        p = os.path.join(multi_models_dir, fn)
        if not os.path.exists(p):
            missing.append(p)
    if missing:
        msg = "Multi-models directory is missing required files:\n" + "\n".join(missing)
        raise FileNotFoundError(msg)

    return {k: joblib.load(os.path.join(multi_models_dir, fn)) for k, fn in MULTI_FILENAMES.items()}


def get_expected_feature_names(model: BaseEstimator) -> List[str]:
    """
    Many sklearn estimators/pipelines store feature_names_in_ after fit.
    If present, we can align prediction data to exactly those columns.
    """
    exp = getattr(model, "feature_names_in_", None)
    if exp is None:
        return []
    return [str(x) for x in list(exp)]


def align_X_to_model(X: pd.DataFrame, model: BaseEstimator) -> pd.DataFrame:
    """
    Make X compatible with what the model was fit on.
    - If model exposes feature_names_in_, ensure those columns exist and are ordered.
    - Add missing columns as NaN.
    - Drop extra columns (to avoid "unseen at fit time" errors).
    """
    expected = get_expected_feature_names(model)
    if not expected:
        # Best effort: pass as-is (works for many Pipelines w/ ColumnTransformer selecting columns)
        return X

    X2 = X.copy()
    missing = [c for c in expected if c not in X2.columns]
    for c in missing:
        X2[c] = np.nan

    X2 = X2[expected]
    return X2


def predict_proba_safe(model: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    X_aligned = align_X_to_model(X, model)
    p = model.predict_proba(X_aligned)[:, 1]
    return np.asarray(p, dtype=float)


def optimal_threshold_youden(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thresholds[idx])


def maybe_compute_thresholds(out_df: pd.DataFrame, prob_cols: List[str]) -> Dict[str, float]:
    if LABEL_COL not in out_df.columns:
        return {}
    y = pd.to_numeric(out_df[LABEL_COL], errors="coerce")
    mask = y.isin([0, 1]) & out_df[prob_cols].notna().all(axis=1)
    if int(mask.sum()) < 10:
        return {}

    y_true = y[mask].astype(int).to_numpy()
    thresholds = {}
    for c in prob_cols:
        p = pd.to_numeric(out_df.loc[mask, c], errors="coerce").to_numpy()
        thresholds[c] = float(optimal_threshold_youden(y_true, p))
    return thresholds


# ----------------------------
# Main scoring
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--pheno-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--undiagnosed-only", action="store_true")

    ap.add_argument("--lab-model-dir", required=True)
    ap.add_argument("--serology-model-dir", required=True)
    ap.add_argument("--clinical-model-dir", required=True)
    ap.add_argument("--genetics-model-dir", required=True)

    ap.add_argument("--multi-models-dir", required=True)
    ap.add_argument("--compute-thresholds", action="store_true")

    args = ap.parse_args()
    ensure_parent_dir(args.out)

    # ---- Phenotype cohort ----
    ph = load_pheno(args.pheno_path)
    if args.undiagnosed_only:
        ph = ph[ph.apply(is_undiagnosed_row, axis=1)].copy()
    if len(ph) == 0:
        raise RuntimeError("No rows left after phenotype filtering (and --undiagnosed-only).")

    # ---- Load modality tables ----
    dfs_mod = {}
    mod_cols = {}
    for m in ["lab", "serology", "clinical", "genetics"]:
        fp = os.path.join(args.input_dir, MODALITY_FILES[m])
        dfm = load_csv_with_fid(fp)
        cols = pick_feature_columns_for_modality(dfm, m)
        if len(cols) == 0:
            raise RuntimeError(f"No usable feature columns found for modality '{m}' in {fp}")
        dfs_mod[m] = dfm[[FID_COL] + cols].copy()
        mod_cols[m] = cols

    # Intersection merge for prediction
    df = ph[[FID_COL] + ([LABEL_COL] if LABEL_COL in ph.columns else [])].copy()
    for m in ["lab", "serology", "clinical", "genetics"]:
        df = df.merge(dfs_mod[m], how="inner", on=FID_COL)

    if len(df) == 0:
        raise RuntimeError("After merging phenotype + all modalities, got 0 rows (no intersection cohort).")

    # ---- Load models ----
    single_models = {
        "lab": load_best_model_from_dir(args.lab_model_dir),
        "serology": load_best_model_from_dir(args.serology_model_dir),
        "clinical": load_best_model_from_dir(args.clinical_model_dir),
        "genetics": load_best_model_from_dir(args.genetics_model_dir),
    }
    multi = load_multi_models(args.multi_models_dir)

    # ---- Output frame ----
    out = df[[FID_COL] + ([LABEL_COL] if LABEL_COL in df.columns else [])].copy()

    # ---- Single-modality predictions ----
    for m in ["lab", "serology", "clinical", "genetics"]:
        X = df[mod_cols[m]]
        out[f"risk_single_{m}"] = predict_proba_safe(single_models[m], X)

    # ---- Multi-early prediction ----
    # Prefer the model's expected features if it exposes feature_names_in_
    early_expected = get_expected_feature_names(multi["early"])
    if early_expected:
        X_early = df.reindex(columns=early_expected, fill_value=np.nan)
    else:
        # fallback: concatenate raw columns across modalities
        early_cols = [c for mm in ["lab", "serology", "clinical", "genetics"] for c in mod_cols[mm]]
        X_early = df[early_cols]
    out["risk_multi_early"] = predict_proba_safe(multi["early"], X_early)

    # ---- Multi-late prediction ----
    p_base_lab = predict_proba_safe(multi["late_base_lab"], df[mod_cols["lab"]])
    p_base_ser = predict_proba_safe(multi["late_base_serology"], df[mod_cols["serology"]])
    p_base_cli = predict_proba_safe(multi["late_base_clinical"], df[mod_cols["clinical"]])
    p_base_gen = predict_proba_safe(multi["late_base_genetics"], df[mod_cols["genetics"]])

    out["risk_base_lab"] = p_base_lab
    out["risk_base_serology"] = p_base_ser
    out["risk_base_clinical"] = p_base_cli
    out["risk_base_genetics"] = p_base_gen

    X_meta = pd.DataFrame(
        {
            "p_lab": p_base_lab,
            "p_serology": p_base_ser,
            "p_clinical": p_base_cli,
            "p_genetics": p_base_gen,
        }
    )
    out["risk_multi_late"] = predict_proba_safe(multi["late_meta"], X_meta)

    # ---- Optional thresholds ----
    thresholds = {}
    if args.compute_thresholds:
        prob_cols = [
            "risk_single_lab",
            "risk_single_serology",
            "risk_single_clinical",
            "risk_single_genetics",
            "risk_multi_early",
            "risk_multi_late",
        ]
        thresholds = maybe_compute_thresholds(out, prob_cols)
        for c, thr in thresholds.items():
            out[c.replace("risk_", "call_") + f"_thr_{thr:.3f}"] = np.where(out[c] >= thr, "Yes", "No")

    # ---- Save ----
    out.to_csv(args.out, index=False)

    if args.compute_thresholds and thresholds:
        thr_path = os.path.splitext(args.out)[0] + "_thresholds.json"
        with open(thr_path, "w") as f:
            json.dump(thresholds, f, indent=2)

    print(f"Wrote: {args.out}")
    if args.compute_thresholds:
        print("Saved thresholds JSON next to output." if thresholds else "Thresholds not computed (no valid labels).")


if __name__ == "__main__":
    main()

