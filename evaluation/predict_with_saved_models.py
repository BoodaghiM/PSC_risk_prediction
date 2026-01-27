#!/usr/bin/env python3
"""
predict_with_saved_models.py

MULTI-ONLY scorer that reproduces your original output schema:

Columns:
FID, PSC,
risk_multi_early_raw, risk_multi_late_raw,
risk_multi_early, risk_multi_late,   (calibrated versions; equals raw if no calibrators)
binary_multi_early_*,
binary_multi_early_cal_*,
binary_multi_late_*,
binary_multi_late_cal_*

Key fixes vs your "gigantic" output:
- NEVER writes risk_base_* columns
- Binaries are applied ONLY to their matching model outputs (no cross-application)
- Filters thresholds CSV to multi_* rows only
- Avoids pandas fragmentation by building all binary cols in a dict then concat once
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator

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

CAL_FILES = {
    "early": "calibrator_multi_early.joblib",
    "late": "calibrator_multi_late.joblib",
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

    # training-consistent filters
    ph = ph[ph["Race_Self"].isin(["White"])].copy()
    if "Diagnosis" in ph.columns:
        ph = ph[ph["Diagnosis"].isin(["CD", "UC"])].copy()

    keep_cols = [FID_COL]
    if LABEL_COL in ph.columns:
        keep_cols.append(LABEL_COL)
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

def load_multi_models(multi_models_dir: str) -> Dict[str, BaseEstimator]:
    missing = []
    for k, fn in MULTI_FILENAMES.items():
        p = os.path.join(multi_models_dir, fn)
        if not os.path.exists(p):
            missing.append(p)
    if missing:
        raise FileNotFoundError("Missing required multi-model files:\n" + "\n".join(missing))
    return {k: joblib.load(os.path.join(multi_models_dir, fn)) for k, fn in MULTI_FILENAMES.items()}

def get_expected_feature_names(model: BaseEstimator) -> List[str]:
    exp = getattr(model, "feature_names_in_", None)
    if exp is None:
        return []
    return [str(x) for x in list(exp)]

def align_X_to_model(X: pd.DataFrame, model: BaseEstimator) -> pd.DataFrame:
    expected = get_expected_feature_names(model)
    if not expected:
        return X
    X2 = X.copy()
    for c in expected:
        if c not in X2.columns:
            X2[c] = np.nan
    return X2[expected]

def predict_proba_safe(model: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    X_aligned = align_X_to_model(X, model)
    p = model.predict_proba(X_aligned)[:, 1]
    return np.asarray(p, dtype=float)

def clip_probs(p: np.ndarray, eps: float) -> np.ndarray:
    eps = float(eps)
    if eps <= 0:
        return p
    return np.clip(p, eps, 1.0 - eps)

def _wants_2d(obj) -> bool:
    nfi = getattr(obj, "n_features_in_", None)
    return isinstance(nfi, (int, np.integer)) and int(nfi) == 1

def apply_calibrator(cal_obj, p_raw: np.ndarray) -> np.ndarray:
    p_raw = np.asarray(p_raw, dtype=float)
    if hasattr(cal_obj, "predict_proba"):
        out = cal_obj.predict_proba(p_raw.reshape(-1, 1))
        if out.ndim == 2 and out.shape[1] >= 2:
            return np.asarray(out[:, 1], dtype=float)
        return np.asarray(out, dtype=float).ravel()
    if hasattr(cal_obj, "predict"):
        out = cal_obj.predict(p_raw.reshape(-1, 1)) if _wants_2d(cal_obj) else cal_obj.predict(p_raw)
        return np.asarray(out, dtype=float).ravel()
    if callable(cal_obj):
        return np.asarray(cal_obj(p_raw), dtype=float).ravel()
    raise TypeError("Unsupported calibrator object.")

# ----------------------------
# Thresholds -> ORIGINAL binary column names
# ----------------------------

def load_thresholds_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def _col_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    return next((c for c in candidates if c in df.columns), None)

def build_original_binaries(out: pd.DataFrame, thr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces columns like:
      binary_multi_early_f1_opt_thr_0.38
      binary_multi_early_cal_recall_0.80_thr_0.05
      binary_multi_late_youden_j_opt_thr_0.41
      ...
    Only applies thresholds to their matching prob column:
      multi_early      -> risk_multi_early_raw
      multi_early_cal  -> risk_multi_early
      multi_late       -> risk_multi_late_raw
      multi_late_cal   -> risk_multi_late
    """
    model_col = _col_present(thr_df, ["model", "model_name", "estimator", "clf"])
    thr_col   = _col_present(thr_df, ["threshold", "thr", "best_threshold", "value"])
    name_col  = _col_present(thr_df, ["threshold_name", "thr_name", "name", "rule", "method", "selector", "type"])

    if model_col is None or thr_col is None:
        # Cannot safely parse => no binaries
        return pd.DataFrame(index=out.index)

    # keep only multi rows you care about
    keep_models = {"multi_early", "multi_early_cal", "multi_late", "multi_late_cal"}
    thr_df = thr_df[thr_df[model_col].astype(str).isin(keep_models)].copy()
    if thr_df.empty:
        return pd.DataFrame(index=out.index)

    # map model -> probability column
    model_to_prob = {
        "multi_early": "risk_multi_early_raw",
        "multi_early_cal": "risk_multi_early",
        "multi_late": "risk_multi_late_raw",
        "multi_late_cal": "risk_multi_late",
    }

    probs = {pc: pd.to_numeric(out[pc], errors="coerce").to_numpy() for pc in set(model_to_prob.values()) if pc in out.columns}

    new_cols: Dict[str, np.ndarray] = {}
    for _, r in thr_df.iterrows():
        m = str(r[model_col]).strip()
        pc = model_to_prob.get(m)
        if pc is None or pc not in probs:
            continue

        thr_val = r[thr_col]
        if pd.isna(thr_val):
            continue
        try:
            thr_val = float(thr_val)
        except Exception:
            continue

        tname = str(r[name_col]).strip() if (name_col is not None and pd.notna(r[name_col])) else "thr"

        # match your original formatting: thr_0.38 etc (2 decimals)
        thr_s = f"{thr_val:.2f}"

        col = f"binary_{m}_{tname}_thr_{thr_s}"
        new_cols[col] = np.where(probs[pc] >= thr_val, "Yes", "No")

    return pd.DataFrame(new_cols, index=out.index)

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--pheno-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--undiagnosed-only", action="store_true")
    ap.add_argument("--multi-models-dir", required=True)

    ap.add_argument("--calibrate-multi", action="store_true")
    ap.add_argument("--calibrators-dir")
    ap.add_argument("--calibration-clip", default="0")

    ap.add_argument("--emit-multi-binaries", action="store_true")
    ap.add_argument("--thresholds-csv")

    args = ap.parse_args()
    ensure_parent_dir(args.out)

    ph = load_pheno(args.pheno_path)
    if args.undiagnosed_only:
        ph = ph[ph.apply(is_undiagnosed_row, axis=1)].copy()
    if ph.empty:
        raise RuntimeError("No rows left after phenotype filtering (and --undiagnosed-only).")

    # Load modalities
    dfs_mod: Dict[str, pd.DataFrame] = {}
    mod_cols: Dict[str, List[str]] = {}
    for m in ["lab", "serology", "clinical", "genetics"]:
        fp = os.path.join(args.input_dir, MODALITY_FILES[m])
        dfm = load_csv_with_fid(fp)
        cols = pick_feature_columns_for_modality(dfm, m)
        if not cols:
            raise RuntimeError(f"No usable feature columns for modality '{m}' in {fp}")
        dfs_mod[m] = dfm[[FID_COL] + cols].copy()
        mod_cols[m] = cols

    # Intersection merge
    df = ph[[FID_COL] + ([LABEL_COL] if LABEL_COL in ph.columns else [])].copy()
    for m in ["lab", "serology", "clinical", "genetics"]:
        df = df.merge(dfs_mod[m], how="inner", on=FID_COL)
    if df.empty:
        raise RuntimeError("No intersection cohort after merging phenotype + all modalities.")

    multi = load_multi_models(args.multi_models_dir)

    # Output base
    out = df[[FID_COL] + ([LABEL_COL] if LABEL_COL in df.columns else [])].copy()

    # Multi early raw
    early_expected = get_expected_feature_names(multi["early"])
    if early_expected:
        X_early = df.reindex(columns=early_expected, fill_value=np.nan)
    else:
        early_cols = [c for mm in ["lab", "serology", "clinical", "genetics"] for c in mod_cols[mm]]
        X_early = df[early_cols]
    out["risk_multi_early_raw"] = predict_proba_safe(multi["early"], X_early)

    # Multi late raw (base probs computed internally ONLY; not written)
    p_base_lab = predict_proba_safe(multi["late_base_lab"], df[mod_cols["lab"]])
    p_base_ser = predict_proba_safe(multi["late_base_serology"], df[mod_cols["serology"]])
    p_base_cli = predict_proba_safe(multi["late_base_clinical"], df[mod_cols["clinical"]])
    p_base_gen = predict_proba_safe(multi["late_base_genetics"], df[mod_cols["genetics"]])

    X_meta = pd.DataFrame({"p_lab": p_base_lab, "p_serology": p_base_ser, "p_clinical": p_base_cli, "p_genetics": p_base_gen})
    out["risk_multi_late_raw"] = predict_proba_safe(multi["late_meta"], X_meta)

    # Calibrated (use original names: risk_multi_early / risk_multi_late)
    out["risk_multi_early"] = out["risk_multi_early_raw"]
    out["risk_multi_late"] = out["risk_multi_late_raw"]

    if args.calibrate_multi:
        if not args.calibrators_dir:
            raise ValueError("--calibrate-multi requires --calibrators-dir")
        p1 = os.path.join(args.calibrators_dir, CAL_FILES["early"])
        p2 = os.path.join(args.calibrators_dir, CAL_FILES["late"])
        if os.path.exists(p1) and os.path.exists(p2):
            cal_early = joblib.load(p1)
            cal_late = joblib.load(p2)
            eps = float(args.calibration_clip) if args.calibration_clip is not None else 0.0
            pe = clip_probs(out["risk_multi_early_raw"].to_numpy(dtype=float), eps)
            pl = clip_probs(out["risk_multi_late_raw"].to_numpy(dtype=float), eps)
            out["risk_multi_early"] = apply_calibrator(cal_early, pe)
            out["risk_multi_late"] = apply_calibrator(cal_late, pl)
        else:
            print("WARNING: calibrators not found; calibrated scores will equal raw.")

    # Binaries in your original style (no fragmentation)
    if args.emit_multi_binaries:
        if not args.thresholds_csv:
            raise ValueError("--emit-multi-binaries requires --thresholds-csv")
        thr_df = load_thresholds_csv(args.thresholds_csv)
        bin_df = build_original_binaries(out, thr_df)
        if not bin_df.empty:
            out = pd.concat([out, bin_df], axis=1)
    out['FID'] = out['FID'].astype(str).str.zfill(7)
    out['FID'] = out['FID'].str[:2] + '-' + out['FID'].str[3:]
    out.to_csv(args.out, index=False)
    print(f"Wrote: {args.out}")
    print(f"Rows: {len(out):,}  Cols: {len(out.columns):,}")

if __name__ == "__main__":
    main()

