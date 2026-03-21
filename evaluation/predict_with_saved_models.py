#!/usr/bin/env python3
"""
predict_with_saved_models.py

MULTI-ONLY scorer that reproduces your original output schema.

Main scored output columns (order):
FID, PSC, Cohort,
[EXTRA clinical columns],
risk_multi_early_raw, risk_multi_late_raw,
risk_multi_early, risk_multi_late,
binary_multi_early_*,
binary_multi_early_cal_*,
binary_multi_late_*,
binary_multi_late_cal_*

Feb 2026 updates:
1) Adds export-only clinical columns to the scored output (from clinical modality file).
2) Adds Cohort to scored output (from phenotype file).
3) Modality availability export:
   - Saved BEFORE Race filter (full phenotype, optionally restricted by --undiagnosed-only)
   - Includes Cohort + Race_Self + Race_Admix
   - Includes NO model prediction columns
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator

# ----------------------------
# Inputs / column definitions
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

# Export-only columns to add to scored output (pulled from clinical modality table)
EXTRA_EVAL_COLS = CLIN_COLS.copy()

FID_COL = "FID"
LABEL_COL = "PSC"
COHORT_COL = "Cohort"

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

def format_fid_display(fid_series: pd.Series) -> pd.Series:
    """
    Matches your current output formatting:
      - zfill(7)
      - then: xx-xxxx (based on your slicing)
    NOTE: This is only for OUTPUT files, never for merge keys.
    """
    s = fid_series.astype(str).str.zfill(7)
    s = s.str[:2] + "-" + s.str[3:]
    return s

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
    ph = ph[ph["Race_Admix"].isin(["White"])].copy()
    if "Diagnosis" in ph.columns:
        ph = ph[ph["Diagnosis"].isin(["CD", "UC"])].copy()

    keep_cols = [FID_COL]
    if LABEL_COL in ph.columns:
        keep_cols.append(LABEL_COL)
    return ph[keep_cols].copy()

def load_pheno_no_filter(pheno_path: str) -> pd.DataFrame:
    """
    Loads phenotype WITHOUT race/diagnosis filters.
    Used for modality availability export (pre-race-filter).
    """
    ph = pd.read_csv(pheno_path)
    if FID_COL not in ph.columns:
        raise ValueError(f"Phenotype file missing '{FID_COL}': {pheno_path}")
    ph[FID_COL] = ph[FID_COL].apply(normalize_fid)
    return ph.copy()

def load_pheno_columns(pheno_path: str, columns: List[str]) -> pd.DataFrame:
    """
    Loads extra phenotype columns for EXPORT ONLY (no filters).
    Normalizes FID and drops duplicate FIDs.
    """
    ph = pd.read_csv(pheno_path)
    if FID_COL not in ph.columns:
        raise ValueError(f"Phenotype file missing '{FID_COL}': {pheno_path}")
    ph[FID_COL] = ph[FID_COL].apply(normalize_fid)

    keep = [FID_COL] + [c for c in columns if c in ph.columns]
    ph = ph[keep].copy()
    ph = ph.drop_duplicates(subset=[FID_COL], keep="first")
    return ph

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
    model_col = _col_present(thr_df, ["model", "model_name", "estimator", "clf"])
    thr_col   = _col_present(thr_df, ["threshold", "thr", "best_threshold", "value"])
    name_col  = _col_present(thr_df, ["threshold_name", "thr_name", "name", "rule", "method", "selector", "type"])

    if model_col is None or thr_col is None:
        return pd.DataFrame(index=out.index)

    keep_models = {"multi_early", "multi_early_cal", "multi_late", "multi_late_cal"}
    thr_df = thr_df[thr_df[model_col].astype(str).isin(keep_models)].copy()
    if thr_df.empty:
        return pd.DataFrame(index=out.index)

    model_to_prob = {
        "multi_early": "risk_multi_early_raw",
        "multi_early_cal": "risk_multi_early",
        "multi_late": "risk_multi_late_raw",
        "multi_late_cal": "risk_multi_late",
    }

    probs = {
        pc: pd.to_numeric(out[pc], errors="coerce").to_numpy()
        for pc in set(model_to_prob.values()) if pc in out.columns
    }

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
        thr_s = f"{thr_val:.2f}"
        col = f"binary_{m}_{tname}_thr_{thr_s}"
        new_cols[col] = np.where(probs[pc] >= thr_val, "Yes", "No")

    return pd.DataFrame(new_cols, index=out.index)

# ----------------------------
# Modality availability export
# ----------------------------

def build_modality_availability(
    ph: pd.DataFrame,
    dfs_mod: Dict[str, pd.DataFrame],
    scored_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Availability is defined as: FID exists in that modality dataframe.
    scored_df is the final merged intersection cohort (the one you score).
    """
    avail = ph[[FID_COL] + ([LABEL_COL] if LABEL_COL in ph.columns else [])].copy()

    for m, dfm in dfs_mod.items():
        have = set(dfm[FID_COL].astype(str))
        avail[f"has_{m}"] = avail[FID_COL].astype(str).isin(have).astype(int)

    has_cols = [f"has_{m}" for m in dfs_mod.keys()]
    avail["n_modalities"] = avail[has_cols].sum(axis=1)

    scored_set = set(scored_df[FID_COL].astype(str)) if (scored_df is not None and not scored_df.empty) else set()
    avail["in_intersection"] = avail[FID_COL].astype(str).isin(scored_set).astype(int)

    return avail

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

    ap.add_argument("--write-modality-availability", action="store_true",
                    help="Write a per-patient modality availability table (pre-race-filter; includes Cohort + Race_Self + Race_Admix; no prediction cols).")
    ap.add_argument("--modality-availability-out",
                    help="Output path for modality availability CSV. If omitted, derives from --out.")

    args = ap.parse_args()
    ensure_parent_dir(args.out)

    # ---------------------------------------------------------
    # Load filtered phenotype for SCORING
    # ---------------------------------------------------------
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

    # Intersection merge (your scoring cohort)
    df = ph[[FID_COL] + ([LABEL_COL] if LABEL_COL in ph.columns else [])].copy()
    for m in ["lab", "serology", "clinical", "genetics"]:
        df = df.merge(dfs_mod[m], how="inner", on=FID_COL)
    if df.empty:
        raise RuntimeError("No intersection cohort after merging phenotype + all modalities.")

    multi = load_multi_models(args.multi_models_dir)

    # Output base (scored intersection only)
    out = df[[FID_COL] + ([LABEL_COL] if LABEL_COL in df.columns else [])].copy()

    # ----------------------------
    # Multi early raw
    # ----------------------------
    early_expected = get_expected_feature_names(multi["early"])
    if early_expected:
        X_early = df.reindex(columns=early_expected, fill_value=np.nan)
    else:
        early_cols = [c for mm in ["lab", "serology", "clinical", "genetics"] for c in mod_cols[mm]]
        X_early = df[early_cols]
    out["risk_multi_early_raw"] = predict_proba_safe(multi["early"], X_early)

    # ----------------------------
    # Multi late raw (base probs computed internally ONLY; not written)
    # ----------------------------
    p_base_lab = predict_proba_safe(multi["late_base_lab"], df[mod_cols["lab"]])
    p_base_ser = predict_proba_safe(multi["late_base_serology"], df[mod_cols["serology"]])
    p_base_cli = predict_proba_safe(multi["late_base_clinical"], df[mod_cols["clinical"]])
    p_base_gen = predict_proba_safe(multi["late_base_genetics"], df[mod_cols["genetics"]])

    X_meta = pd.DataFrame({
        "p_lab": p_base_lab,
        "p_serology": p_base_ser,
        "p_clinical": p_base_cli,
        "p_genetics": p_base_gen
    })
    out["risk_multi_late_raw"] = predict_proba_safe(multi["late_meta"], X_meta)

    # ----------------------------
    # Calibrated (use original names: risk_multi_early / risk_multi_late)
    # ----------------------------
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

    # ----------------------------
    # Binaries
    # ----------------------------
    if args.emit_multi_binaries:
        if not args.thresholds_csv:
            raise ValueError("--emit-multi-binaries requires --thresholds-csv")
        thr_df = load_thresholds_csv(args.thresholds_csv)
        bin_df = build_original_binaries(out, thr_df)
        if not bin_df.empty:
            out = pd.concat([out, bin_df], axis=1)

    # ---------------------------------------------------------
    # Add Cohort column from phenotype file (export-only)
    # ---------------------------------------------------------
    ph_cohort = load_pheno_columns(args.pheno_path, [COHORT_COL])
    if COHORT_COL in ph_cohort.columns:
        out = out.merge(ph_cohort, how="left", on=FID_COL)

    # ---------------------------------------------------------
    # Add extra evaluation columns from CLINICAL modality (export-only)
    # ---------------------------------------------------------
    clin_export_cols = [c for c in EXTRA_EVAL_COLS if c in dfs_mod["clinical"].columns]
    if not clin_export_cols:
        print("WARNING: No EXTRA_EVAL_COLS found in clinical file; skipping export extras.")
    else:
        clin_extra = dfs_mod["clinical"][[FID_COL] + clin_export_cols].copy()
        clin_extra = clin_extra.drop_duplicates(subset=[FID_COL], keep="first")
        out = out.merge(clin_extra, how="left", on=FID_COL)

    # ---------------------------------------------------------
    # Reorder scored output columns: FID, PSC, Cohort, extras, then rest
    # ---------------------------------------------------------
    base_cols = [FID_COL] + ([LABEL_COL] if LABEL_COL in out.columns else [])
    if COHORT_COL in out.columns:
        base_cols.append(COHORT_COL)

    extras = [c for c in EXTRA_EVAL_COLS if c in out.columns]
    rest = [c for c in out.columns if c not in base_cols + extras]
    out = out[base_cols + extras + rest]

    # ---------------------------------------------------------
    # Write modality availability (pre-race-filter; include Race_Self + Race_Admix; NO prediction cols)
    # ---------------------------------------------------------
    if args.write_modality_availability:
        mod_out = args.modality_availability_out
        if not mod_out:
            root, _ext = os.path.splitext(args.out)
            mod_out = f"{root}__modality_availability.csv"
        ensure_parent_dir(mod_out)

        # Load phenotype WITHOUT race/diagnosis filters for this export
        ph_raw = load_pheno_no_filter(args.pheno_path)
        if args.undiagnosed_only:
            ph_raw = ph_raw[ph_raw.apply(is_undiagnosed_row, axis=1)].copy()

        # Build availability table from full phenotype
        avail_df = build_modality_availability(ph_raw, dfs_mod, df)

        # Add Cohort + Race columns (export-only)
        ph_extra = load_pheno_columns(args.pheno_path, [COHORT_COL, "Race_Self", "Race_Admix"])
        avail_df = avail_df.merge(ph_extra, how="left", on=FID_COL)

        # Keep ONLY desired columns
        desired_cols = [
            FID_COL,
            LABEL_COL if LABEL_COL in avail_df.columns else None,
            COHORT_COL if COHORT_COL in avail_df.columns else None,
            "Race_Self" if "Race_Self" in avail_df.columns else None,
            "Race_Admix" if "Race_Admix" in avail_df.columns else None,
            "has_lab",
            "has_serology",
            "has_clinical",
            "has_genetics",
            "n_modalities",
        ]
        desired_cols = [c for c in desired_cols if c is not None and c in avail_df.columns]
        avail_df = avail_df[desired_cols]

        # Format FID for display
        avail_df[FID_COL] = format_fid_display(avail_df[FID_COL])

        avail_df.to_csv(mod_out, index=False)
        print(f"Wrote modality availability (pre-race-filter; no prediction cols): {mod_out}")
        print(f"Rows: {len(avail_df):,}  Cols: {len(avail_df.columns):,}")

    # Display formatting only (do not affect merges)
    out[FID_COL] = format_fid_display(out[FID_COL])

    out.to_csv(args.out, index=False)
    print(f"Wrote: {args.out}")
    print(f"Rows: {len(out):,}  Cols: {len(out.columns):,}")

if __name__ == "__main__":
    main()

