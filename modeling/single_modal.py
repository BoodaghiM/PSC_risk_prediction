#!/usr/bin/env python3
"""
single_modal.py

Single-modality Random Forest modeling with nested cross-validation.

Key behavior:
- Inner CV hyperparameter optimization: AUPRC (Average Precision / AP)
- Outer CV: produces out-of-fold (OOF) probabilities
- Threshold selection is performed ONCE using OOF predictions (out-of-sample)
- NO extra subfolders are created
- Metrics for multiple recall targets are saved as additional rows in the SAME files.

Optional calibration (leakage-safe):
- For each OUTER fold, probability calibration (sigmoid/isotonic) is fit using ONLY outer-train data
  (via StratifiedKFold inside CalibratedClassifierCV) and applied to outer-test predictions.
- Calibrated OOF stream is saved alongside uncalibrated.
- Thresholding/metrics are computed separately for uncalibrated vs calibrated streams.

Outputs (per modality out_dir):
  Core artifacts:
  - best_model.joblib
  - training_columns.json
  - nestedcv_oof_predictions.csv                    (y_true, y_oof_prob, y_oof_prob_cal + thresholded preds)
  - fold_ranking_metrics.csv                        (per-fold AUC/AUPRC; includes calibrated if enabled)
  - oof_ranking_metrics.csv                         (OOF AUC/AUPRC; includes calibrated if enabled)

  Curves (OOF):
  - nestedcv_roc_curve.csv
  - nestedcv_pr_curve.csv
  - nestedcv_roc_curve_cal.csv                      (if enabled)
  - nestedcv_pr_curve_cal.csv                       (if enabled)

  Threshold sweeps + metrics (OOF):
  - threshold_sweep_pr.csv
  - threshold_sweep_roc.csv
  - threshold_sweep_recall.csv
  - threshold_metrics_oof.csv
  - threshold_metrics_by_fold.csv

  Calibrated versions (if enabled):
  - threshold_sweep_pr_cal.csv
  - threshold_sweep_roc_cal.csv
  - threshold_sweep_recall_cal.csv
  - threshold_metrics_oof_cal.csv
  - threshold_metrics_by_fold_cal.csv
  - best_model_calibrator.joblib                    (calibrator fit on full data via CV; for deployment)

  SHAP:
  - shap_global_importance.csv                       (if SHAP available and enabled)

  Summary:
  - nestedcv_summary.json                            (all metrics + thresholds + calibration info)

Example:
  python modeling/single_modal.py \
    --modality lab \
    --input-dir /path/to/imputed_data \
    --pheno-path /path/to/data_phenotype_original.csv \
    --out-dir /path/to/out \
    --target-recall 0.90 0.85 0.80 \
    --calibration-method sigmoid

Notes on reproducibility:
- Sets PYTHONHASHSEED=0 and BLAS thread env (OMP/OPENBLAS/MKL/...) to reduce oversubscription.
- Full bitwise determinism is not guaranteed across machines/threads, but should be stable.
"""

from __future__ import annotations

import os
import json
import argparse
from collections import Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib

from scipy.stats import randint

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# Optional SHAP
try:
    import shap  # type: ignore
except Exception:
    shap = None

# Reproducibility baseline
os.environ["PYTHONHASHSEED"] = "0"


# =========================
# Utilities
# =========================

FID_COL = "FID"
LABEL_COL = "PSC"


def set_thread_env_for_hpc() -> None:
    # Prevent BLAS oversubscription when sklearn parallelizes.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def get_n_jobs(n_jobs_arg: Optional[int] = None) -> int:
    if n_jobs_arg is not None and n_jobs_arg > 0:
        return int(n_jobs_arg)
    # SLURM fallback
    v = os.environ.get("SLURM_CPUS_PER_TASK")
    if v:
        try:
            return max(1, int(str(v).strip()))
        except Exception:
            pass
    return int(os.cpu_count() or 1)


def normalize_fid(fid) -> str:
    fid = str(fid).strip()
    return str(int(fid)) if fid.isdigit() else fid


def safe_ohe():
    # scikit-learn changed OneHotEncoder arg from sparse -> sparse_output
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def most_common_params(params_list: List[Dict]) -> Dict:
    if len(params_list) == 0:
        return {}
    tuples = [tuple(sorted(p.items())) for p in params_list]
    return dict(Counter(tuples).most_common(1)[0][0])


def strip_feature_name(name: str) -> str:
    s = str(name)
    for pref in ("num__", "cat__", "remainder__"):
        if s.startswith(pref):
            s = s[len(pref):]
    if s.endswith(" EU"):
        s = s[:-3]
    return s


def build_preprocessor(df: pd.DataFrame, input_columns: List[str]) -> Tuple[ColumnTransformer, List[str], List[str]]:
    X = df[input_columns]
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in input_columns if c not in cat_cols]

    transformers = []
    if num_cols:
        transformers.append(("num", "passthrough", num_cols))
    if cat_cols:
        transformers.append(("cat", safe_ohe(), cat_cols))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")
    return pre, num_cols, cat_cols


def build_pipeline(df: pd.DataFrame, input_columns: List[str], n_jobs: int, seed: int) -> Pipeline:
    pre, _, _ = build_preprocessor(df, input_columns)
    rf = RandomForestClassifier(
        n_jobs=n_jobs,
        random_state=seed,
        class_weight="balanced",
        max_features="sqrt",
        bootstrap=True,
    )
    return Pipeline([("pre", pre), ("clf", rf)])


def rf_param_dist() -> Dict[str, object]:
    # Note: keep space reasonable; you can expand later if desired
    return {
        "clf__n_estimators": randint(50, 300),
        "clf__max_depth": [None] + list(range(2, 15)),
        "clf__min_samples_split": randint(2, 20),
        "clf__min_samples_leaf": randint(1, 15),
        "clf__max_features": ["sqrt", "log2", 0.3, 0.5],
        # Optionally tune max_samples if you want (bootstrap must be True)
        # "clf__max_samples": [None, 0.6, 0.8, 0.9],
    }


# =========================
# Calibration helpers
# =========================

def _make_calibrator(base_estimator, method: str, cv, n_jobs: int):
    """
    sklearn changed arg name from base_estimator -> estimator in newer versions.
    This helper supports both.
    """
    try:
        return CalibratedClassifierCV(
            estimator=base_estimator,
            method=method,
            cv=cv,
            n_jobs=n_jobs,
        )
    except TypeError:
        return CalibratedClassifierCV(
            base_estimator=base_estimator,
            method=method,
            cv=cv,
            n_jobs=n_jobs,
        )


def fit_calibrator_on_train(
    fitted_model,
    X_train,
    y_train,
    method: str,
    n_splits: int,
    random_state: int,
    n_jobs: int,
):
    """
    Fit a probability calibrator using ONLY training data.
    Uses StratifiedKFold to avoid leakage.
    Returns a fitted calibrator with predict_proba().
    """
    calib_cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    cal = _make_calibrator(
        base_estimator=fitted_model,
        method=method,
        cv=calib_cv,
        n_jobs=n_jobs,
    )
    cal.fit(X_train, y_train)
    return cal

def fit_prob_calibrator(p: np.ndarray, y: np.ndarray, method: str, clip: float = 1e-6):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)

    p = np.clip(p, clip, 1.0 - clip)

    if method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(p, y)
        return cal

    # sigmoid / Platt on logit(p)
    z = np.log(p / (1.0 - p)).reshape(-1, 1)
    lr = LogisticRegression(solver="lbfgs", max_iter=2000)
    lr.fit(z, y)
    return lr

def apply_prob_calibrator(cal, p: np.ndarray, method: str, clip: float = 1e-6) -> np.ndarray:
    """
    Apply a saved probability-mapper calibrator to raw probabilities p.
    This does NOT modify p in-place; it returns a new array.
    """
    p = np.asarray(p, dtype=float)
    p_clip = np.clip(p, clip, 1.0 - clip)

    if method == "isotonic":
        return cal.predict(p_clip).astype(float)

    # sigmoid case depends on how YOU trained it:
    # A) If you trained LogisticRegression on p.reshape(-1,1):
    return cal.predict_proba(p_clip.reshape(-1, 1))[:, 1].astype(float)

    # B) If instead you trained on logit(p) (recommended), then use this instead:
    # z = np.log(p_clip / (1.0 - p_clip)).reshape(-1, 1)
    # return cal.predict_proba(z)[:, 1].astype(float)

# =========================
# SHAP
# =========================

def shap_global_importance_for_pipeline(
    fitted_pipe: Pipeline,
    X_raw: pd.DataFrame,
    max_samples: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """
    Compute global SHAP importance for a fitted pipeline with a tree model.
    We compute SHAP on the transformed feature matrix.
    """
    if shap is None:
        return pd.DataFrame(columns=["feature", "mean_abs_shap", "mean_shap"])

    if len(X_raw) == 0:
        return pd.DataFrame(columns=["feature", "mean_abs_shap", "mean_shap"])

    if max_samples is not None and len(X_raw) > max_samples:
        X_raw = X_raw.sample(n=max_samples, random_state=seed)

    pre = fitted_pipe.named_steps["pre"]
    clf = fitted_pipe.named_steps["clf"]

    X_mat = pre.transform(X_raw)

    try:
        feat_names = pre.get_feature_names_out().tolist()
    except Exception:
        feat_names = [f"f{i}" for i in range(X_mat.shape[1])]

    feat_names = [strip_feature_name(x) for x in feat_names]

    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X_mat)

    # Binary classification: shap may be list [class0, class1] or array
    if isinstance(shap_vals, list):
        sv = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
    else:
        sv = shap_vals
        if sv.ndim == 3 and sv.shape[-1] >= 2:
            sv = sv[:, :, 1]

    mean_abs = np.abs(sv).mean(axis=0)
    mean_sv = sv.mean(axis=0)

    out = pd.DataFrame({
        "feature": feat_names,
        "mean_abs_shap": mean_abs,
        "mean_shap": mean_sv,
    }).sort_values("mean_abs_shap", ascending=False)

    return out


# =========================
# Threshold + metrics helpers
# =========================

def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def metrics_at_threshold(y_true: np.ndarray, prob: np.ndarray, thr: float) -> Dict[str, float]:
    y_hat = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()

    prec = precision_score(y_true, y_hat, zero_division=0)
    rec = recall_score(y_true, y_hat, zero_division=0)
    f1 = f1_score(y_true, y_hat, zero_division=0)

    tpr = rec
    fpr = _safe_div(fp, fp + tn)
    tnr = _safe_div(tn, tn + fp)  # specificity
    youden_j = tpr - fpr
    npv = _safe_div(tn, tn + fn)

    return {
        "threshold": float(thr),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "specificity": float(tnr),
        "fpr": float(fpr),
        "tpr": float(tpr),
        "youden_j": float(youden_j),
        "npv": float(npv),
        "pred_pos_rate": float(y_hat.mean()),
    }


def pick_threshold_f1_from_pr(y_true: np.ndarray, prob: np.ndarray) -> Tuple[float, pd.DataFrame]:
    """
    Pick threshold that maximizes F1 using PR-curve thresholds.
    """
    prec, rec, thr = precision_recall_curve(y_true, prob)
    if len(thr) == 0:
        return 0.5, pd.DataFrame(columns=["threshold", "precision", "recall", "f1"])

    # thr length = len(prec)-1
    prec_t = prec[:-1]
    rec_t = rec[:-1]
    f1 = (2 * prec_t * rec_t) / np.maximum(prec_t + rec_t, 1e-12)

    sweep = pd.DataFrame({
        "threshold": thr.astype(float),
        "precision": prec_t.astype(float),
        "recall": rec_t.astype(float),
        "f1": f1.astype(float),
    })
    best_idx = int(np.nanargmax(sweep["f1"].to_numpy()))
    return float(sweep.iloc[best_idx]["threshold"]), sweep


def pick_threshold_youden_from_roc(y_true: np.ndarray, prob: np.ndarray) -> Tuple[float, pd.DataFrame]:
    """
    Pick threshold that maximizes Youden J = TPR - FPR using ROC thresholds.
    """
    fpr, tpr, thr = roc_curve(y_true, prob)
    if len(thr) == 0:
        return 0.5, pd.DataFrame(columns=["threshold", "tpr", "fpr", "youden_j"])

    youden = tpr - fpr
    sweep = pd.DataFrame({
        "threshold": thr.astype(float),
        "tpr": tpr.astype(float),
        "fpr": fpr.astype(float),
        "youden_j": youden.astype(float),
    })
    best_idx = int(np.nanargmax(sweep["youden_j"].to_numpy()))
    return float(sweep.iloc[best_idx]["threshold"]), sweep


def pick_threshold_for_recall(
    y_true: np.ndarray,
    prob: np.ndarray,
    target_recall: float,
) -> Tuple[float, pd.DataFrame]:
    """
    Choose the HIGHEST threshold that still achieves recall >= target_recall
    (minimize FPs while meeting recall constraint).

    Returns (thr, sweep_df) where sweep_df includes:
      threshold, precision, recall, f1, meets_recall, target_recall
    """
    prec, rec, thr = precision_recall_curve(y_true, prob)
    if len(thr) == 0:
        sweep = pd.DataFrame(columns=["target_recall", "threshold", "precision", "recall", "f1", "meets_recall"])
        return 0.5, sweep

    prec_t = prec[:-1]
    rec_t = rec[:-1]
    f1 = (2 * prec_t * rec_t) / np.maximum(prec_t + rec_t, 1e-12)

    sweep = pd.DataFrame({
        "target_recall": float(target_recall),
        "threshold": thr.astype(float),
        "precision": prec_t.astype(float),
        "recall": rec_t.astype(float),
        "f1": f1.astype(float),
    })
    sweep["meets_recall"] = sweep["recall"] >= float(target_recall)

    if not sweep["meets_recall"].any():
        # fallback: max recall, then max precision
        max_rec = float(sweep["recall"].max())
        cand = sweep[sweep["recall"] == max_rec]
        best_row = cand.sort_values(["precision", "threshold"], ascending=[False, True]).iloc[0]
        return float(best_row["threshold"]), sweep

    cand = sweep[sweep["meets_recall"]].copy()
    best_row = cand.sort_values(["threshold"], ascending=False).iloc[0]
    return float(best_row["threshold"]), sweep


def compute_and_save_curves(
    y: np.ndarray,
    prob: np.ndarray,
    out_dir: str,
    suffix: str = "",
) -> None:
    """
    Save ROC and PR curves for a probability stream.
    """
    fpr, tpr, roc_thr = roc_curve(y, prob)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thr}).to_csv(
        os.path.join(out_dir, f"nestedcv_roc_curve{suffix}.csv"), index=False
    )

    pr_prec, pr_rec, pr_thr = precision_recall_curve(y, prob)
    pd.DataFrame({
        "precision": pr_prec,
        "recall": pr_rec,
        "threshold": np.r_[pr_thr, np.nan],
    }).to_csv(os.path.join(out_dir, f"nestedcv_pr_curve{suffix}.csv"), index=False)


def threshold_bundle(
    y: np.ndarray,
    prob: np.ndarray,
    thr_eval: float,
    target_recalls: List[float],
    out_dir: str,
    suffix: str = "",
) -> Dict[str, object]:
    """
    Compute:
      - thresholds: fixed, f1_opt, youden_j_opt, recall_X
      - sweeps: PR/ROC/recall
      - OOF threshold metrics
    Saves files with optional suffix (e.g., "_cal").
    Returns dict with thresholds + metrics rows.
    """
    thr_f1, sweep_pr = pick_threshold_f1_from_pr(y, prob)
    thr_j, sweep_roc = pick_threshold_youden_from_roc(y, prob)

    sweep_pr.to_csv(os.path.join(out_dir, f"threshold_sweep_pr{suffix}.csv"), index=False)
    sweep_roc.to_csv(os.path.join(out_dir, f"threshold_sweep_roc{suffix}.csv"), index=False)

    thr_set: Dict[str, float] = {
        "fixed": float(thr_eval),
        "f1_opt": float(thr_f1),
        "youden_j_opt": float(thr_j),
    }

    recall_sweeps = []
    for r in target_recalls:
        thr_r, sweep_r = pick_threshold_for_recall(y, prob, target_recall=r)
        key = f"recall_{r:.2f}"
        thr_set[key] = float(thr_r)
        recall_sweeps.append(sweep_r)

    if recall_sweeps:
        pd.concat(recall_sweeps, ignore_index=True).to_csv(
            os.path.join(out_dir, f"threshold_sweep_recall{suffix}.csv"), index=False
        )

    # OOF threshold metrics
    oof_auc = float(roc_auc_score(y, prob))
    oof_auprc = float(average_precision_score(y, prob))

    thr_metrics_rows = []
    for name, thr in thr_set.items():
        m = metrics_at_threshold(y, prob, thr)
        thr_metrics_rows.append({
            "threshold_name": name,
            **m,
            "oof_auc": oof_auc,
            "oof_auprc": oof_auprc,
        })

    thr_metrics_df = pd.DataFrame(thr_metrics_rows).sort_values("threshold_name")
    thr_metrics_df.to_csv(os.path.join(out_dir, f"threshold_metrics_oof{suffix}.csv"), index=False)

    return {
        "oof_auc": oof_auc,
        "oof_auprc": oof_auprc,
        "thresholds": {k: float(v) for k, v in thr_set.items()},
        "threshold_metrics_rows": thr_metrics_rows,
    }


def fold_threshold_metrics(
    y: np.ndarray,
    oof_prob: np.ndarray,
    outer_folds: int,
    seed: int,
    thr_set: Dict[str, float],
    out_dir: str,
    suffix: str = "",
) -> pd.DataFrame:
    """
    Evaluate thresholded metrics per outer fold, using the global OOF-chosen thresholds.
    Saves threshold_metrics_by_fold{suffix}.csv
    """
    rows = []
    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)
    for fold, (_, te) in enumerate(outer_cv.split(np.zeros(len(y)), y), start=1):
        y_te = y[te]
        p_te = oof_prob[te]
        base = {"fold": int(fold), "n_test": int(len(te)), "n_pos_test": int(y_te.sum())}
        for name, thr in thr_set.items():
            m = metrics_at_threshold(y_te, p_te, thr)
            rows.append({
                **base,
                "threshold_name": name,
                "threshold": float(thr),
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "specificity": m["specificity"],
                "youden_j": m["youden_j"],
                "npv": m["npv"],
                "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"],
                "pred_pos_rate": m["pred_pos_rate"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, f"threshold_metrics_by_fold{suffix}.csv"), index=False)
    return df


# =========================
# Main runner
# =========================

def load_modality_and_merge(
    modality: str,
    input_dir: str,
    pheno_path: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Returns merged dataframe with columns:
      FID + input_columns + PSC
    and input_columns list.
    """
    if modality == "genetics":
        df = pd.read_csv(os.path.join(input_dir, "data_genetics_only_significant_snps_imputed.csv"))
        input_columns = [c for c in df.columns if ":" in c]

    elif modality == "lab":
        df = pd.read_csv(os.path.join(input_dir, "data_lab_imputed.csv"))
        input_columns = [c for c in df.columns if c != FID_COL]

    elif modality == "serology":
        df = pd.read_csv(os.path.join(input_dir, "data_serology_imputed_MIRIAD.csv"))
        input_columns = ["ANCA EU", "CBir1 EU", "OmpC EU", "IgA ASCA EU", "IgG ASCA EU"]

    elif modality == "clinical":
        df = pd.read_csv(os.path.join(input_dir, "data_clinical_imputed.csv"))
        input_columns = [
            "Gender_Self", "Jewish_Self", "Diagnosis", "AgeDx", "Smoking",
            "FamilyHistory", "extent_UC", "Surgery", "upper_GI",
            "disease_location_CD", "behavior_CD", "Perianal"
        ]
    else:
        raise ValueError(f"Unknown modality: {modality}")

    pheno = pd.read_csv(pheno_path)

    df[FID_COL] = df[FID_COL].apply(normalize_fid)
    pheno[FID_COL] = pheno[FID_COL].apply(normalize_fid)

    # Keep your exact filters
    pheno = pheno[
        (pheno["Race_Admix"] == "White") &
        (pheno["Diagnosis"].isin(["CD", "UC"]))
    ].dropna(subset=[LABEL_COL]).reset_index(drop=True)

    merged = pd.merge(df[[FID_COL] + input_columns], pheno[[FID_COL, LABEL_COL]], on=FID_COL, how="inner")
    return merged, input_columns


def run_modality(
    modality: str,
    input_dir: str,
    pheno_path: str,
    out_dir: str,
    n_jobs: int,
    seed: int,
    outer_folds: int = 10,
    inner_folds: int = 5,
    inner_iters: int = 30,
    shap_max_samples: int = 2000,
    thr_eval: float = 0.50,
    target_recalls: Optional[List[float]] = None,
    # calibration
    do_calibration: bool = True,
    calib_method: str = "sigmoid",
    calib_folds: int = 5,
    # shap control
    shap_do: bool = True,
    save_oof_calibrators: bool = False,
    calibration_clip: float = 1e-6,
):

    if target_recalls is None or len(target_recalls) == 0:
        target_recalls = [0.90]
    target_recalls = sorted({float(x) for x in target_recalls}, reverse=True)

    ensure_dir(out_dir)

    print(f"\n=== Running single modality: {modality} ===")
    print(f"Saving outputs to: {out_dir}")
    print(f"Using n_jobs = {n_jobs}")
    print("Inner CV scoring = average_precision (AUPRC/AP)")
    print(f"Recall targets (OOF thresholding): {', '.join(f'{r:.2f}' for r in target_recalls)}")
    print(f"Calibration: {'ON' if do_calibration else 'OFF'}"
          f"{' | method=' + calib_method + ', folds=' + str(calib_folds) if do_calibration else ''}")
    print(f"SHAP: {'ON' if (shap_do and shap is not None) else 'OFF'}")

    # -------------------------
    # Load data
    # -------------------------
    df, input_columns = load_modality_and_merge(modality, input_dir, pheno_path)
    df = df.drop_duplicates(subset=[FID_COL]).reset_index(drop=True)

    X_raw = df[input_columns].copy()
    y = df[LABEL_COL].astype(int).to_numpy()
    fids = df[FID_COL].to_numpy()

    with open(os.path.join(out_dir, "training_columns.json"), "w") as f:
        json.dump({"modality": modality, "input_columns": input_columns}, f, indent=2)

    # -------------------------
    # Nested CV: outer (OOF) + inner (tuning)
    # -------------------------
    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)

    oof_prob = np.full(len(y), np.nan, dtype=float)
    oof_prob_cal = np.full(len(y), np.nan, dtype=float) if do_calibration else None
    fold_id = np.full(len(y), -1, dtype=int)

    fold_rows: List[Dict[str, object]] = []
    best_params: List[Dict] = []

    for fold, (tr, te) in enumerate(outer_cv.split(X_raw, y), start=1):
        X_tr, y_tr = X_raw.iloc[tr], y[tr]
        X_te, y_te = X_raw.iloc[te], y[te]
        fold_id[te] = fold

        # Build pipeline for this fold
        pipe = build_pipeline(df=df.iloc[tr], input_columns=input_columns, n_jobs=n_jobs, seed=seed + fold)

        inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed + 100 + fold)

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=rf_param_dist(),
            n_iter=inner_iters,
            scoring="average_precision",
            n_jobs=n_jobs,
            cv=inner_cv,
            random_state=seed + 200 + fold,
            verbose=0,
        )
        search.fit(X_tr, y_tr)
        best_pipe = search.best_estimator_
        best_params.append(search.best_params_)

        # Uncalibrated test probs
        probs = best_pipe.predict_proba(X_te)[:, 1].astype(float)
        oof_prob[te] = probs

        fold_auc = roc_auc_score(y_te, probs)
        fold_ap = average_precision_score(y_te, probs)

        row = {
            "fold": int(fold),
            "n_test": int(len(te)),
            "n_pos_test": int(y_te.sum()),
            "auc": float(fold_auc),
            "auprc": float(fold_ap),
        }

        print(f"Fold {fold}: AUC={fold_auc:.4f} | AUPRC(AP)={fold_ap:.4f}")

        # Calibrated test probs (fit calibrator ONLY on outer-train)
        if do_calibration:
            cal = fit_calibrator_on_train(
                fitted_model=best_pipe,
                X_train=X_tr,
                y_train=y_tr,
                method=calib_method,
                n_splits=calib_folds,
                random_state=seed + 3000 + fold,
                n_jobs=n_jobs,
            )
            probs_cal = cal.predict_proba(X_te)[:, 1].astype(float)
            assert oof_prob_cal is not None
            oof_prob_cal[te] = probs_cal

            fold_auc_cal = roc_auc_score(y_te, probs_cal)
            fold_ap_cal = average_precision_score(y_te, probs_cal)
            row.update({
                "auc_cal": float(fold_auc_cal),
                "auprc_cal": float(fold_ap_cal),
            })
            print(f"         CAL: AUC={fold_auc_cal:.4f} | AUPRC(AP)={fold_ap_cal:.4f}")

        fold_rows.append(row)

    # Sanity
    if np.isnan(oof_prob).any():
        raise RuntimeError("OOF probabilities contain NaNs. Something went wrong in CV assignment.")
    if do_calibration and (oof_prob_cal is None or np.isnan(oof_prob_cal).any()):
        raise RuntimeError("Calibrated OOF probabilities contain NaNs. Check calibration setup.")

    # ------------------------------------------------------------
    # Save deployable calibrators (OOF probability mapping)
    #   - Uncalibrated stream always
    #   - Calibrated stream (if enabled) optionally
    # ------------------------------------------------------------
    if save_oof_calibrators:
        models_dir = os.path.join(out_dir, "models")
        ensure_dir(models_dir)

        y_oof = y.astype(int)

        # ---- Uncalibrated OOF mapping ----
        p_uncal = np.asarray(oof_prob, dtype=float)
        m_uncal = np.isfinite(p_uncal) & np.isfinite(y_oof)
        p_uncal = np.clip(p_uncal[m_uncal], calibration_clip, 1.0 - calibration_clip)
        y_uncal = y_oof[m_uncal]

        saved_paths = {}

        if len(np.unique(y_uncal)) < 2:
            print("WARNING: Cannot fit OOF calibrator for uncalibrated stream (need both classes). Skipping.")
        else:
            cal_uncal = fit_prob_calibrator(p_uncal, y_uncal, method=calib_method)
            path_uncal = os.path.join(models_dir, "calibrator_oof_uncal.joblib")
            joblib.dump(cal_uncal, path_uncal)
            saved_paths["calibrator_oof_uncal"] = path_uncal

        # ---- Calibrated OOF mapping (optional) ----
        if do_calibration:
            assert oof_prob_cal is not None
            p_cal = np.asarray(oof_prob_cal, dtype=float)
            m_cal = np.isfinite(p_cal) & np.isfinite(y_oof)
            p_cal = np.clip(p_cal[m_cal], calibration_clip, 1.0 - calibration_clip)
            y_cal2 = y_oof[m_cal]

            if len(np.unique(y_cal2)) < 2:
                print("WARNING: Cannot fit OOF calibrator for calibrated stream (need both classes). Skipping.")
            else:
                cal_cal = fit_prob_calibrator(p_cal, y_cal2, method=calib_method)
                path_cal = os.path.join(models_dir, "calibrator_oof_cal.joblib")
                joblib.dump(cal_cal, path_cal)
                saved_paths["calibrator_oof_cal"] = path_cal

        meta = {
            "kind": "probability_mapping",
            "modality": modality,
            "method": calib_method,
            "clip": float(calibration_clip),
            "n_oof": int(len(y_oof)),
            "n_pos": int(y_oof.sum()),
            "prevalence": float(y_oof.mean()),
            "streams": {
                "uncalibrated": "y_oof_prob",
                "calibrated": "y_oof_prob_cal" if do_calibration else None,
            },
            "paths": saved_paths,
        }
        with open(os.path.join(models_dir, "calibrators_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        if saved_paths:
            print("Saved deployable OOF calibrators:")
            for k, v in saved_paths.items():
                print(f"  - {k}: {v}")

    # Save per-fold ranking metrics
    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(os.path.join(out_dir, "fold_ranking_metrics.csv"), index=False)

    # OOF ranking metrics
    oof_auc = float(roc_auc_score(y, oof_prob))
    oof_auprc = float(average_precision_score(y, oof_prob))
    oof_rank_rows = [{"stream": "uncalibrated", "oof_auc": oof_auc, "oof_auprc": oof_auprc}]

    if do_calibration:
        assert oof_prob_cal is not None
        oof_auc_cal = float(roc_auc_score(y, oof_prob_cal))
        oof_auprc_cal = float(average_precision_score(y, oof_prob_cal))
        oof_rank_rows.append({"stream": "calibrated", "oof_auc": oof_auc_cal, "oof_auprc": oof_auprc_cal})

    pd.DataFrame(oof_rank_rows).to_csv(os.path.join(out_dir, "oof_ranking_metrics.csv"), index=False)

    # -------------------------
    # Curves (OOF)
    # -------------------------
    compute_and_save_curves(y, oof_prob, out_dir, suffix="")
    if do_calibration:
        compute_and_save_curves(y, oof_prob_cal, out_dir, suffix="_cal")  # type: ignore

    # -------------------------
    # Threshold selection + metrics (OOF) for uncalibrated
    # -------------------------
    thr_uncal = threshold_bundle(
        y=y,
        prob=oof_prob,
        thr_eval=thr_eval,
        target_recalls=target_recalls,
        out_dir=out_dir,
        suffix="",
    )
    thr_set_uncal: Dict[str, float] = thr_uncal["thresholds"]  # type: ignore

    fold_thr_uncal = fold_threshold_metrics(
        y=y,
        oof_prob=oof_prob,
        outer_folds=outer_folds,
        seed=seed,
        thr_set=thr_set_uncal,
        out_dir=out_dir,
        suffix="",
    )

    # -------------------------
    # Threshold selection + metrics (OOF) for calibrated (if enabled)
    # -------------------------
    thr_cal = None
    thr_set_cal: Dict[str, float] = {}
    fold_thr_cal = None

    if do_calibration:
        assert oof_prob_cal is not None
        thr_cal = threshold_bundle(
            y=y,
            prob=oof_prob_cal,
            thr_eval=thr_eval,
            target_recalls=target_recalls,
            out_dir=out_dir,
            suffix="_cal",
        )
        thr_set_cal = thr_cal["thresholds"]  # type: ignore

        fold_thr_cal = fold_threshold_metrics(
            y=y,
            oof_prob=oof_prob_cal,
            outer_folds=outer_folds,
            seed=seed,
            thr_set=thr_set_cal,
            out_dir=out_dir,
            suffix="_cal",
        )

    # -------------------------
    # Save OOF predictions (single file)
    # -------------------------
    out_pred = pd.DataFrame({
        FID_COL: fids,
        "fold_id": fold_id,
        "y_true": y.astype(int),
        "y_oof_prob": oof_prob.astype(float),
    })

    # uncal thresholded preds
    for name, thr in thr_set_uncal.items():
        out_pred[f"y_oof_pred__{name}"] = (oof_prob >= thr).astype(int)

    # calibrated stream + calibrated thresholded preds
    if do_calibration:
        assert oof_prob_cal is not None
        out_pred["y_oof_prob_cal"] = oof_prob_cal.astype(float)
        for name, thr in thr_set_cal.items():
            out_pred[f"y_oof_pred_cal__{name}"] = (oof_prob_cal >= thr).astype(int)

    out_pred.to_csv(os.path.join(out_dir, "nestedcv_oof_predictions.csv"), index=False)

    # -------------------------
    # Summary JSON (ALL metrics + thresholds)
    # -------------------------
    mean_fold_at_thr_uncal = (
        fold_thr_uncal
        .groupby("threshold_name")[["precision", "recall", "f1", "specificity", "youden_j", "npv", "pred_pos_rate"]]
        .agg(["mean", "std"])
    )

    fold_auc_mean = float(fold_df["auc"].mean())
    fold_auc_std = float(fold_df["auc"].std(ddof=0)) if len(fold_df) else 0.0
    fold_auprc_mean = float(fold_df["auprc"].mean())
    fold_auprc_std = float(fold_df["auprc"].std(ddof=0)) if len(fold_df) else 0.0

    summary = {
        "modality": modality,
        "n_jobs": int(n_jobs),
        "seed": int(seed),
        "n_samples": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
        "prevalence": float(y.mean()),
        "outer_folds": int(outer_folds),
        "inner_folds": int(inner_folds),
        "inner_iters": int(inner_iters),
        "scoring_inner_cv": "average_precision",
        "thr_eval": float(thr_eval),
        "target_recalls": [float(r) for r in target_recalls],
        "calibration": {
            "enabled": bool(do_calibration),
            "method": str(calib_method),
            "folds": int(calib_folds),
        },
        "uncalibrated": {
            "oof_auc": float(thr_uncal["oof_auc"]),      # type: ignore
            "oof_auprc": float(thr_uncal["oof_auprc"]),  # type: ignore
            "thresholds": {k: float(v) for k, v in thr_set_uncal.items()},
            "fold_auc_mean": fold_auc_mean,
            "fold_auc_std": fold_auc_std,
            "fold_auprc_mean": fold_auprc_mean,
            "fold_auprc_std": fold_auprc_std,
            "fold_threshold_metrics_mean_std": {
                thr_name: {
                    "precision_mean": float(mean_fold_at_thr_uncal.loc[thr_name, ("precision", "mean")]),
                    "precision_std": float(mean_fold_at_thr_uncal.loc[thr_name, ("precision", "std")]),
                    "recall_mean": float(mean_fold_at_thr_uncal.loc[thr_name, ("recall", "mean")]),
                    "recall_std": float(mean_fold_at_thr_uncal.loc[thr_name, ("recall", "std")]),
                    "f1_mean": float(mean_fold_at_thr_uncal.loc[thr_name, ("f1", "mean")]),
                    "f1_std": float(mean_fold_at_thr_uncal.loc[thr_name, ("f1", "std")]),
                    "specificity_mean": float(mean_fold_at_thr_uncal.loc[thr_name, ("specificity", "mean")]),
                    "specificity_std": float(mean_fold_at_thr_uncal.loc[thr_name, ("specificity", "std")]),
                    "youden_j_mean": float(mean_fold_at_thr_uncal.loc[thr_name, ("youden_j", "mean")]),
                    "youden_j_std": float(mean_fold_at_thr_uncal.loc[thr_name, ("youden_j", "std")]),
                    "npv_mean": float(mean_fold_at_thr_uncal.loc[thr_name, ("npv", "mean")]),
                    "npv_std": float(mean_fold_at_thr_uncal.loc[thr_name, ("npv", "std")]),
                    "pred_pos_rate_mean": float(mean_fold_at_thr_uncal.loc[thr_name, ("pred_pos_rate", "mean")]),
                    "pred_pos_rate_std": float(mean_fold_at_thr_uncal.loc[thr_name, ("pred_pos_rate", "std")]),
                }
                for thr_name in mean_fold_at_thr_uncal.index.tolist()
            },
        },
    }

    if do_calibration:
        assert oof_prob_cal is not None
        assert thr_cal is not None
        assert fold_thr_cal is not None

        # fold ranking stats for calibrated stream
        if "auc_cal" in fold_df.columns and "auprc_cal" in fold_df.columns:
            fold_auc_cal_mean = float(fold_df["auc_cal"].mean())
            fold_auc_cal_std = float(fold_df["auc_cal"].std(ddof=0)) if len(fold_df) else 0.0
            fold_auprc_cal_mean = float(fold_df["auprc_cal"].mean())
            fold_auprc_cal_std = float(fold_df["auprc_cal"].std(ddof=0)) if len(fold_df) else 0.0
        else:
            fold_auc_cal_mean, fold_auc_cal_std = 0.0, 0.0
            fold_auprc_cal_mean, fold_auprc_cal_std = 0.0, 0.0

        mean_fold_at_thr_cal = (
            fold_thr_cal
            .groupby("threshold_name")[["precision", "recall", "f1", "specificity", "youden_j", "npv", "pred_pos_rate"]]
            .agg(["mean", "std"])
        )

        summary["calibrated"] = {
            "oof_auc": float(thr_cal["oof_auc"]),      # type: ignore
            "oof_auprc": float(thr_cal["oof_auprc"]),  # type: ignore
            "thresholds": {k: float(v) for k, v in thr_set_cal.items()},
            "fold_auc_mean": fold_auc_cal_mean,
            "fold_auc_std": fold_auc_cal_std,
            "fold_auprc_mean": fold_auprc_cal_mean,
            "fold_auprc_std": fold_auprc_cal_std,
            "fold_threshold_metrics_mean_std": {
                thr_name: {
                    "precision_mean": float(mean_fold_at_thr_cal.loc[thr_name, ("precision", "mean")]),
                    "precision_std": float(mean_fold_at_thr_cal.loc[thr_name, ("precision", "std")]),
                    "recall_mean": float(mean_fold_at_thr_cal.loc[thr_name, ("recall", "mean")]),
                    "recall_std": float(mean_fold_at_thr_cal.loc[thr_name, ("recall", "std")]),
                    "f1_mean": float(mean_fold_at_thr_cal.loc[thr_name, ("f1", "mean")]),
                    "f1_std": float(mean_fold_at_thr_cal.loc[thr_name, ("f1", "std")]),
                    "specificity_mean": float(mean_fold_at_thr_cal.loc[thr_name, ("specificity", "mean")]),
                    "specificity_std": float(mean_fold_at_thr_cal.loc[thr_name, ("specificity", "std")]),
                    "youden_j_mean": float(mean_fold_at_thr_cal.loc[thr_name, ("youden_j", "mean")]),
                    "youden_j_std": float(mean_fold_at_thr_cal.loc[thr_name, ("youden_j", "std")]),
                    "npv_mean": float(mean_fold_at_thr_cal.loc[thr_name, ("npv", "mean")]),
                    "npv_std": float(mean_fold_at_thr_cal.loc[thr_name, ("npv", "std")]),
                    "pred_pos_rate_mean": float(mean_fold_at_thr_cal.loc[thr_name, ("pred_pos_rate", "mean")]),
                    "pred_pos_rate_std": float(mean_fold_at_thr_cal.loc[thr_name, ("pred_pos_rate", "std")]),
                }
                for thr_name in mean_fold_at_thr_cal.index.tolist()
            },
        }

    with open(os.path.join(out_dir, "nestedcv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # -------------------------
    # Final refit (+ optional calibrator) + SHAP
    # -------------------------
    final_params = most_common_params(best_params)

    final_pipe = build_pipeline(df=df, input_columns=input_columns, n_jobs=n_jobs, seed=seed)
    if final_params:
        final_pipe.set_params(**final_params)
    final_pipe.fit(X_raw, y)
    joblib.dump(final_pipe, os.path.join(out_dir, "best_model.joblib"))

    # For deployment: fit a calibrator on FULL data via CV (no leakage concern here; no held-out test)
    if do_calibration:
        final_cal = fit_calibrator_on_train(
            fitted_model=final_pipe,
            X_train=X_raw,
            y_train=y,
            method=calib_method,
            n_splits=calib_folds,
            random_state=seed + 9999,
            n_jobs=n_jobs,
        )
        joblib.dump(final_cal, os.path.join(out_dir, "best_model_calibrator.joblib"))

    # SHAP
    if shap_do and shap is not None:
        shap_df = shap_global_importance_for_pipeline(
            fitted_pipe=final_pipe,
            X_raw=X_raw,
            max_samples=shap_max_samples,
            seed=seed,
        )
        shap_df.to_csv(os.path.join(out_dir, "shap_global_importance.csv"), index=False)

    # -------------------------
    # Print summary
    # -------------------------
    print("\n==============================")
    print(f"DONE [{modality}]")
    print("==============================")
    print(f"OOF (uncal) AUC={oof_auc:.4f} | OOF (uncal) AUPRC={oof_auprc:.4f}")
    print("Selected thresholds (uncal):")
    for k in sorted(thr_set_uncal.keys()):
        print(f"  - {k}: {thr_set_uncal[k]:.6f}")

    if do_calibration:
        assert oof_prob_cal is not None
        oof_auc_cal = float(roc_auc_score(y, oof_prob_cal))
        oof_auprc_cal = float(average_precision_score(y, oof_prob_cal))
        print(f"\nOOF (cal)   AUC={oof_auc_cal:.4f} | OOF (cal)   AUPRC={oof_auprc_cal:.4f}")
        print("Selected thresholds (cal):")
        for k in sorted(thr_set_cal.keys()):
            print(f"  - {k}: {thr_set_cal[k]:.6f}")

    print(f"\nSaved: {os.path.join(out_dir, 'best_model.joblib')}")
    if do_calibration:
        print(f"Saved: {os.path.join(out_dir, 'best_model_calibrator.joblib')}")
    print(f"Saved: {os.path.join(out_dir, 'nestedcv_oof_predictions.csv')}")
    print("execution done!")


# =========================
# Entry point
# =========================

if __name__ == "__main__":
    set_thread_env_for_hpc()

    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", required=True, choices=["genetics", "lab", "serology", "clinical"])
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--pheno-path", required=True)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--outer-folds", type=int, default=10)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--inner-iters", type=int, default=30)

    # SHAP
    parser.add_argument("--no-shap", action="store_true")
    parser.add_argument("--shap-max-samples", type=int, default=2000)

    # Thresholding
    parser.add_argument("--thr-eval", type=float, default=0.50)
    parser.add_argument("--target-recall", type=float, nargs="+", default=[0.90])

    # Calibration
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument("--calibration-method", type=str, default="sigmoid", choices=["sigmoid", "isotonic"])
    parser.add_argument("--calibration-folds", type=int, default=5)
    # Save deployable OOF calibrators (probability mapping)
    parser.add_argument("--save-calibrators", action="store_true")
    parser.add_argument("--calibration-clip",type=float,default=1e-6)


    args = parser.parse_args()

    run_modality(
        modality=args.modality,
        input_dir=args.input_dir,
        pheno_path=args.pheno_path,
        out_dir=args.out_dir,
        n_jobs=get_n_jobs(args.n_jobs),
        seed=args.seed,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        inner_iters=args.inner_iters,
        shap_max_samples=args.shap_max_samples,
        thr_eval=args.thr_eval,
        target_recalls=args.target_recall,
        do_calibration=(not args.no_calibration),
        calib_method=args.calibration_method,
        calib_folds=args.calibration_folds,
        shap_do=(not args.no_shap),
        save_oof_calibrators=args.save_calibrators,
        calibration_clip=args.calibration_clip,
    )

