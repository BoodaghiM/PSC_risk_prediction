#!/usr/bin/env python3
"""
multi_modal.py

Shared-outer-CV evaluation on the INTERSECTION cohort (subjects with ALL modalities),
with fair comparison across:
  1) Single-modality RF baselines (trained/evaluated on intersection folds)
  2) Early integration RF (concatenated features, intersection only)
  3) Late integration (stacking):
       - Base RFs trained on EXPANDED modality cohorts (excluding outer test FIDs per fold)
       - Meta LogisticRegression trained on intersection-train using INNER OOF base probs
       - Evaluated on intersection test fold

UPDATED (to match your single_modal changes):
- Inner CV scoring = AUPRC / Average Precision ("average_precision")
- Reports BOTH AUC and AUPRC (per fold + OOF)
- OOF-based thresholding ONCE per model (no subfolders):
    * fixed (thr-eval)
    * f1_opt (max F1 on PR thresholds)
    * youden_j_opt (max Youden J on ROC thresholds)
    * recall_X for one or more recall targets (choose highest threshold achieving recall >= target)
- Saves all threshold metrics and thresholds to CSV/JSON
- Expands RF hyperparameter search space

Outputs (under --out-dir):
  - oof_predictions.csv                      (proba_* columns)
  - oof_thresholded_predictions.csv          (adds y_pred__<model>__<threshold_name>)
  - fold_ranking_metrics.csv                 (AUC/AUPRC per fold per model)
  - oof_ranking_metrics.csv                  (OOF AUC/AUPRC per model)
  - threshold_sweep_pr.csv                   (combined across models)
  - threshold_sweep_roc.csv                  (combined across models)
  - threshold_sweep_recall.csv               (combined across models & targets)
  - threshold_metrics_oof.csv                (OOF metrics per model per threshold)
  - threshold_metrics_by_fold.csv            (fold metrics per model per threshold)
  - final_performance_matrix.csv             (OOF metrics using youden_j_opt by default)
  - final_results.json                       (all summaries + saved model paths)
  - shap_*.csv (optional, same behavior as before)
  - models/*.joblib + models/models_manifest.json


Optional calibration:
- If enabled, probability calibration (sigmoid/isotonic) is fit using ONLY the outer-train data
  (via StratifiedKFold inside CalibratedClassifierCV) and applied to outer-test predictions.
- Calibrated probability streams are saved as proba_multi_early_cal / proba_multi_late_cal.

Notes:
- Thresholds are computed from out-of-sample OOF probabilities, separately per model/probability stream
  (e.g., uncalibrated vs calibrated), then evaluated globally and by fold.
- final_performance_matrix.csv reports one line per model using the global OOF-selected youden_j_opt threshold.

"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import joblib
from scipy.stats import randint, uniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression



from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

# SHAP optional
try:
    import shap  # type: ignore
except Exception:
    shap = None


# ----------------------------
# Defaults: file paths / columns
# ----------------------------

INPUT_DIR_DEFAULT = "/common/mcgoverndlab/usr/Miad/PSC/data/data_cleaned/imputed_data/"
PHENO_PATH_DEFAULT = "/common/mcgoverndlab/usr/Miad/PSC/data/data_cleaned/phenotype_data/data_phenotype_original.csv"

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


LABEL_COL = "PSC"
FID_COL = "FID"
MODALITY_SEED = {
    "lab": 11,
    "serology": 22,
    "genetics": 33,
    "clinical": 44,
}

# ----------------------------
# Helpers
# ----------------------------

def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def normalize_fid(fid) -> str:
    fid = str(fid).strip()
    if fid.isdigit():
        return str(int(fid))
    return fid


def safe_ohe():
    # scikit-learn changed OneHotEncoder arg from sparse -> sparse_output.
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def get_allocated_cpus_fallback(n_jobs_arg: Optional[int]) -> int:
    if n_jobs_arg is not None and n_jobs_arg > 0:
        return int(n_jobs_arg)
    env = os.environ
    for k in ["SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE", "SLURM_CPUS_ON_NODE"]:
        v = env.get(k)
        if v:
            try:
                v2 = str(v).split("(")[0].split("x")[0].strip()
                n = int(v2)
                if n > 0:
                    return n
            except Exception:
                pass
    return int(os.cpu_count() or 1)


def set_thread_env_for_hpc():
    # Prevent BLAS oversubscription when sklearn parallelizes.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def strip_feature_name(name: str) -> str:
    s = str(name)
    for pref in ("num__", "cat__", "remainder__"):
        if s.startswith(pref):
            s = s[len(pref):]
    if s.endswith(" EU"):
        s = s[:-3]
    return s


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


@dataclass
class RFSearchConfig:
    outer_folds: int = 10
    inner_folds: int = 5
    inner_iters: int = 30
    random_state: int = 42

def rf_param_space() -> Dict[str, object]:
    return {
        "clf__n_estimators": randint(50, 250),           # 50–250
        "clf__max_depth": randint(3, 11),               # 3–10
        "clf__min_samples_split": randint(2, 16),       # 2–15
        "clf__min_samples_leaf": randint(3, 21),        # 3–20
        "clf__max_features": ["sqrt", 0.5, 0.7],
        "clf__max_samples": uniform(0.6, 0.35),         # 0.6–0.95
    }

def build_preprocessor(df: pd.DataFrame, feature_cols: List[str]) -> ColumnTransformer:
    X = df[feature_cols]
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in feature_cols if c not in cat_cols]

    transformers = []
    if num_cols:
        transformers.append(("num", "passthrough", num_cols))
    if cat_cols:
        transformers.append(("cat", safe_ohe(), cat_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def rf_pipeline(df: pd.DataFrame, feature_cols: List[str], n_jobs: int, random_state: int) -> Pipeline:
    pre = build_preprocessor(df, feature_cols)
    clf = RandomForestClassifier(
        n_jobs=n_jobs,
        random_state=random_state,
        class_weight="balanced",
        bootstrap=True,
        max_features="sqrt",
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def fit_tuned_rf(
    df_train: pd.DataFrame,
    feature_cols: List[str],
    y_train: np.ndarray,
    n_jobs: int,
    cfg: RFSearchConfig,
    seed_offset: int = 0,
) -> Pipeline:
    pipe = rf_pipeline(df_train, feature_cols, n_jobs=n_jobs, random_state=cfg.random_state + seed_offset)
    inner_cv = StratifiedKFold(n_splits=cfg.inner_folds, shuffle=True, random_state=cfg.random_state + 100 + seed_offset)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=rf_param_space(),
        n_iter=cfg.inner_iters,
        scoring="average_precision",  # UPDATED: optimize AUPRC
        n_jobs=n_jobs,
        cv=inner_cv,
        random_state=cfg.random_state + 200 + seed_offset,
        verbose=0,
    )
    search.fit(df_train[feature_cols], y_train)
    return search.best_estimator_


# ----------------------------
# Threshold selection + metrics
# ----------------------------

def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> Dict[str, object]:
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    fpr = _safe_div(fp, fp + tn)
    tpr = rec
    specificity = _safe_div(tn, tn + fp)
    youden_j = tpr - fpr

    return {
        "threshold": float(thr),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "specificity": float(specificity),
        "fpr": float(fpr),
        "tpr": float(tpr),
        "youden_j": float(youden_j),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "pred_pos_rate": float(y_pred.mean()),
    }

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

def fit_prob_calibrator(p: np.ndarray, y: np.ndarray, method: str):
    """
    Fit a *probability-to-probability* calibrator: p -> p_cal
    method:
      - isotonic: IsotonicRegression(out_of_bounds='clip')
      - sigmoid : Platt scaling via LogisticRegression on p
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=int)

    if method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(p, y)
        return cal

    # sigmoid / Platt scaling
    lr = LogisticRegression(solver="lbfgs", max_iter=2000)
    lr.fit(p.reshape(-1, 1), y)
    return lr


def apply_prob_calibrator(cal, p: np.ndarray, method: str) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if method == "isotonic":
        return cal.predict(p)
    return cal.predict_proba(p.reshape(-1, 1))[:, 1]


def pick_threshold_f1_from_pr(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, pd.DataFrame]:
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    if len(thr) == 0:
        return 0.5, pd.DataFrame(columns=["threshold", "precision", "recall", "f1"])

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


def pick_threshold_youden_from_roc(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, pd.DataFrame]:
    fpr, tpr, thr = roc_curve(y_true, y_prob)
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
    y_prob: np.ndarray,
    target_recall: float,
) -> Tuple[float, pd.DataFrame]:
    """
    Choose the HIGHEST threshold that still achieves recall >= target_recall
    (minimize FPs while meeting recall constraint).
    """
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
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


def auc_ap_summary(name: str, fold_auc: List[float], fold_ap: List[float], y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, object]:
    fold_auc_arr = np.array(fold_auc, dtype=float)
    fold_ap_arr = np.array(fold_ap, dtype=float)
    return {
        "name": name,
        "fold_auc": [float(x) for x in fold_auc_arr],
        "fold_ap": [float(x) for x in fold_ap_arr],
        "fold_auc_mean": float(fold_auc_arr.mean()) if len(fold_auc_arr) else None,
        "fold_auc_sd": float(fold_auc_arr.std(ddof=1)) if len(fold_auc_arr) > 1 else None,
        "fold_ap_mean": float(fold_ap_arr.mean()) if len(fold_ap_arr) else None,
        "fold_ap_sd": float(fold_ap_arr.std(ddof=1)) if len(fold_ap_arr) > 1 else None,
        "oof_auc": float(roc_auc_score(y_true, y_prob)),
        "oof_auprc": float(average_precision_score(y_true, y_prob)),
        "n": int(len(y_true)),
    }


# ----------------------------
# SHAP (unchanged behavior)
# ----------------------------

def shap_mean_abs_for_pipeline(
    fitted_pipe: Pipeline,
    X_df: pd.DataFrame,
    max_samples: int,
    random_state: int = 0,
) -> Tuple[np.ndarray, List[str]]:
    if shap is None:
        return np.array([]), []
    if len(X_df) == 0:
        return np.array([]), []
    if max_samples is not None and len(X_df) > max_samples:
        X_df = X_df.sample(n=max_samples, random_state=random_state)

    pre = fitted_pipe.named_steps["pre"]
    clf = fitted_pipe.named_steps["clf"]

    X_mat = pre.transform(X_df)
    try:
        feat_names = pre.get_feature_names_out().tolist()
    except Exception:
        feat_names = [f"f{i}" for i in range(X_mat.shape[1])]

    feat_names = [strip_feature_name(x) for x in feat_names]

    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X_mat)

    if isinstance(shap_vals, list):
        sv = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
    else:
        sv = shap_vals
        if sv.ndim == 3 and sv.shape[-1] >= 2:
            sv = sv[:, :, 1]

    mean_abs = np.abs(sv).mean(axis=0)
    return mean_abs, feat_names


def shap_mean_abs_for_lr(lr_model: LogisticRegression, X_train: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    if shap is None:
        return np.array([]), []
    feat_names = X_train.columns.tolist()
    explainer = shap.LinearExplainer(
        lr_model,
        X_train,
        feature_perturbation="correlation_dependent",
    )
    sv = explainer.shap_values(X_train)
    if isinstance(sv, list):
        sv = sv[0]
    mean_abs = np.abs(sv).mean(axis=0)
    return mean_abs, feat_names


def map_probs_by_fid(fids: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    return {normalize_fid(f): float(p) for f, p in zip(fids, probs)}


# ----------------------------
# Data loading
# ----------------------------

def load_pheno(pheno_path: str) -> pd.DataFrame:
    ph = pd.read_csv(pheno_path)
    ph[FID_COL] = ph[FID_COL].apply(normalize_fid)
    # keep your exact filters
    ph = ph[ph["Race_Admix"].isin(["White"])].reset_index(drop=True)
    ph = ph[ph["Diagnosis"].isin(["CD", "UC"])].reset_index(drop=True)
    ph = ph.dropna(subset=[LABEL_COL]).reset_index(drop=True)
    return ph[[FID_COL, LABEL_COL]]


def load_modality_df(input_dir: str, modality: str) -> pd.DataFrame:
    fp = os.path.join(input_dir, MODALITY_FILES[modality])
    df = pd.read_csv(fp)
    df[FID_COL] = df[FID_COL].apply(normalize_fid)

    if modality == "serology":
        df = df[[FID_COL] + SEROL_COLS]
    elif modality == "clinical":
        df = df[[FID_COL] + CLIN_COLS]
    elif modality == "genetics":
        geno_cols = [c for c in df.columns if ":" in c]
        df = df[[FID_COL] + geno_cols]
    elif modality == "lab":
        lab_cols = [c for c in df.columns if c != FID_COL]
        df = df[[FID_COL] + lab_cols]
    else:
        raise ValueError(f"Unknown modality: {modality}")
    return df


def merge_with_label(df_mod: pd.DataFrame, pheno: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(df_mod, pheno, how="inner", on=FID_COL)


def build_intersection_dataset(
    input_dir: str,
    pheno: pd.DataFrame,
    modalities: List[str],
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, List[str]]]:
    """
    Returns:
      df_int: intersection cohort with ALL modality columns merged
      mod_dfs: expanded per-modality labeled dataframes
      mod_cols: per-modality feature columns list (excluding FID, label)
    """
    mod_dfs: Dict[str, pd.DataFrame] = {}
    mod_cols: Dict[str, List[str]] = {}

    for m in modalities:
        dfm = load_modality_df(input_dir, m)
        dfm_l = merge_with_label(dfm, pheno)
        mod_dfs[m] = dfm_l
        mod_cols[m] = [c for c in dfm_l.columns if c not in [FID_COL, LABEL_COL]]

    fids_sets = [set(mod_dfs[m][FID_COL].unique().tolist()) for m in modalities]
    fids_inter = set.intersection(*fids_sets) if fids_sets else set()

    dfs_int = []
    for m in modalities:
        dfs_int.append(mod_dfs[m][[FID_COL] + mod_cols[m]].copy())

    df_int = dfs_int[0]
    for d in dfs_int[1:]:
        df_int = pd.merge(df_int, d, how="inner", on=FID_COL)

    df_int = pd.merge(df_int, pheno, how="inner", on=FID_COL)
    df_int = df_int[df_int[FID_COL].isin(fids_inter)].drop_duplicates(subset=[FID_COL]).reset_index(drop=True)
    return df_int, mod_dfs, mod_cols


# ----------------------------
# Unified evaluation + saving final models
# ----------------------------

def run_all_models_one_pass(
    df_int: pd.DataFrame,
    mod_dfs_expanded: Dict[str, pd.DataFrame],
    mod_cols: Dict[str, List[str]],
    modalities: List[str],
    outer_cv: StratifiedKFold,
    n_jobs: int,
    cfg: RFSearchConfig,
    out_dir: str,
    shap_do: bool,
    shap_max_samples: int,
    do_calibration: bool,
    calib_method: str,
    calib_folds: int,
) -> Tuple[pd.DataFrame, Dict[str, object], Dict[str, object]]:

    """
    Returns:
      oof: table with probs
      results: dict with AUC/AUPRC summaries
      final_models_info: dict with paths/notes for saved final models
    """
    y = df_int[LABEL_COL].to_numpy()
    fids = df_int[FID_COL].to_numpy()

    all_feature_cols = [c for c in df_int.columns if c not in [FID_COL, LABEL_COL]]

    oof = pd.DataFrame({FID_COL: fids, "y_true": y})
    oof["fold_id"] = -1

    # proba columns
    for m in modalities:
        oof[f"proba_single_{m}"] = np.nan
        oof[f"proba_base_{m}"] = np.nan  # late base probs on intersection test
    oof["proba_multi_early"] = np.nan
    oof["proba_multi_late"] = np.nan
    oof["proba_multi_early_cal"] = np.nan
    oof["proba_multi_late_cal"] = np.nan

    # per-fold ranking metrics
    fold_auc: Dict[str, List[float]] = {f"single_{m}": [] for m in modalities}
    fold_ap: Dict[str, List[float]] = {f"single_{m}": [] for m in modalities}
    fold_auc["multi_early"] = []
    fold_auc["multi_late"] = []
    fold_ap["multi_early"] = []
    fold_ap["multi_late"] = []

    shap_acc_single = {m: {"sum": None, "names": None, "count": 0} for m in modalities}
    shap_acc_early = {"sum": None, "names": None, "count": 0}
    shap_acc_meta = {"sum": None, "names": None, "count": 0}

    print("\n=== RUNNING ALL MODELS WITH SHARED OUTER FOLDS (intersection test) ===")
    print("Inner CV scoring = average_precision (AUPRC/AP)")

    for fold_i, (tr_idx, te_idx) in enumerate(outer_cv.split(df_int, y), start=1):
        df_tr_int = df_int.iloc[tr_idx].reset_index(drop=True)
        df_te_int = df_int.iloc[te_idx].reset_index(drop=True)

        y_tr = df_tr_int[LABEL_COL].to_numpy()
        y_te = df_te_int[LABEL_COL].to_numpy()

        fids_te = set(df_te_int[FID_COL].tolist())
        oof.loc[te_idx, "fold_id"] = fold_i

        # A) Single-modality baselines (intersection-only)
        for m in modalities:
            feat_cols_m = mod_cols[m]
            model_single = fit_tuned_rf(
                df_train=df_tr_int,
                feature_cols=feat_cols_m,
                y_train=y_tr,
                n_jobs=n_jobs,
                cfg=cfg,
                seed_offset = 1000 * fold_i + MODALITY_SEED[m],
            )
            p_te_m = model_single.predict_proba(df_te_int[feat_cols_m])[:, 1]
            oof.loc[te_idx, f"proba_single_{m}"] = p_te_m

            auc_m = roc_auc_score(y_te, p_te_m)
            ap_m = average_precision_score(y_te, p_te_m)
            fold_auc[f"single_{m}"].append(float(auc_m))
            fold_ap[f"single_{m}"].append(float(ap_m))
            print(f"Fold {fold_i} | single[{m}] AUC={auc_m:.4f} | AP={ap_m:.4f}")

            if shap_do:
                mean_abs, feat_names = shap_mean_abs_for_pipeline(
                    fitted_pipe=model_single,
                    X_df=df_tr_int[feat_cols_m],
                    max_samples=shap_max_samples,
                    random_state=cfg.random_state + 1000 + fold_i,
                )
                if mean_abs.size > 0:
                    acc = shap_acc_single[m]
                    if acc["sum"] is None:
                        acc["sum"] = mean_abs.copy()
                        acc["names"] = feat_names
                    else:
                        if len(mean_abs) == len(acc["sum"]):
                            acc["sum"] += mean_abs
                    acc["count"] += 1

        # B) Early integration (intersection-only)
        model_early = fit_tuned_rf(
            df_train=df_tr_int,
            feature_cols=all_feature_cols,
            y_train=y_tr,
            n_jobs=1,
            cfg=cfg,
            seed_offset=2000 * fold_i,
        )
        p_te_early = model_early.predict_proba(df_te_int[all_feature_cols])[:, 1]
        oof.loc[te_idx, "proba_multi_early"] = p_te_early

        if do_calibration:
            cal_early = fit_calibrator_on_train(
                fitted_model=model_early,
                X_train=df_tr_int[all_feature_cols],
                y_train=y_tr,
                method=calib_method,
                n_splits=calib_folds,
                random_state=cfg.random_state + 7100 + fold_i,
                n_jobs=n_jobs,
            )
            p_te_early_cal = cal_early.predict_proba(df_te_int[all_feature_cols])[:, 1]
            oof.loc[te_idx, "proba_multi_early_cal"] = p_te_early_cal


        auc_early = roc_auc_score(y_te, p_te_early)
        ap_early = average_precision_score(y_te, p_te_early)
        fold_auc["multi_early"].append(float(auc_early))
        fold_ap["multi_early"].append(float(ap_early))
        print(f"Fold {fold_i} | multi[EARLY] AUC={auc_early:.4f} | AP={ap_early:.4f}")

        if shap_do:
            mean_abs, feat_names = shap_mean_abs_for_pipeline(
                fitted_pipe=model_early,
                X_df=df_tr_int[all_feature_cols],
                max_samples=shap_max_samples,
                random_state=cfg.random_state + 2000 + fold_i,
            )
            if mean_abs.size > 0:
                if shap_acc_early["sum"] is None:
                    shap_acc_early["sum"] = mean_abs.copy()
                    shap_acc_early["names"] = feat_names
                else:
                    if len(mean_abs) == len(shap_acc_early["sum"]):
                        shap_acc_early["sum"] += mean_abs
                shap_acc_early["count"] += 1

        # C) Late integration (expanded base training + LR meta)
        inner_cv = StratifiedKFold(
            n_splits=cfg.inner_folds,
            shuffle=True,
            random_state=cfg.random_state + fold_i,
        )

        meta_train = pd.DataFrame({FID_COL: df_tr_int[FID_COL].to_numpy(), "y_true": y_tr})
        for m in modalities:
            meta_train[f"p_{m}"] = np.nan

        # Inner loop to generate OOF base probs on intersection-train,
        # while base models are trained on expanded cohorts excluding:
        #   - outer test fids
        #   - inner val fids
        for inner_j, (_itr_idx, ival_idx) in enumerate(inner_cv.split(df_tr_int, y_tr), start=1):
            df_ival = df_tr_int.iloc[ival_idx].reset_index(drop=True)
            fids_ival = set(df_ival[FID_COL].tolist())

            for m in modalities:
                dfm_exp = mod_dfs_expanded[m]
                feat_cols_m = mod_cols[m]

                dfm_train = dfm_exp[~dfm_exp[FID_COL].isin(fids_te.union(fids_ival))].reset_index(drop=True)
                dfm_val = dfm_exp[dfm_exp[FID_COL].isin(fids_ival)].reset_index(drop=True)

                if len(dfm_val) == 0 or len(dfm_train) == 0:
                    continue

                y_train_m = dfm_train[LABEL_COL].to_numpy()

                base_model = fit_tuned_rf(
                    df_train=dfm_train,
                    feature_cols=feat_cols_m,
                    y_train=y_train_m,
                    n_jobs=n_jobs,
                    cfg=cfg,
                    seed_offset = 3000 * fold_i + 10 * inner_j + MODALITY_SEED[m],
                )

                p_val = base_model.predict_proba(dfm_val[feat_cols_m])[:, 1]
                pmap = map_probs_by_fid(dfm_val[FID_COL].to_numpy(), p_val)

                mask = meta_train[FID_COL].isin(pmap.keys())
                meta_train.loc[mask, f"p_{m}"] = meta_train.loc[mask, FID_COL].map(pmap).to_numpy()

        needed_cols = [f"p_{m}" for m in modalities]
        if meta_train[needed_cols].isna().any().any():
            bad = meta_train[meta_train[needed_cols].isna().any(axis=1)][FID_COL].head(10).tolist()
            raise RuntimeError(
                f"[Fold {fold_i}] Meta-train missing base probs for some rows. Example FIDs: {bad}"
            )

        X_meta_tr = meta_train[needed_cols].copy()
        y_meta_tr = meta_train["y_true"].to_numpy()

        meta_model = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="liblinear",
        )
        meta_model.fit(X_meta_tr, y_meta_tr)

        if shap_do:
            sv_mean, feat_names = shap_mean_abs_for_lr(meta_model, X_meta_tr)
            if sv_mean.size > 0:
                if shap_acc_meta["sum"] is None:
                    shap_acc_meta["sum"] = sv_mean.copy()
                    shap_acc_meta["names"] = feat_names
                else:
                    if len(sv_mean) == len(shap_acc_meta["sum"]):
                        shap_acc_meta["sum"] += sv_mean
                shap_acc_meta["count"] += 1

        # Fit base models on expanded training excluding outer test,
        # predict base probs on intersection test
        base_prob_maps_te: Dict[str, Dict[str, float]] = {}

        for m in modalities:
            dfm_exp = mod_dfs_expanded[m]
            feat_cols_m = mod_cols[m]

            dfm_train_full = dfm_exp[~dfm_exp[FID_COL].isin(fids_te)].reset_index(drop=True)
            dfm_test = dfm_exp[dfm_exp[FID_COL].isin(fids_te)].reset_index(drop=True)

            if len(dfm_test) == 0 or len(dfm_train_full) == 0:
                continue

            y_train_m = dfm_train_full[LABEL_COL].to_numpy()

            base_model_full = fit_tuned_rf(
                df_train=dfm_train_full,
                feature_cols=feat_cols_m,
                y_train=y_train_m,
                n_jobs=n_jobs,
                cfg=cfg,
                seed_offset = 4000 * fold_i + MODALITY_SEED[m],
            )

            p_te_m = base_model_full.predict_proba(dfm_test[feat_cols_m])[:, 1]
            base_prob_maps_te[m] = map_probs_by_fid(dfm_test[FID_COL].to_numpy(), p_te_m)

        missing = [m for m in modalities if m not in base_prob_maps_te]
        if missing:
            raise RuntimeError(
                f"[Fold {fold_i}] Missing base probabilities for modalities: {missing}"
            )

        te_fids_series = oof.loc[te_idx, FID_COL]
        for m in modalities:
            oof.loc[te_idx, f"proba_base_{m}"] = (
                te_fids_series.map(base_prob_maps_te[m]).to_numpy()
            )


        X_meta_te = pd.DataFrame({f"p_{m}": oof.loc[te_idx, f"proba_base_{m}"].to_numpy() for m in modalities})
        p_te_late = meta_model.predict_proba(X_meta_te)[:, 1]
        oof.loc[te_idx, "proba_multi_late"] = p_te_late

        if do_calibration:
            # Calibrate meta model on meta-train (outer-train only), then apply to meta-test
            cal_meta = fit_calibrator_on_train(
                fitted_model=meta_model,
                X_train=X_meta_tr,
                y_train=y_meta_tr,
                method=calib_method,
                n_splits=calib_folds,
                random_state=cfg.random_state + 7200 + fold_i,
                n_jobs=n_jobs,
            )
            p_te_late_cal = cal_meta.predict_proba(X_meta_te)[:, 1]
            oof.loc[te_idx, "proba_multi_late_cal"] = p_te_late_cal

        auc_late = roc_auc_score(y_te, p_te_late)
        ap_late = average_precision_score(y_te, p_te_late)
        fold_auc["multi_late"].append(float(auc_late))
        fold_ap["multi_late"].append(float(ap_late))
        print(f"Fold {fold_i} | multi[LATE ] AUC={auc_late:.4f} | AP={ap_late:.4f}")

    # Summaries
    results: Dict[str, object] = {}
    for m in modalities:
        name = f"single_{m}"
        results[name] = auc_ap_summary(
            name=name,
            fold_auc=fold_auc[name],
            fold_ap=fold_ap[name],
            y_true=oof["y_true"].to_numpy(),
            y_prob=oof[f"proba_single_{m}"].to_numpy(),
        )

    results["multi_early"] = auc_ap_summary(
        name="multi_early",
        fold_auc=fold_auc["multi_early"],
        fold_ap=fold_ap["multi_early"],
        y_true=oof["y_true"].to_numpy(),
        y_prob=oof["proba_multi_early"].to_numpy(),
    )
    results["multi_late"] = auc_ap_summary(
        name="multi_late",
        fold_auc=fold_auc["multi_late"],
        fold_ap=fold_ap["multi_late"],
        y_true=oof["y_true"].to_numpy(),
        y_prob=oof["proba_multi_late"].to_numpy(),
    )

    if do_calibration:
        results["multi_early_cal"] = auc_ap_summary(
            name="multi_early_cal",
            fold_auc=[],
            fold_ap=[],
            y_true=oof["y_true"].to_numpy(),
            y_prob=oof["proba_multi_early_cal"].to_numpy(),
        )
        results["multi_late_cal"] = auc_ap_summary(
            name="multi_late_cal",
            fold_auc=[],
            fold_ap=[],
            y_true=oof["y_true"].to_numpy(),
            y_prob=oof["proba_multi_late_cal"].to_numpy(),
        )



    # Save SHAP (optional)
    if shap_do:
        for m in modalities:
            acc = shap_acc_single[m]
            if acc["sum"] is not None and acc["count"] > 0:
                shap_mean = acc["sum"] / float(acc["count"])
                shap_df = pd.DataFrame({"feature": acc["names"], "mean_abs_shap": shap_mean})
                shap_df.sort_values("mean_abs_shap", ascending=False, inplace=True)
                shap_df.to_csv(os.path.join(out_dir, f"shap_single_{m}_intersection.csv"), index=False)

        if shap_acc_early["sum"] is not None and shap_acc_early["count"] > 0:
            shap_mean = shap_acc_early["sum"] / float(shap_acc_early["count"])
            shap_df = pd.DataFrame({"feature": shap_acc_early["names"], "mean_abs_shap": shap_mean})
            shap_df.sort_values("mean_abs_shap", ascending=False, inplace=True)
            shap_df.to_csv(os.path.join(out_dir, "shap_multi_early.csv"), index=False)

        if shap_acc_meta["sum"] is not None and shap_acc_meta["count"] > 0:
            shap_mean = shap_acc_meta["sum"] / float(shap_acc_meta["count"])
            shap_df = pd.DataFrame({"feature": shap_acc_meta["names"], "mean_abs_shap": shap_mean})
            shap_df.sort_values("mean_abs_shap", ascending=False, inplace=True)
            shap_df.to_csv(os.path.join(out_dir, "shap_multi_late_meta_lr.csv"), index=False)

    # -------------------------
    # FINAL MODEL SAVING (same as before, but tuning uses AUPRC now)
    # -------------------------
    models_dir = os.path.join(out_dir, "models")
    ensure_dir(models_dir)

    final_models_info: Dict[str, object] = {
        "saved": {},
        "notes": {},
        "created_at": now_str(),
        "modalities": modalities,
    }

    # 1) Final single-modality models (fit on ALL intersection)
    y_int_full = df_int[LABEL_COL].to_numpy()
    for m in modalities:
        feat_cols_m = mod_cols[m]
        model_single_final = fit_tuned_rf(
            df_train=df_int,
            feature_cols=feat_cols_m,
            y_train=y_int_full,
            n_jobs=n_jobs,
            cfg=cfg,
            seed_offset=9100 + MODALITY_SEED[m],
        )
        path = os.path.join(models_dir, f"single_{m}.joblib")
        joblib.dump(model_single_final, path)
        final_models_info["saved"][f"single_{m}"] = path
        final_models_info["notes"][f"single_{m}"] = "RF Pipeline(pre+rf) tuned(AUPRC)+fit on full intersection cohort."

    # 2) Final early integration model (fit on ALL intersection)
    all_feature_cols = [c for c in df_int.columns if c not in [FID_COL, LABEL_COL]]
    model_early_final = fit_tuned_rf(
        df_train=df_int,
        feature_cols=all_feature_cols,
        y_train=y_int_full,
        n_jobs=n_jobs,
        cfg=cfg,
        seed_offset=9200,
    )
    path_early = os.path.join(models_dir, "multi_early_rf.joblib")
    joblib.dump(model_early_final, path_early)
    final_models_info["saved"]["multi_early_rf"] = path_early
    final_models_info["notes"]["multi_early_rf"] = "RF Pipeline(pre+rf) tuned(AUPRC)+fit on full intersection cohort."

    # 3) Final late base models (fit on FULL expanded data per modality)
    base_probs_for_meta = pd.DataFrame({FID_COL: df_int[FID_COL].to_numpy()})
    for m in modalities:
        dfm_exp = mod_dfs_expanded[m]
        feat_cols_m = mod_cols[m]

        model_base_final = fit_tuned_rf(
            df_train=dfm_exp,
            feature_cols=feat_cols_m,
            y_train=dfm_exp[LABEL_COL].to_numpy(),
            n_jobs=n_jobs,
            cfg=cfg,
            seed_offset=9300 + MODALITY_SEED[m],
        )
        path_base = os.path.join(models_dir, f"multi_late_base_{m}.joblib")
        joblib.dump(model_base_final, path_base)
        final_models_info["saved"][f"multi_late_base_{m}"] = path_base
        final_models_info["notes"][f"multi_late_base_{m}"] = "RF Pipeline(pre+rf) tuned(AUPRC)+fit on full expanded cohort."

        # compute probs on intersection for meta training
        p_int = model_base_final.predict_proba(df_int[feat_cols_m])[:, 1]
        base_probs_for_meta[f"p_{m}"] = p_int

    # 4) Final meta LR (fit on intersection using base probs from final base models)
    X_meta = base_probs_for_meta[[f"p_{m}" for m in modalities]].copy()
    y_meta = y_int_full

    meta_final = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        solver="liblinear",
    )
    meta_final.fit(X_meta, y_meta)

    path_meta = os.path.join(models_dir, "multi_late_meta_lr.joblib")
    joblib.dump(meta_final, path_meta)
    final_models_info["saved"]["multi_late_meta_lr"] = path_meta
    final_models_info["notes"]["multi_late_meta_lr"] = (
        "Meta LR fit on full intersection using base probabilities from final expanded base pipelines."
    )

    manifest_path = os.path.join(models_dir, "models_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(final_models_info, f, indent=2)

    return oof, results, final_models_info


# ----------------------------
# Thresholding over OOF predictions for ALL models
# ----------------------------

def build_thresholds_and_metrics(
    oof: pd.DataFrame,
    model_prob_cols: Dict[str, str],
    thr_eval: float,
    target_recalls: List[float],
    out_dir: str,
) -> Dict[str, object]:
    """
    For each model (name -> proba column), compute:
      - OOF AUC/AUPRC
      - thresholds: fixed, f1_opt, youden_j_opt, recall_X
      - OOF metrics at each threshold
      - curve sweeps saved (combined across models)
    """
    y = oof["y_true"].to_numpy().astype(int)

    # combined sweeps
    pr_sweeps = []
    roc_sweeps = []
    recall_sweeps = []

    thr_metrics_rows = []
    oof_rank_rows = []

    thresholds_by_model: Dict[str, Dict[str, float]] = {}

    for model_name, prob_col in model_prob_cols.items():
        p = oof[prob_col].to_numpy().astype(float)

        # ranking metrics
        oof_auc = float(roc_auc_score(y, p))
        oof_ap = float(average_precision_score(y, p))
        oof_rank_rows.append({"model": model_name, "oof_auc": oof_auc, "oof_auprc": oof_ap})

        # sweeps for this model
        thr_f1, sweep_pr = pick_threshold_f1_from_pr(y, p)
        thr_j, sweep_roc = pick_threshold_youden_from_roc(y, p)

        sweep_pr = sweep_pr.copy()
        sweep_pr.insert(0, "model", model_name)
        pr_sweeps.append(sweep_pr)

        sweep_roc = sweep_roc.copy()
        sweep_roc.insert(0, "model", model_name)
        roc_sweeps.append(sweep_roc)

        thr_set: Dict[str, float] = {
            "fixed": float(thr_eval),
            "f1_opt": float(thr_f1),
            "youden_j_opt": float(thr_j),
        }

        # recall targets
        for r in sorted(set(float(x) for x in target_recalls), reverse=True):
            thr_r, sweep_r = pick_threshold_for_recall(y, p, target_recall=r)
            key = f"recall_{r:.2f}"
            thr_set[key] = float(thr_r)
            sweep_r = sweep_r.copy()
            sweep_r.insert(0, "model", model_name)
            recall_sweeps.append(sweep_r)

        thresholds_by_model[model_name] = thr_set

        # metrics at selected thresholds
        for thr_name, thr in thr_set.items():
            m = metrics_at_threshold(y, p, thr)
            thr_metrics_rows.append({
                "model": model_name,
                "threshold_name": thr_name,
                **m,
                "oof_auc": oof_auc,
                "oof_auprc": oof_ap,
            })

    # save combined sweeps
    if pr_sweeps:
        pd.concat(pr_sweeps, ignore_index=True).to_csv(os.path.join(out_dir, "threshold_sweep_pr.csv"), index=False)
    if roc_sweeps:
        pd.concat(roc_sweeps, ignore_index=True).to_csv(os.path.join(out_dir, "threshold_sweep_roc.csv"), index=False)
    if recall_sweeps:
        pd.concat(recall_sweeps, ignore_index=True).to_csv(os.path.join(out_dir, "threshold_sweep_recall.csv"), index=False)

    # save OOF ranking metrics
    oof_rank_df = pd.DataFrame(oof_rank_rows).sort_values("model")
    oof_rank_df.to_csv(os.path.join(out_dir, "oof_ranking_metrics.csv"), index=False)

    # save OOF threshold metrics
    thr_metrics_df = pd.DataFrame(thr_metrics_rows).sort_values(["model", "threshold_name"])
    thr_metrics_df.to_csv(os.path.join(out_dir, "threshold_metrics_oof.csv"), index=False)

    return {
        "thresholds_by_model": thresholds_by_model,
        "oof_ranking": oof_rank_rows,
        "oof_threshold_metrics": thr_metrics_rows,
    }


def build_fold_threshold_metrics(
    oof: pd.DataFrame,
    model_prob_cols: Dict[str, str],
    thresholds_by_model: Dict[str, Dict[str, float]],
    out_dir: str,
) -> None:
    """
    Evaluate thresholded metrics per outer fold, using the global OOF-chosen thresholds.
    Saves: threshold_metrics_by_fold.csv
    """
    rows = []
    for fold_id in sorted(oof["fold_id"].unique().tolist()):
        if fold_id == -1:
            continue
        df_fold = oof[oof["fold_id"] == fold_id].copy()
        y = df_fold["y_true"].to_numpy().astype(int)

        for model_name, prob_col in model_prob_cols.items():
            p = df_fold[prob_col].to_numpy().astype(float)
            for thr_name, thr in thresholds_by_model[model_name].items():
                m = metrics_at_threshold(y, p, thr)
                rows.append({
                    "fold_id": int(fold_id),
                    "model": model_name,
                    "threshold_name": thr_name,
                    **m,
                })

    out = pd.DataFrame(rows).sort_values(["model", "threshold_name", "fold_id"])
    out.to_csv(os.path.join(out_dir, "threshold_metrics_by_fold.csv"), index=False)


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=INPUT_DIR_DEFAULT)
    parser.add_argument("--pheno-path", default=PHENO_PATH_DEFAULT)
    parser.add_argument("--out-dir", required=True)

    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--outer-folds", type=int, default=10)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--inner-iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)


    # calibration controls
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument("--calibration-method", type=str, default="sigmoid", choices=["sigmoid", "isotonic"])
    parser.add_argument("--calibration-folds", type=int, default=5)
    parser.add_argument("--save-calibrators", action="store_true",
                    help="Fit calibrators on OOF (EI/LI) and save under models/")
    parser.add_argument("--calibration-clip", type=float, default=1e-6,
                    help="Clip probs to [clip, 1-clip] before fitting calibrators.")

    parser.add_argument("--no-shap", action="store_true")
    parser.add_argument("--shap-max-samples", type=int, default=500)

    # threshold controls
    parser.add_argument("--thr-eval", type=float, default=0.50)
    parser.add_argument("--target-recall", type=float, nargs="+", default=[0.90, 0.85, 0.80])

    args = parser.parse_args()
    do_calibration = (not args.no_calibration)

    set_thread_env_for_hpc()
    ensure_dir(args.out_dir)

    n_jobs = get_allocated_cpus_fallback(args.n_jobs)
    cfg = RFSearchConfig(
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
        inner_iters=args.inner_iters,
        random_state=args.seed,
    )
    shap_do = (not args.no_shap)

    target_recalls = sorted({float(x) for x in args.target_recall}, reverse=True)

    print(f"Saving to: {args.out_dir}")
    print(f"Using n_jobs = {n_jobs}")
    print(f"Outer folds = {cfg.outer_folds}, Inner folds = {cfg.inner_folds}, Inner iters = {cfg.inner_iters}")
    print("Inner CV scoring = average_precision (AUPRC/AP)")
    print(f"Thresholding: fixed={args.thr_eval} | recall targets={', '.join(f'{r:.2f}' for r in target_recalls)}")
    print(f"SHAP enabled: {shap_do} (max samples per fold={args.shap_max_samples})")
    print(f"Start: {now_str()}\n")

    pheno = load_pheno(args.pheno_path)
    modalities = ["genetics", "lab", "serology", "clinical"]

    df_int, mod_dfs_exp, mod_cols = build_intersection_dataset(args.input_dir, pheno, modalities)
    outer_cv = StratifiedKFold(n_splits=cfg.outer_folds, shuffle=True, random_state=cfg.random_state)

    oof, results, final_models_info = run_all_models_one_pass(
        df_int=df_int,
        mod_dfs_expanded=mod_dfs_exp,
        mod_cols=mod_cols,
        modalities=modalities,
        outer_cv=outer_cv,
        n_jobs=n_jobs,
        cfg=cfg,
        out_dir=args.out_dir,
        shap_do=shap_do,
        shap_max_samples=args.shap_max_samples,
        do_calibration=do_calibration,
        calib_method=args.calibration_method,
        calib_folds=args.calibration_folds,
    )


    # Save OOF predictions (probabilities)
    oof_path = os.path.join(args.out_dir, "oof_predictions.csv")
    oof.to_csv(oof_path, index=False)


    # ------------------------------------------------------------
    # Save deployable calibrators (OOF probability mapping)
    # ------------------------------------------------------------
    if args.save_calibrators:
        models_dir = os.path.join(args.out_dir, "models")
        ensure_dir(models_dir)

        y_oof = oof["y_true"].astype(int).to_numpy()

        # only multi-modal models
        p_ei = oof["proba_multi_early"].astype(float).to_numpy()
        p_li = oof["proba_multi_late"].astype(float).to_numpy()

        # keep valid rows
        m_ei = np.isfinite(p_ei) & np.isfinite(y_oof)
        m_li = np.isfinite(p_li) & np.isfinite(y_oof)

        p_ei = np.clip(
            p_ei[m_ei],
            args.calibration_clip,
            1.0 - args.calibration_clip,
        )
        p_li = np.clip(
            p_li[m_li],
            args.calibration_clip,
            1.0 - args.calibration_clip,
        )

        y_ei = y_oof[m_ei]
        y_li = y_oof[m_li]

        if len(np.unique(y_ei)) < 2 or len(np.unique(y_li)) < 2:
            print(
                "WARNING: Cannot fit OOF calibrators (need both classes). "
                "Skipping saving calibrators."
            )
        else:
            cal_ei = fit_prob_calibrator(
                p_ei,
                y_ei,
                method=args.calibration_method,
            )
            cal_li = fit_prob_calibrator(
                p_li,
                y_li,
                method=args.calibration_method,
            )

            cal_early_path = os.path.join(
                models_dir,
                "calibrator_multi_early.joblib",
            )
            cal_late_path = os.path.join(
                models_dir,
                "calibrator_multi_late.joblib",
            )

            joblib.dump(cal_ei, cal_early_path)
            joblib.dump(cal_li, cal_late_path)

            meta = {
                "kind": "probability_mapping",
                "method": args.calibration_method,
                "clip": float(args.calibration_clip),
                "n_oof": int(len(oof)),
                "n_pos": int(y_oof.sum()),
                "prevalence": float(y_oof.mean()),
                "paths": {
                    "calibrator_multi_early": cal_early_path,
                    "calibrator_multi_late": cal_late_path,
                },
            }

            with open(
                os.path.join(models_dir, "calibrators_meta.json"),
                "w",
            ) as f:
                json.dump(meta, f, indent=2)

            print("Saved deployable OOF calibrators:")
            print(f"  - {cal_early_path}")
            print(f"  - {cal_late_path}")

    # Fold ranking metrics (AUC/AP) saved from `results` + fold lists are inside results;
    # but we also save a per-fold table directly from the OOF file.
    # (Fold-level AUC/AP computed per model per fold)
    fold_rows = []
    y_all = oof["y_true"].to_numpy().astype(int)
    for fold_id in sorted(oof["fold_id"].unique().tolist()):
        if fold_id == -1:
            continue
        df_fold = oof[oof["fold_id"] == fold_id]
        y = df_fold["y_true"].to_numpy().astype(int)
        for model_name, prob_col in {
            "multi_late": "proba_multi_late",
            "multi_early": "proba_multi_early",
            "single_serology": "proba_single_serology",
            "single_lab": "proba_single_lab",
            "single_clinical": "proba_single_clinical",
            "single_genetics": "proba_single_genetics",
        }.items():
            p = df_fold[prob_col].to_numpy().astype(float)
            fold_rows.append({
                "fold_id": int(fold_id),
                "model": model_name,
                "auc": float(roc_auc_score(y, p)),
                "auprc": float(average_precision_score(y, p)),
                "n": int(len(y)),
                "n_pos": int(y.sum()),
            })
    pd.DataFrame(fold_rows).sort_values(["model", "fold_id"]).to_csv(
        os.path.join(args.out_dir, "fold_ranking_metrics.csv"), index=False
    )

    # Build thresholds + OOF threshold metrics for all models
    model_prob_cols = {
        "multi_late": "proba_multi_late",
        "multi_early": "proba_multi_early",
        "single_serology": "proba_single_serology",
        "single_lab": "proba_single_lab",
        "single_clinical": "proba_single_clinical",
        "single_genetics": "proba_single_genetics",
    }
    if do_calibration:
        model_prob_cols.update({
            "multi_late_cal": "proba_multi_late_cal",
            "multi_early_cal": "proba_multi_early_cal",
        })

    thr_bundle = build_thresholds_and_metrics(
        oof=oof,
        model_prob_cols=model_prob_cols,
        thr_eval=args.thr_eval,
        target_recalls=target_recalls,
        out_dir=args.out_dir,
    )
    thresholds_by_model = thr_bundle["thresholds_by_model"]

    # Fold-level threshold metrics
    build_fold_threshold_metrics(
        oof=oof,
        model_prob_cols=model_prob_cols,
        thresholds_by_model=thresholds_by_model,
        out_dir=args.out_dir,
    )

    # Save thresholded predictions (adds y_pred__model__thrname columns)
    df_thr = oof[[FID_COL, "fold_id", "y_true"] + list(model_prob_cols.values())].copy()
    for model_name, prob_col in model_prob_cols.items():
        p = df_thr[prob_col].to_numpy().astype(float)
        for thr_name, thr in thresholds_by_model[model_name].items():
            df_thr[f"y_pred__{model_name}__{thr_name}"] = (p >= thr).astype(int)
    df_thr.to_csv(os.path.join(args.out_dir, "oof_thresholded_predictions.csv"), index=False)

    # Final performance matrix (OOF-based) using youden_j_opt by default
    perf_rows = []
    for model_name, prob_col in model_prob_cols.items():
        p = oof[prob_col].to_numpy().astype(float)
        thr = thresholds_by_model[model_name]["youden_j_opt"]
        mm = metrics_at_threshold(y_all, p, thr)
        perf_rows.append({
            "Model": model_name,
            "Threshold": float(thr),
            "AUC": float(roc_auc_score(y_all, p)),
            "AUPRC": float(average_precision_score(y_all, p)),
            "Precision": float(mm["precision"]),
            "Recall": float(mm["recall"]),
            "F1": float(mm["f1"]),
            "Balanced Accuracy": float(mm["balanced_accuracy"]),
            "Pred_Pos_Rate": float(mm["pred_pos_rate"]),
            "TP": int(mm["tp"]), "FP": int(mm["fp"]), "FN": int(mm["fn"]), "TN": int(mm["tn"]),
        })
    metrics_df = pd.DataFrame(perf_rows).sort_values("Model")

    out_csv = os.path.join(args.out_dir, "final_performance_matrix.csv")
    metrics_df.to_csv(out_csv, index=False)

    print("\n==============================")
    print("FINAL OOF PERFORMANCE MATRIX (Youden-J threshold)")
    print("==============================")
    show_cols = ["Model", "AUC", "AUPRC", "Precision", "Recall", "F1", "Pred_Pos_Rate", "Threshold"]
    print(metrics_df[show_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved performance matrix -> {out_csv}")

    # Save final results JSON
    final = {
        "run_info": {
            "start_time": now_str(),
            "outer_folds": cfg.outer_folds,
            "inner_folds": cfg.inner_folds,
            "inner_iters": cfg.inner_iters,
            "seed": cfg.random_state,
            "n_jobs": n_jobs,
            "n_intersection": int(len(oof)),
            "modalities": modalities,
            "scoring_inner_cv": "average_precision",
            "thresholding": {
                "thr_eval_fixed": float(args.thr_eval),
                "target_recalls": [float(x) for x in target_recalls],
                "thresholds_by_model": thresholds_by_model,
            },
            "note": (
                "Same outer folds for single/early/late; late base models trained on expanded cohorts "
                "excluding outer test FIDs; meta LR trained on intersection-train using inner OOF base probs."
            ),
        },
        "models": results,
        "saved_models": final_models_info,
    }

    out_json = os.path.join(args.out_dir, "final_results.json")
    with open(out_json, "w") as f:
        json.dump(final, f, indent=2)

    print("\n==============================")
    print("FAIR COMPARISON (same outer test folds on intersection)")
    print("==============================")
    for k in ["single_genetics", "single_lab", "single_serology", "single_clinical", "multi_early", "multi_late"]:
        print(
            f"{k:>16s} | OOF AUC={final['models'][k]['oof_auc']:.4f} "
            f"| OOF AUPRC={final['models'][k]['oof_auprc']:.4f} "
            f"(mean fold AUC={final['models'][k]['fold_auc_mean']:.4f}, AP={final['models'][k]['fold_ap_mean']:.4f})"
        )

    print(f"\nSaved OOF predictions            -> {oof_path}")
    print(f"Saved OOF thresholded preds      -> {os.path.join(args.out_dir, 'oof_thresholded_predictions.csv')}")
    print(f"Saved OOF threshold metrics      -> {os.path.join(args.out_dir, 'threshold_metrics_oof.csv')}")
    print(f"Saved fold threshold metrics     -> {os.path.join(args.out_dir, 'threshold_metrics_by_fold.csv')}")
    print(f"Saved threshold sweeps (PR/ROC)  -> {args.out_dir}")
    print(f"Saved final results              -> {out_json}")
    print(f"Saved final models               -> {os.path.join(args.out_dir, 'models')}")
    print(f"End: {now_str()}")
    print("execution done!")


if __name__ == "__main__":
    main()

