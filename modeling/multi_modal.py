#!/usr/bin/env python3
"""
multi_modal.py

Shared-outer-CV evaluation on the INTERSECTION cohort (subjects with ALL modalities),
with fair AUC comparison across:
  1) Single-modality RF baselines (trained/evaluated on intersection folds)
  2) Early integration RF (concatenated features, intersection only)
  3) Late integration (stacking):
       - Base RFs trained on EXPANDED modality cohorts (excluding outer test FIDs per fold)
       - Meta LogisticRegression trained on intersection-train using INNER OOF base probs
       - Evaluated on intersection test fold

This version fixes the "feature names mismatch" issue by ensuring that ALL saved models
(single, early, and late base) are sklearn Pipelines that include preprocessing
(ColumnTransformer + OneHotEncoder). Therefore prediction can pass RAW columns and
the model will handle one-hot encoding consistently.

It also saves final refit models under --out-dir/models:
  * single_{modality}.joblib           (RF pipeline, tuned+fit on full intersection)
  * multi_early_rf.joblib              (RF pipeline, tuned+fit on full intersection)
  * multi_late_base_{modality}.joblib  (RF pipeline, tuned+fit on full expanded cohort)
  * multi_late_meta_lr.joblib          (LR meta model, fit on intersection using base probs)
  * models/models_manifest.json

Outputs (under --out-dir):
  - oof_predictions.csv
  - final_results.json
  - final_performance_matrix.csv
  - shap_*.csv (optional)
  - models/*.joblib

Notes:
- This script is for evaluation + saving reusable models for downstream scoring.
- For late-integration meta LR final fit, we train on base probabilities produced by
  the final base models on the intersection cohort (deterministic approximation).
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

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    balanced_accuracy_score,
)

from scipy.stats import randint
import joblib

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


@dataclass
class RFSearchConfig:
    outer_folds: int = 10
    inner_folds: int = 5
    inner_iters: int = 25
    random_state: int = 42


def rf_param_space() -> Dict[str, object]:
    # Keep runtime sane but tune enough to matter.
    return {
        "clf__n_estimators": randint(20, 120),
        "clf__max_depth": randint(2, 10),
        "clf__min_samples_split": randint(2, 10),
        "clf__min_samples_leaf": randint(1, 8),
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
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def fit_tuned_rf(
    df_train: pd.DataFrame,
    feature_cols: List[str],
    y_train: np.ndarray,
    n_jobs: int,
    cfg: RFSearchConfig,
) -> Pipeline:
    pipe = rf_pipeline(df_train, feature_cols, n_jobs=n_jobs, random_state=cfg.random_state)
    inner_cv = StratifiedKFold(n_splits=cfg.inner_folds, shuffle=True, random_state=cfg.random_state)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=rf_param_space(),
        n_iter=cfg.inner_iters,
        scoring="roc_auc",
        n_jobs=n_jobs,
        cv=inner_cv,
        random_state=cfg.random_state,
        verbose=0,
    )
    search.fit(df_train[feature_cols], y_train)
    return search.best_estimator_


def optimal_threshold_youden(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thresholds[idx])


def compute_metrics_row(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, object]:
    thr = optimal_threshold_youden(y_true, y_prob)
    y_pred = (y_prob >= thr).astype(int)

    return {
        "Model": name,
        "Threshold": float(thr),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "AUC": float(roc_auc_score(y_true, y_prob)),
    }


def auc_summary(name: str, fold_aucs: List[float], y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, object]:
    fold_aucs_arr = np.array(fold_aucs, dtype=float)
    return {
        "name": name,
        "fold_aucs": [float(x) for x in fold_aucs_arr],
        "fold_mean": float(fold_aucs_arr.mean()) if len(fold_aucs_arr) else None,
        "fold_sd": float(fold_aucs_arr.std(ddof=1)) if len(fold_aucs_arr) > 1 else None,
        "oof_auc": float(roc_auc_score(y_true, y_prob)),
        "n": int(len(y_true)),
    }


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
) -> Tuple[pd.DataFrame, Dict[str, object], Dict[str, object]]:
    """
    Returns:
      oof: table with probs
      results: dict with AUC summaries
      final_models_info: dict with paths/notes for saved final models
    """
    y = df_int[LABEL_COL].to_numpy()
    fids = df_int[FID_COL].to_numpy()

    all_feature_cols = [c for c in df_int.columns if c not in [FID_COL, LABEL_COL]]

    oof = pd.DataFrame({FID_COL: fids, "y_true": y})
    oof["fold_id"] = -1

    for m in modalities:
        oof[f"proba_single_{m}"] = np.nan
    oof["proba_multi_early"] = np.nan
    oof["proba_multi_late"] = np.nan
    for m in modalities:
        oof[f"proba_base_{m}"] = np.nan

    fold_aucs = {f"single_{m}": [] for m in modalities}
    fold_aucs["multi_early"] = []
    fold_aucs["multi_late"] = []

    shap_acc_single = {m: {"sum": None, "names": None, "count": 0} for m in modalities}
    shap_acc_early = {"sum": None, "names": None, "count": 0}
    shap_acc_meta = {"sum": None, "names": None, "count": 0}

    print("\n=== RUNNING ALL MODELS WITH SHARED OUTER FOLDS (intersection test) ===")

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
            )
            p_te_m = model_single.predict_proba(df_te_int[feat_cols_m])[:, 1]
            oof.loc[te_idx, f"proba_single_{m}"] = p_te_m

            auc_m = roc_auc_score(y_te, p_te_m)
            fold_aucs[f"single_{m}"].append(float(auc_m))
            print(f"Fold {fold_i} | single[{m}] AUC={auc_m:.4f}")

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
            n_jobs=n_jobs,
            cfg=cfg,
        )
        p_te_early = model_early.predict_proba(df_te_int[all_feature_cols])[:, 1]
        oof.loc[te_idx, "proba_multi_early"] = p_te_early

        auc_early = roc_auc_score(y_te, p_te_early)
        fold_aucs["multi_early"].append(float(auc_early))
        print(f"Fold {fold_i} | multi[EARLY] AUC={auc_early:.4f}")

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
        for _inner_j, (_itr_idx, ival_idx) in enumerate(inner_cv.split(df_tr_int, y_tr), start=1):
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
            max_iter=2000,
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
            )

            p_te_m = base_model_full.predict_proba(dfm_test[feat_cols_m])[:, 1]
            base_prob_maps_te[m] = map_probs_by_fid(dfm_test[FID_COL].to_numpy(), p_te_m)

        te_fids_series = oof.loc[te_idx, FID_COL]
        for m in modalities:
            oof.loc[te_idx, f"proba_base_{m}"] = te_fids_series.map(base_prob_maps_te[m]).to_numpy()

        X_meta_te = pd.DataFrame({f"p_{m}": oof.loc[te_idx, f"proba_base_{m}"].to_numpy() for m in modalities})
        p_te_late = meta_model.predict_proba(X_meta_te)[:, 1]
        oof.loc[te_idx, "proba_multi_late"] = p_te_late

        auc_late = roc_auc_score(y_te, p_te_late)
        fold_aucs["multi_late"].append(float(auc_late))
        print(f"Fold {fold_i} | multi[LATE ] AUC={auc_late:.4f}")

    # Summaries
    results: Dict[str, object] = {}
    for m in modalities:
        name = f"single_{m}"
        results[name] = auc_summary(
            name=name,
            fold_aucs=fold_aucs[name],
            y_true=oof["y_true"].to_numpy(),
            y_prob=oof[f"proba_single_{m}"].to_numpy(),
        )

    results["multi_early"] = auc_summary(
        name="multi_early",
        fold_aucs=fold_aucs["multi_early"],
        y_true=oof["y_true"].to_numpy(),
        y_prob=oof["proba_multi_early"].to_numpy(),
    )
    results["multi_late"] = auc_summary(
        name="multi_late",
        fold_aucs=fold_aucs["multi_late"],
        y_true=oof["y_true"].to_numpy(),
        y_prob=oof["proba_multi_late"].to_numpy(),
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
    # FINAL MODEL SAVING
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
        )
        path = os.path.join(models_dir, f"single_{m}.joblib")
        joblib.dump(model_single_final, path)
        final_models_info["saved"][f"single_{m}"] = path
        final_models_info["notes"][f"single_{m}"] = "RF Pipeline(pre+rf) tuned+fit on full intersection cohort."

    # 2) Final early integration model (fit on ALL intersection)
    all_feature_cols = [c for c in df_int.columns if c not in [FID_COL, LABEL_COL]]
    model_early_final = fit_tuned_rf(
        df_train=df_int,
        feature_cols=all_feature_cols,
        y_train=y_int_full,
        n_jobs=n_jobs,
        cfg=cfg,
    )
    path_early = os.path.join(models_dir, "multi_early_rf.joblib")
    joblib.dump(model_early_final, path_early)
    final_models_info["saved"]["multi_early_rf"] = path_early
    final_models_info["notes"]["multi_early_rf"] = "RF Pipeline(pre+rf) tuned+fit on full intersection cohort."

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
        )
        path_base = os.path.join(models_dir, f"multi_late_base_{m}.joblib")
        joblib.dump(model_base_final, path_base)
        final_models_info["saved"][f"multi_late_base_{m}"] = path_base
        final_models_info["notes"][f"multi_late_base_{m}"] = "RF Pipeline(pre+rf) tuned+fit on full expanded cohort."

        # compute probs on intersection for meta training
        p_int = model_base_final.predict_proba(df_int[[FID_COL] + feat_cols_m][feat_cols_m])[:, 1]
        base_probs_for_meta[f"p_{m}"] = p_int

    # 4) Final meta LR (fit on intersection using base probs from final base models)
    X_meta = base_probs_for_meta[[f"p_{m}" for m in modalities]].copy()
    y_meta = y_int_full

    meta_final = LogisticRegression(
        max_iter=2000,
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
    parser.add_argument("--inner-iters", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--no-shap", action="store_true")
    parser.add_argument("--shap-max-samples", type=int, default=500)

    args = parser.parse_args()

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

    print(f"Saving to: {args.out_dir}")
    print(f"Using n_jobs = {n_jobs}")
    print(f"Outer folds = {cfg.outer_folds}, Inner folds = {cfg.inner_folds}, Inner iters = {cfg.inner_iters}")
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
    )

    # Save OOF predictions
    oof_path = os.path.join(args.out_dir, "oof_predictions.csv")
    oof.to_csv(oof_path, index=False)

    # Final performance matrix (OOF-based)
    y_true = oof["y_true"].to_numpy()
    rows = [
        compute_metrics_row("Multi-modality (LI)", y_true, oof["proba_multi_late"].to_numpy()),
        compute_metrics_row("Multi-modality (EI)", y_true, oof["proba_multi_early"].to_numpy()),
        compute_metrics_row("Serology", y_true, oof["proba_single_serology"].to_numpy()),
        compute_metrics_row("Lab", y_true, oof["proba_single_lab"].to_numpy()),
        compute_metrics_row("Clinical", y_true, oof["proba_single_clinical"].to_numpy()),
        compute_metrics_row("Genetics", y_true, oof["proba_single_genetics"].to_numpy()),
    ]
    metrics_df = pd.DataFrame(rows)

    print("\n==============================")
    print("FINAL OOF PERFORMANCE MATRIX")
    print("==============================")
    print(
        metrics_df[["Model", "Precision", "Recall", "Balanced Accuracy", "AUC"]]
        .to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )

    out_csv = os.path.join(args.out_dir, "final_performance_matrix.csv")
    metrics_df.to_csv(out_csv, index=False)
    print(f"\nSaved performance matrix -> {out_csv}")

    # Save final results
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
        print(f"{k:>16s} | OOF AUC = {final['models'][k]['oof_auc']:.4f} "
              f"(mean fold {final['models'][k]['fold_mean']:.4f})")

    print(f"\nSaved OOF predictions -> {oof_path}")
    print(f"Saved final results    -> {out_json}")
    print(f"Saved final models     -> {os.path.join(args.out_dir, 'models')}")
    print(f"End: {now_str()}")
    print("execution done!")


if __name__ == "__main__":
    main()

