#!/usr/bin/env python3
"""
single_modal.py

Single-modality Random Forest modeling with nested cross-validation.
Saves a *Pipeline* (preprocess + RF) so prediction is consistent and does not
require manually recreating one-hot columns.

Outputs (per modality out_dir):
  - best_model.joblib                (sklearn Pipeline: preprocessor + RF)
  - nestedcv_oof_predictions.csv
  - nestedcv_roc_curve.csv
  - nestedcv_summary.json
  - shap_global_importance.csv       (mean_abs_shap, mean_shap)
  - training_columns.json            (raw columns used at fit/predict time)

Supported modalities:
  - genetics
  - lab
  - serology
  - clinical
"""

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
from sklearn.metrics import roc_auc_score, roc_curve

import shap


# =========================
# Utilities
# =========================

def get_n_jobs(n_jobs_arg: Optional[int] = None) -> int:
    if n_jobs_arg is not None and n_jobs_arg > 0:
        return int(n_jobs_arg)
    return int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))


def normalize_fid(fid) -> str:
    fid = str(fid).strip()
    return str(int(fid)) if fid.isdigit() else fid


def safe_ohe():
    # scikit-learn changed OneHotEncoder arg from sparse -> sparse_output
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def most_common_params(params_list: List[Dict]) -> Dict:
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
    # Tune RF inside the pipeline
    return {
        "clf__n_estimators": randint(50, 300),
        "clf__max_depth": [None] + list(range(2, 15)),
        "clf__min_samples_split": randint(2, 20),
        "clf__min_samples_leaf": randint(1, 15),
        "clf__max_features": ["sqrt", "log2", 0.3, 0.5],
    }


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
# Main runner
# =========================

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
):

    ensure_dir(out_dir)

    print(f"\n=== Running single modality: {modality} ===")
    print(f"Saving outputs to: {out_dir}")
    print(f"Using n_jobs = {n_jobs}")

    # -------------------------
    # Load modality data
    # -------------------------
    if modality == "genetics":
        df = pd.read_csv(os.path.join(input_dir, "data_genetics_only_significant_snps_imputed.csv"))
        input_columns = [c for c in df.columns if ":" in c]

    elif modality == "lab":
        df = pd.read_csv(os.path.join(input_dir, "data_lab_imputed.csv"))
        input_columns = [c for c in df.columns if c != "FID"]

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

    df["FID"] = df["FID"].apply(normalize_fid)
    pheno["FID"] = pheno["FID"].apply(normalize_fid)

    pheno = pheno[
        (pheno["Race_Admix"] == "White") &
        (pheno["Diagnosis"].isin(["CD", "UC"]))
    ].dropna(subset=["PSC"])

    df = pd.merge(df[["FID"] + input_columns], pheno[["FID", "PSC"]], on="FID", how="inner")

    # Raw X (no manual encoding) — pipeline handles it
    X_raw = df[input_columns].copy()
    y = df["PSC"].astype(int).to_numpy()

    # Save raw training columns (useful for prediction scripts)
    with open(os.path.join(out_dir, "training_columns.json"), "w") as f:
        json.dump({"modality": modality, "input_columns": input_columns}, f, indent=2)

    # -------------------------
    # Model + Nested CV (Pipeline)
    # -------------------------
    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=seed)

    oof_pred = np.zeros(len(y), dtype=float)
    fold_aucs: List[float] = []
    best_params: List[Dict] = []

    for fold, (tr, te) in enumerate(outer_cv.split(X_raw, y), start=1):
        X_tr, y_tr = X_raw.iloc[tr], y[tr]
        X_te, y_te = X_raw.iloc[te], y[te]

        pipe = build_pipeline(df=df.iloc[tr], input_columns=input_columns, n_jobs=n_jobs, seed=seed)

        inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed + fold)

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=rf_param_dist(),
            n_iter=inner_iters,
            scoring="roc_auc",
            n_jobs=n_jobs,
            cv=inner_cv,
            random_state=seed,
            verbose=0,
        )
        search.fit(X_tr, y_tr)

        best_pipe = search.best_estimator_
        oof_pred[te] = best_pipe.predict_proba(X_te)[:, 1].astype(float)

        auc_fold = roc_auc_score(y_te, oof_pred[te])
        fold_aucs.append(float(auc_fold))
        best_params.append(search.best_params_)

        print(f"Fold {fold}: AUC = {auc_fold:.4f}")

    # -------------------------
    # Save CV outputs
    # -------------------------
    fpr, tpr, thr = roc_curve(y, oof_pred)

    pd.DataFrame({"y_true": y, "y_oof_prob": oof_pred}).to_csv(
        os.path.join(out_dir, "nestedcv_oof_predictions.csv"), index=False
    )
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr}).to_csv(
        os.path.join(out_dir, "nestedcv_roc_curve.csv"), index=False
    )

    summary = {
        "oof_auc": float(roc_auc_score(y, oof_pred)),
        "mean_fold_auc": float(np.mean(fold_aucs)),
        "std_fold_auc": float(np.std(fold_aucs)),
        "n_samples": int(len(y)),
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
        "outer_folds": int(outer_folds),
        "inner_folds": int(inner_folds),
        "inner_iters": int(inner_iters),
        "seed": int(seed),
    }
    with open(os.path.join(out_dir, "nestedcv_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # -------------------------
    # Final refit (Pipeline) + SHAP
    # -------------------------
    final_params = most_common_params(best_params)

    final_pipe = build_pipeline(df=df, input_columns=input_columns, n_jobs=n_jobs, seed=seed)
    final_pipe.set_params(**final_params)
    final_pipe.fit(X_raw, y)

    joblib.dump(final_pipe, os.path.join(out_dir, "best_model.joblib"))

    shap_df = shap_global_importance_for_pipeline(
        fitted_pipe=final_pipe,
        X_raw=X_raw,
        max_samples=shap_max_samples,
        seed=seed,
    )
    shap_df.to_csv(os.path.join(out_dir, "shap_global_importance.csv"), index=False)

    print(f"\nDONE [{modality}] | OOF AUC = {summary['oof_auc']:.4f}")
    print(f"Saved: {os.path.join(out_dir, 'best_model.joblib')}")


# =========================
# Entry point
# =========================

if __name__ == "__main__":
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
    parser.add_argument("--shap-max-samples", type=int, default=2000)

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
    )

