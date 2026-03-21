#!/usr/bin/env python3
"""
aggregate_shap_importance.py

Aggregate one-hot-encoded SHAP global importance features back to their
original variable names when applicable.

Key behavior:
- Aggregates ONLY mean_abs_shap (mean_shap is ignored and removed)
- One-hot encoded features (e.g. extent_UC_E3) -> extent_UC
- Output is sorted by mean_abs_shap (descending)
- Values are rounded at save time

Example:
    python aggregate_shap_importance.py \
        --input shap_global_importance.csv \
        --output shap_global_importance_aggregated.csv
"""

from __future__ import annotations

import argparse
import os
from typing import List

import pandas as pd


CATEGORICAL_ROOTS: List[str] = [
    "Gender_Self",
    "Jewish_Self",
    "Diagnosis",
    "AgeDx",
    "Smoking",
    "FamilyHistory",
    "extent_UC",
    "Surgery",
    "upper_GI",
    "disease_location_CD",
    "behavior_CD",
    "Perianal",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate one-hot SHAP importance features (mean_abs_shap only)."
    )
    parser.add_argument("--input", required=True, help="Input SHAP CSV file")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument(
        "--round-decimals",
        type=int,
        default=4,
        help="Decimal places for output (default: 4)",
    )
    return parser.parse_args()


def map_feature_name(feature: str) -> str:
    feature = str(feature).strip()

    # Keep SNP-like features unchanged
    if ":" in feature:
        return feature

    for root in CATEGORICAL_ROOTS:
        prefix = root + "_"
        if feature.startswith(prefix):
            return root

    return feature


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.input)

    if "feature" not in df.columns or "mean_abs_shap" not in df.columns:
        raise ValueError(
            f"Input must contain 'feature' and 'mean_abs_shap'. "
            f"Found: {list(df.columns)}"
        )

    df["feature_original"] = df["feature"].astype(str).str.strip()
    df["feature"] = df["feature_original"].apply(map_feature_name)

    n_changed = int((df["feature"] != df["feature_original"]).sum())

    out = (
        df.groupby("feature", as_index=False)
        .agg(mean_abs_shap=("mean_abs_shap", "sum"))
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out.to_csv(
        args.output,
        index=False,
        float_format=f"%.{args.round_decimals}f",
    )

    print(f"[INFO] Input rows: {len(df)}")
    print(f"[INFO] Aggregated rows: {len(out)}")
    print(f"[INFO] Features relabeled: {n_changed}")
    print(f"[INFO] Saved to: {args.output}")


if __name__ == "__main__":
    main()
