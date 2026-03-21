#!/usr/bin/env python3
"""
merge_shap_with_annotation.py

Merge a SHAP importance file with an annotation file using SNP keys of the form:
    Chr:Start

Adds selected annotation columns to the SHAP table, drops the duplicate SNP key,
and saves numeric columns with fixed decimal formatting.

Example:
    python merge_shap_with_annotation.py \
        --annotation /common/mcgoverndlab/usr/Miad/PSC/data/data_cleaned/mapping_data/MIRIAD_annotated.hg38_multianno.txt \
        --shap /common/mcgoverndlab/usr/Miad/PSC/results_GitHub/single_modal/genetics/shap_global_importance.csv \
        --output /common/mcgoverndlab/usr/Miad/PSC/results_GitHub/single_modal/genetics/shap_global_importance_annotated.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import pandas as pd


ANNOTATION_COLUMNS_TO_KEEP: List[str] = [
    "SNP",
    "Ref",
    "Alt",
    "Func.refGene",
    "Gene.refGene",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge SHAP feature importance file with annotation file."
    )
    parser.add_argument(
        "--annotation",
        required=True,
        help="Path to annotation file (e.g. MIRIAD_annotated.hg38_multianno.txt)",
    )
    parser.add_argument(
        "--shap",
        required=True,
        help="Path to SHAP CSV file (must contain a 'feature' column)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save merged output file",
    )
    parser.add_argument(
        "--round-decimals",
        type=int,
        default=4,
        help="Number of decimal places for numeric columns (default: 4)",
    )
    parser.add_argument(
        "--output-format",
        choices=["csv", "tsv"],
        default="csv",
        help="Output format (default: csv)",
    )
    return parser.parse_args()


def validate_columns(df: pd.DataFrame, required_cols: List[str], df_name: str) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {df_name}: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def load_annotation(annotation_path: str) -> pd.DataFrame:
    annotated_df = pd.read_csv(annotation_path, sep="\t", dtype=str)

    validate_columns(
        annotated_df,
        ["Chr", "Start", "Ref", "Alt", "Func.refGene", "Gene.refGene"],
        "annotation file",
    )

    annotated_df["SNP"] = (
        annotated_df["Chr"].astype(str).str.strip()
        + ":"
        + annotated_df["Start"].astype(str).str.strip()
    )

    annotated_df = annotated_df[ANNOTATION_COLUMNS_TO_KEEP].copy()

    duplicated_count = annotated_df["SNP"].duplicated().sum()
    if duplicated_count > 0:
        print(
            f"[INFO] Found {duplicated_count} duplicated SNP entries in annotation file. "
            "Keeping the first occurrence for each SNP.",
            file=sys.stderr,
        )
        annotated_df = annotated_df.drop_duplicates(subset="SNP", keep="first")

    return annotated_df


def load_shap(shap_path: str) -> pd.DataFrame:
    shap_df = pd.read_csv(shap_path)
    validate_columns(shap_df, ["feature"], "SHAP file")
    shap_df["feature"] = shap_df["feature"].astype(str).str.strip()
    return shap_df


def save_output(
    df: pd.DataFrame,
    output_path: str,
    output_format: str,
    decimals: int,
) -> None:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    sep = "," if output_format == "csv" else "\t"
    float_fmt = f"%.{decimals}f"

    df.to_csv(
        output_path,
        sep=sep,
        index=False,
        float_format=float_fmt,
    )


def main() -> None:
    args = parse_args()

    annotation_df = load_annotation(args.annotation)
    shap_df = load_shap(args.shap)

    merged_df = shap_df.merge(
        annotation_df,
        how="left",
        left_on="feature",
        right_on="SNP",
    )

    # Drop SNP because feature already contains the same key
    if "SNP" in merged_df.columns:
        merged_df = merged_df.drop(columns=["SNP"])

    # Reorder columns
    preferred_order = [
        "feature",
        "Ref",
        "Alt",
        "Func.refGene",
        "Gene.refGene",
        "mean_abs_shap",
        "mean_shap",
    ]
    existing_preferred = [c for c in preferred_order if c in merged_df.columns]
    remaining = [c for c in merged_df.columns if c not in existing_preferred]
    merged_df = merged_df[existing_preferred + remaining]

    matched = merged_df["Ref"].notna().sum() if "Ref" in merged_df.columns else 0
    total = len(merged_df)

    print(f"[INFO] Total SHAP rows: {total}")
    print(f"[INFO] Matched annotations: {matched}")
    print(f"[INFO] Unmatched rows: {total - matched}")

    save_output(
        df=merged_df,
        output_path=args.output,
        output_format=args.output_format,
        decimals=args.round_decimals,
    )

    print(f"[INFO] Saved merged file to: {args.output}")


if __name__ == "__main__":
    main()
