#!/usr/bin/env python3
"""
categorical_risk_analysis.py

Argparse-based script for:
1) Reading input data from Excel or CSV
2) Filtering to PSC == 1
3) Creating derived categorical variables
4) Generating summary tables
5) Running binary linear regression (HC3 robust SE)
6) Saving regression summaries and boxplot subplots
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf


plt.rcParams.update({
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 12,
})


# ============================================================
# Helper functions
# ============================================================
def p_to_text(p):
    if pd.isna(p):
        return r"$\it{P}$ = NA"
    return rf"$\it{{P}}$ = {p:.4f}"


def format_numeric_columns_for_export(df_in: pd.DataFrame) -> pd.DataFrame:
    df_out = df_in.copy()
    for col in df_out.columns:
        if pd.api.types.is_numeric_dtype(df_out[col]):
            df_out[col] = df_out[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "NA")
    return df_out


def fibrosis_to_numeric(x):
    if pd.isna(x):
        return np.nan

    x = str(x).strip().lower()

    if x in ["0", "1", "2", "3", "4"]:
        return float(x)
    if "0-1" in x:
        return 0.5
    if "3 or 4" in x:
        return 3.5
    if "f2-f3" in x:
        return 2.5
    if "f1 fibrosis with s0 steatosis" in x:
        return 1.0
    if "f2 and s0" in x:
        return 2.0
    if "f0" in x:
        return 0.0
    if x in ["unknown", "no", ""]:
        return np.nan

    return np.nan


def fibrosis_to_binary_group(x):
    if pd.isna(x):
        return np.nan
    if x <= 2:
        return "Non-advanced (0–2)"
    if x >= 3:
        return "Advanced (3–4)"
    return np.nan


def clean_yes_no(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "":
        return np.nan
    s_low = s.lower()
    if s_low == "yes":
        return "Yes"
    if s_low == "no":
        return "No"
    if s_low in ["unknown", "nan", "none"]:
        return np.nan
    return np.nan


def malignancy_yes_no(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip()
    if s == "":
        return np.nan

    s_low = s.lower()
    if s_low == "no":
        return "No"
    if s_low == "unknown":
        return np.nan

    return "Yes"


def numeric_presence_yes_no(x):
    """
    Categorical Yes/No version for age columns.
    - any numeric content -> Yes
    - No -> No
    - Unknown -> missing
    """
    if pd.isna(x):
        return np.nan

    s = str(x).strip()
    if s == "":
        return np.nan

    s_low = s.lower()
    if s_low == "no":
        return "No"
    if s_low == "unknown":
        return np.nan

    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if len(nums) >= 1:
        return "Yes"

    return np.nan


def make_class_count_string(sub, var, order):
    counts = sub[var].value_counts()
    pieces = []
    for g in order:
        pieces.append(f"{g}: {int(counts.get(g, 0))}")
    return "; ".join(pieces)


def run_binary_lm(data, var, label, order, y="risk_multi_late", reference="No"):
    sub = data.dropna(subset=[var, y]).copy()
    sub = sub[sub[var].isin(order)].copy()

    if sub[var].nunique() < 2:
        return None, None, None

    formula = f'{y} ~ C(Q("{var}"), Treatment(reference="{reference}"))'
    model = smf.ols(formula, data=sub).fit(cov_type="HC3")

    coef_names = [x for x in model.params.index if x != "Intercept"]
    if len(coef_names) == 0:
        return None, None, None

    coef_name = coef_names[0]
    other_group = [g for g in order if g != reference][0]
    class_count_str = make_class_count_string(sub, var, order)

    summary_row = pd.DataFrame({
        "Variable": [label],
        "Original_column": [var],
        "Test": ["Binary regression (HC3 robust SE)"],
        "Reference": [reference],
        "Comparison_group": [other_group],
        "N_total": [len(sub)],
        "Class_counts": [class_count_str],
        "N_reference": [(sub[var] == reference).sum()],
        "N_other": [(sub[var] == other_group).sum()],
        "Coefficient": [model.params[coef_name]],
        "CI_lower": [model.conf_int().loc[coef_name, 0]],
        "CI_upper": [model.conf_int().loc[coef_name, 1]],
        "p_value": [model.pvalues[coef_name]],
        "Mean_reference": [sub.loc[sub[var] == reference, y].mean()],
        "Mean_other": [sub.loc[sub[var] == other_group, y].mean()],
        "Median_reference": [sub.loc[sub[var] == reference, y].median()],
        "Median_other": [sub.loc[sub[var] == other_group, y].median()],
    })

    return model, sub, summary_row


def load_input_data(path: str, sheet_name: str | int | None = 0) -> pd.DataFrame:
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path, sheet_name=sheet_name)
    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported input format: {suffix}. Use .xlsx, .xls, or .csv")


def ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Run categorical summary, binary regression, and plotting for PSC risk score data."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input Excel or CSV file."
    )
    parser.add_argument(
        "--sheet",
        default=0,
        help="Excel sheet name or index (used only for Excel files). Default: 0"
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory."
    )
    parser.add_argument(
        "--psc-col",
        default="PSC",
        help="Column name used to filter PSC==1. Default: PSC"
    )
    parser.add_argument(
        "--risk-col",
        default="risk_multi_late",
        help="Risk score column for summaries/regression/plots. Default: risk_multi_late"
    )
    parser.add_argument(
        "--fibrosis-col",
        default="Fibrosis stage",
        help="Fibrosis stage column name. Default: 'Fibrosis stage'"
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=6.5,
        help="Width per subplot column. Default: 6.5"
    )
    parser.add_argument(
        "--fig-height",
        type=float,
        default=5.5,
        help="Height per subplot row. Default: 5.5"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Figure DPI. Default: 600"
    )

    args = parser.parse_args()

    try:
        sheet_value = int(args.sheet)
    except (ValueError, TypeError):
        sheet_value = args.sheet

    ensure_outdir(args.outdir)

    df = load_input_data(args.input, sheet_name=sheet_value)

    if args.psc_col not in df.columns:
        raise ValueError(f"PSC column '{args.psc_col}' not found in input file.")
    if args.risk_col not in df.columns:
        raise ValueError(f"Risk column '{args.risk_col}' not found in input file.")

    df = df[df[args.psc_col] == 1].copy()

    if args.fibrosis_col in df.columns:
        df["fibrosis_numeric"] = df[args.fibrosis_col].apply(fibrosis_to_numeric)
        df["fibrosis_group_binary"] = df["fibrosis_numeric"].apply(fibrosis_to_binary_group)

    yn_cols = ["Recurrent Cholangitis", "ERCP balloon dilation or stenting"]
    for col in yn_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_yes_no)

    malig_raw_col = "Malignancy asscoiated with PSC"
    if malig_raw_col in df.columns:
        df["Malignancy_associated_with_PSC_yes_no"] = df[malig_raw_col].apply(malignancy_yes_no)

    age_cols = [
        "Age at Liver transplatation",
        "Age at Liver transplatation needed",
        "Age at cirrhosis",
        "Age at PSC Dx",
    ]

    for col in age_cols:
        if col in df.columns:
            df[f"{col}__yes_no"] = df[col].apply(numeric_presence_yes_no)

    display_label_map = {
        "fibrosis_group_binary": "Fibrosis group",
        "Age at Liver transplatation__yes_no": "Liver transplatation",
        "Age at Liver transplatation needed__yes_no": "Liver transplatation needed",
        "Age at cirrhosis__yes_no": "cirrhosis",
        "Age at PSC Dx__yes_no": "Age at PSC Dx",
        "Recurrent Cholangitis": "Recurrent Cholangitis",
        "ERCP balloon dilation or stenting": "ERCP balloon/stent",
        "Malignancy_associated_with_PSC_yes_no": "Malignancy with PSC",
    }

    categorical_specs = []

    if "fibrosis_group_binary" in df.columns:
        categorical_specs.append({
            "var": "fibrosis_group_binary",
            "label": display_label_map["fibrosis_group_binary"],
            "order": ["Non-advanced (0–2)", "Advanced (3–4)"],
            "reference": "Non-advanced (0–2)"
        })

    if "Recurrent Cholangitis" in df.columns:
        categorical_specs.append({
            "var": "Recurrent Cholangitis",
            "label": display_label_map["Recurrent Cholangitis"],
            "order": ["No", "Yes"],
            "reference": "No"
        })

    if "ERCP balloon dilation or stenting" in df.columns:
        categorical_specs.append({
            "var": "ERCP balloon dilation or stenting",
            "label": display_label_map["ERCP balloon dilation or stenting"],
            "order": ["No", "Yes"],
            "reference": "No"
        })

    if "Malignancy_associated_with_PSC_yes_no" in df.columns:
        categorical_specs.append({
            "var": "Malignancy_associated_with_PSC_yes_no",
            "label": display_label_map["Malignancy_associated_with_PSC_yes_no"],
            "order": ["No", "Yes"],
            "reference": "No"
        })

    for raw_col in age_cols:
        yn_col = f"{raw_col}__yes_no"
        if yn_col in df.columns:
            categorical_specs.append({
                "var": yn_col,
                "label": display_label_map.get(yn_col, yn_col),
                "order": ["No", "Yes"],
                "reference": "No"
            })

    summary_rows = []

    for spec in categorical_specs:
        var = spec["var"]
        label = spec["label"]
        order = spec["order"]

        sub = df.dropna(subset=[var, args.risk_col]).copy()
        sub = sub[sub[var].isin(order)].copy()

        if sub.empty:
            continue

        counts = sub[var].value_counts()

        grouped = (
            sub.groupby(var)[args.risk_col]
            .agg(N="count", mean="mean", median="median", std="std", min="min", max="max")
            .reindex(order)
            .reset_index()
            .rename(columns={var: "Category"})
        )

        grouped.insert(0, "Variable", label)
        grouped["Feature_total_N"] = len(sub)
        grouped["Class_count"] = grouped["Category"].map(lambda x: int(counts.get(x, 0)))
        grouped["Class_counts_all"] = make_class_count_string(sub, var, order)

        summary_rows.append(grouped)

    summary_all = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    summary_all_export = format_numeric_columns_for_export(summary_all)
    summary_csv = os.path.join(args.outdir, "categorical_summary_all.csv")
    summary_all_export.to_csv(summary_csv, index=False)

    regression_summary_rows = []
    plot_results = {}

    for spec in categorical_specs:
        var = spec["var"]
        label = spec["label"]
        order = spec["order"]
        ref = spec["reference"]

        model, sub, out = run_binary_lm(
            df,
            var=var,
            label=label,
            order=order,
            y=args.risk_col,
            reference=ref
        )

        if out is not None:
            regression_summary_rows.append(out)
            plot_results[var] = {
                "sub": sub,
                "label": label,
                "order": order,
                "annot": p_to_text(out.loc[0, "p_value"])
            }

    regression_summary = (
        pd.concat(regression_summary_rows, ignore_index=True)
        if regression_summary_rows else pd.DataFrame()
    )
    regression_summary_export = format_numeric_columns_for_export(regression_summary)
    regression_csv = os.path.join(args.outdir, "categorical_linear_regression_summary.csv")
    regression_summary_export.to_csv(regression_csv, index=False)

    plot_specs = [spec for spec in categorical_specs if spec["var"] != "Age at cirrhosis__yes_no"]

    ncols = 3
    nplots = len(plot_specs)
    nrows = 2

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(args.fig_width * ncols, args.fig_height * nrows)
    )
    axes = np.array(axes).reshape(-1)

    rng = np.random.default_rng(42)

    for i, ax in enumerate(axes):
        if i >= len(plot_specs):
            ax.axis("off")
            continue

        spec = plot_specs[i]
        var = spec["var"]

        if var not in plot_results:
            ax.axis("off")
            continue

        sub = plot_results[var]["sub"]
        label = plot_results[var]["label"]
        order = plot_results[var]["order"]
        annot = plot_results[var]["annot"]

        present_groups = [g for g in order if len(sub.loc[sub[var] == g]) > 0]
        box_data = [sub.loc[sub[var] == g, args.risk_col].values for g in present_groups]

        ax.boxplot(box_data, labels=present_groups, showfliers=False)

        for j, g in enumerate(present_groups, start=1):
            y = sub.loc[sub[var] == g, args.risk_col].values
            x = rng.normal(loc=j, scale=0.05, size=len(y))
            ax.scatter(x, y, alpha=0.75, s=35)

        panel_label = chr(65 + i)
        ax.set_title(f"({panel_label}) {label}", fontweight="bold")

        # Keep y-label and y-ticks only for left column
        if i % ncols == 0:
            ax.set_ylabel("Multi-modality risk")
            ax.tick_params(axis="y", left=True, labelleft=True)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", left=True, labelleft=False)

        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=25)

        ax.text(
            0.5,
            0.97,
            annot,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=16,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.9,
                edgecolor="gray"
            )
        )

    plt.tight_layout()
    plot_png = os.path.join(args.outdir, "risk_multi_late_categorical_subplots_linear_regression.png")
    plt.savefig(plot_png, dpi=args.dpi, bbox_inches="tight")
    plt.close()

    review_cols = ["FID", args.fibrosis_col, "fibrosis_numeric", "fibrosis_group_binary"]
    review_cols = [c for c in review_cols if c in df.columns]
    if review_cols:
        fibrosis_review = df[review_cols].copy()
        fibrosis_review_export = format_numeric_columns_for_export(fibrosis_review)
        fibrosis_review_csv = os.path.join(args.outdir, "fibrosis_stage_conversion_review.csv")
        fibrosis_review_export.to_csv(fibrosis_review_csv, index=False)
    else:
        fibrosis_review_csv = None

    print("\nCombined categorical summary:\n")
    print(summary_all_export)

    print("\nRegression summary:\n")
    print(regression_summary_export)

    print("\nFiles saved:")
    print(summary_csv)
    print(regression_csv)
    print(plot_png)
    if fibrosis_review_csv is not None:
        print(fibrosis_review_csv)


if __name__ == "__main__":
    main()

