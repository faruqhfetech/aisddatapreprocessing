#Phase 1 data audit. Writes phase1_strategy/audit_output.txt

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parents[1]
TABLE_PATH = ROOT / "data" / "processed" / "word_table.csv"
OUTPUT_PATH = ROOT / "phase1_strategy" / "audit_output.txt"
TRAIN_ONLY_OUTPUT_PATH = ROOT / "phase1_strategy" / "audit_output_train_only.txt"
# the split value of the six training reciters; the two test reciters are "holdout"
TRAIN_SPLIT = "dev"

TARGET = "is_error_word"
CONTINUOUS = [
    "orig_duration_ms", "new_duration_ms", "stretch_rate", "test_frac",
    "ref_frac", "dtw_cost", "width", "dur_ratio", "dur_z", "cost_z",
    "log_dev", "suspicion",
]
DISCRETE = ["word_idx", "n_words", "surah", "ayah"]
BINARY = ["collapsed"]
NOMINAL = ["reciter", "error_type", "tier", "split"]
IDENTIFIER = ["recording_id"]

report_lines = []


# Adds a line to the report (printed and saved at the end).
def out(text=""):
    report_lines.append(str(text))


def section(title):
    out()
    out("=" * 78)
    out(title)
    out("=" * 78)


def audit_shape(df, features):
    section("A1. SHAPE")
    n_rows, n_cols = df.shape
    n_features = features.shape[1]
    n_recordings = df["recording_id"].nunique()
    n_reciters = df["reciter"].nunique()
    # a verse is a (surah, ayah) pair
    n_verses = df[["surah", "ayah"]].drop_duplicates().shape[0]
    out(f"rows N = {n_rows}, columns = {n_cols} (P = {n_features} features + 1 target)")
    out(f"recordings = {n_recordings}, reciters = {n_reciters}, distinct verses = {n_verses}")


def audit_types(df, features):
    section("A2. VARIABLE INVENTORY")
    groups = [
        ("continuous", CONTINUOUS),
        ("discrete", DISCRETE),
        ("binary", BINARY),
        ("nominal", NOMINAL),
        ("identifier", IDENTIFIER),
    ]
    for name, columns in groups:
        out(f"{name:11s} ({len(columns)}): {', '.join(columns)}")
    out("ordinal: none; datetime/text: none")

    # every column must be in exactly one type list
    every_listed_column = set(CONTINUOUS + DISCRETE + BINARY + NOMINAL + IDENTIFIER)
    assert every_listed_column == set(features.columns)

    out()
    for column in NOMINAL + BINARY:
        n_levels = df[column].nunique()
        level_counts = df[column].value_counts().to_dict()
        out(f"{column}: {n_levels} levels -> {level_counts}")
    out(f"collapsed: unique values {sorted(df['collapsed'].unique())}")


def audit_missingness(df, features):
    section("A3. MISSINGNESS")
    missing_per_column = features.isna().sum()
    n_rows = len(df)

    for column in features.columns:
        n_missing = missing_per_column[column]
        if n_missing > 0:
            out(f"{column}: {n_missing} missing ({n_missing / n_rows:.2%})")

    total_missing = missing_per_column.sum()
    total_cells = features.size
    out(f"overall: {total_missing} of {total_cells} cells missing "
        f"({total_missing / total_cells:.2%})")

    # a row is complete when none of its features is missing
    row_is_complete = features.notna().all(axis=1)
    n_complete = row_is_complete.sum()
    out(f"complete rows: {n_complete}, rows with >=1 missing: {n_rows - n_complete}")

    out()
    out("stretch_rate presence by error_type (rows):")
    has_stretch_rate = df["stretch_rate"].notna()
    presence = has_stretch_rate.groupby(df["error_type"]).sum()
    presence.name = "stretch_rate"
    out(presence.to_string())

    # Little's MCAR test compares missingness patterns across several incomplete columns.
    # Only one column is incomplete here, so we test its missingness indicator against the
    # error type and the target instead (chi-square and Cramer's V).
    out()
    out("stretch_rate missingness indicator against other variables:")
    is_missing = df["stretch_rate"].isna()
    for column in ["error_type", TARGET]:
        table = pd.crosstab(is_missing, df[column])
        chi2, p, dof, _ = stats.chi2_contingency(table)
        out(f"  missing vs {column:13s} chi2={chi2:9.2f} dof={dof:2d} p={p:.3g} "
            f"V={cramers_v(chi2, table):.3f}")


def audit_target(df):
    section("A4. TARGET")
    class_counts = df[TARGET].value_counts()
    n_negative = class_counts[0]
    n_positive = class_counts[1]
    positive_rate = n_positive / len(df)
    # correct words per error word
    imbalance_ratio = n_negative / n_positive
    out(f"class 0: {n_negative}, class 1: {n_positive}, positive rate {positive_rate:.2%}, "
        f"imbalance ratio {imbalance_ratio:.2f}:1")

    out()
    out("positive rate by reciter:")
    by_reciter = df.groupby("reciter")[TARGET].agg(["count", "mean"])
    out(by_reciter.round(3).to_string())


# Shape statistics and outlier counts (1.5×IQR and |z| > 3) for one feature.
def describe_feature(values):
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    is_iqr_outlier = (values < lower_fence) | (values > upper_fence)
    z_scores = np.abs(stats.zscore(values))
    is_z_outlier = z_scores > 3

    return {
        "n": len(values),
        "mean": values.mean(),
        "median": values.median(),
        "std": values.std(),
        "min": values.min(),
        "max": values.max(),
        "skew": stats.skew(values),
        "kurt": stats.kurtosis(values),
        "iqr_lo": lower_fence,
        "iqr_hi": upper_fence,
        "iqr_out": int(is_iqr_outlier.sum()),
        "iqr_out_pct": is_iqr_outlier.mean() * 100,
        "z3_out": int(is_z_outlier.sum()),
    }


def audit_distributions(df, numeric_columns):
    section("B. DISTRIBUTIONS AND OUTLIERS (continuous + discrete)")
    summaries = {}
    for column in numeric_columns:
        values = df[column].dropna()
        summaries[column] = describe_feature(values)

    summary_table = pd.DataFrame(summaries).T
    summary_table.index.name = "feature"
    # the transpose gives object columns; make them numeric, then keep counts as integers
    summary_table = summary_table.astype(float)
    for count_column in ["n", "iqr_out", "z3_out"]:
        summary_table[count_column] = summary_table[count_column].astype(int)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        out(summary_table.round(3).to_string())

    out()
    out("skew rule: |skew| < 0.5 ~symmetric, 0.5-1 moderate, > 1 high")
    out()
    out("log1p skew check for positive right-skewed features:")
    for column in numeric_columns:
        values = df[column].dropna()
        is_non_negative = values.min() >= 0
        skew_before = stats.skew(values)
        # only positive, highly skewed features are log candidates
        if is_non_negative and skew_before > 1:
            skew_after = stats.skew(np.log1p(values))
            out(f"{column}: skew {skew_before:.2f} -> log1p skew {skew_after:.2f}")


# Point-biserial r, Mann-Whitney U and mutual information between one numeric feature and the target.
def numeric_association(feature_values, target_values):
    has_value = feature_values.notna()
    x = feature_values[has_value]
    y = target_values[has_value]

    r, r_p = stats.pointbiserialr(y, x)

    x_errors = x[y == 1]
    x_correct = x[y == 0]
    mann_whitney_p = stats.mannwhitneyu(x_errors, x_correct).pvalue

    # mutual_info_classif expects a 2-D table, even for one feature
    x_as_frame = x.to_frame()
    mutual_info = mutual_info_classif(x_as_frame, y, random_state=0)[0]

    return {"pb_r": r, "pb_p": r_p, "mwu_p": mann_whitney_p, "MI": mutual_info}


# Effect size for chi-square, between 0 (no association) and 1.
def cramers_v(chi2, table):
    n = table.values.sum()
    smaller_side = min(table.shape) - 1
    return np.sqrt(chi2 / (n * smaller_side))


def audit_association(df, numeric_columns):
    section("C. FEATURE-TARGET ASSOCIATION")
    target_values = df[TARGET]

    out("continuous/discrete vs binary target: point-biserial r, Mann-Whitney U, mutual information")
    associations = {}
    for column in numeric_columns:
        associations[column] = numeric_association(df[column], target_values)
    association_table = pd.DataFrame(associations).T
    association_table.index.name = "feature"
    association_table = association_table.sort_values("MI", ascending=False)
    out(association_table.to_string(float_format=lambda v: f"{v:.4g}"))

    out()
    out("categorical vs binary target: chi-square, Cramer's V")
    for column in NOMINAL + BINARY:
        # counts of each category against each target class
        table = pd.crosstab(df[column], target_values)
        # a column with one level (split, in the train-only audit) has no test to run
        if min(table.shape) < 2:
            out(f"{column:10s} only one level or class; chi-square skipped")
            continue
        chi2, p, dof, _ = stats.chi2_contingency(table)
        v = cramers_v(chi2, table)
        out(f"{column:10s} chi2={chi2:9.2f} dof={dof:2d} p={p:.3g} V={v:.3f}")

    out()
    out("positive rate by error_type (leakage check):")
    rate_by_error_type = df.groupby("error_type")[TARGET].mean()
    out(rate_by_error_type.round(3).to_string())


# VIF per column. Also imported by the notebook for the before/after table.
def compute_vif(frame):
    standardized = (frame - frame.mean()) / frame.std()
    # statsmodels' VIF expects the intercept column to be included
    standardized.insert(0, "const", 1.0)
    matrix = standardized.values

    vif_by_feature = {}
    for position, column in enumerate(standardized.columns):
        if column == "const":
            continue
        vif_by_feature[column] = variance_inflation_factor(matrix, position)
    return pd.Series(vif_by_feature)


def audit_collinearity(df, numeric_columns):
    section("D. MULTICOLLINEARITY")
    # stretch_rate is 78% missing, so it would break the correlation and VIF
    columns = []
    for column in numeric_columns:
        if column != "stretch_rate":
            columns.append(column)

    pearson = df[columns].corr("pearson")
    spearman = df[columns].corr("spearman")

    out("pairs with |r| >= 0.85 (Pearson / Spearman):")
    # j starts at i + 1 so each pair is checked once
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            a = columns[i]
            b = columns[j]
            pearson_r = pearson.loc[a, b]
            spearman_r = spearman.loc[a, b]
            if abs(pearson_r) >= 0.85 or abs(spearman_r) >= 0.85:
                out(f"  {a:18s} {b:18s} pearson={pearson_r:+.3f} spearman={spearman_r:+.3f}")

    out()
    out("VIF (all numeric features, standardized, with constant):")
    vif = compute_vif(df[columns])
    for column, value in vif.items():
        out(f"  {column:18s} {value:10.2f}")


# Checks that the Phase 1 report quotes: the first-word rule, the split error rates and
# rec_len_ratio, the engineered feature that is added after the audit.
def audit_extra_checks(df):
    section("E. EXTRA CHECKS QUOTED IN THE REPORTS")
    first_word = df[df["word_idx"] == 0]
    out(f"first words: {len(first_word)}, of which errors: {int(first_word[TARGET].sum())} "
        f"({first_word[TARGET].mean():.1%})")

    out()
    out("positive rate by split (dev = the 6 training reciters, holdout = the 2 test reciters):")
    by_split = df.groupby("split")[TARGET].agg(["count", "sum", "mean"])
    by_split["neg_per_pos"] = (by_split["count"] - by_split["sum"]) / by_split["sum"]
    out(by_split.round(3).to_string())

    out()
    rec_len_ratio = df["new_duration_ms"] / df["orig_duration_ms"]
    out(f"rec_len_ratio skew {stats.skew(rec_len_ratio):.2f}, "
        f"log1p skew {stats.skew(np.log1p(rec_len_ratio)):.2f}")


def main(train_only=False):
    df = pd.read_csv(TABLE_PATH)
    output_path = OUTPUT_PATH
    if train_only:
        # keep the test reciters out of every statistic, so the drops cannot depend on them
        df = df[df["split"] == TRAIN_SPLIT].reset_index(drop=True)
        output_path = TRAIN_ONLY_OUTPUT_PATH
        out(f"TRAIN-ONLY AUDIT: {len(df)} rows from the {df['reciter'].nunique()} training reciters")
    # everything except the target counts as a feature in the audit
    features = df.drop(columns=[TARGET])
    numeric_columns = CONTINUOUS + DISCRETE

    audit_shape(df, features)
    audit_types(df, features)
    audit_missingness(df, features)
    audit_target(df)
    audit_distributions(df, numeric_columns)
    audit_association(df, numeric_columns)
    audit_collinearity(df, numeric_columns)
    audit_extra_checks(df)

    report = "\n".join(report_lines)
    output_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nwrote {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-only", action="store_true",
                        help="audit only the training reciters")
    main(train_only=parser.parse_args().train_only)
