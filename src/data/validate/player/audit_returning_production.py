"""
Audit CFBD returning production data.

Purpose:
    Diagnose the statistical behavior of the CFBD /player/returning endpoint,
    with particular attention to the percentage-PPA fields.

This script is diagnostic only:
    - It does not modify source data.
    - It does not clip, winsorize, or transform model features.
    - It does not make modeling decisions.

Input:
    data/processed/player/returning/returning_production_*.csv

Output:
    Console diagnostics only.
"""

import glob
import os

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_PATTERN = "data/processed/player/returning/returning_production_*.csv"

PPA_COLUMNS = [
    "returning_total_ppa",
    "returning_passing_ppa",
    "returning_receiving_ppa",
    "returning_rushing_ppa",
]

PERCENT_COLUMNS = [
    "returning_percent_ppa",
    "returning_percent_passing_ppa",
    "returning_percent_receiving_ppa",
    "returning_percent_rushing_ppa",
]

USAGE_COLUMNS = [
    "returning_usage",
    "returning_passing_usage",
    "returning_receiving_usage",
    "returning_rushing_usage",
]

IDENTIFIER_COLUMNS = [
    "season",
    "team",
    "conference",
]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_processed_data():
    """Load and combine all yearly processed returning-production files."""

    files = sorted(glob.glob(INPUT_PATTERN))

    if not files:
        raise FileNotFoundError(
            f"No processed returning-production files found matching:\n"
            f"    {INPUT_PATTERN}"
        )

    print(f"Found {len(files)} processed files.")

    dataframes = []

    for filepath in files:
        filename = os.path.basename(filepath)
        df = pd.read_csv(filepath)

        print(f"  {filename}: {len(df):,} rows × {len(df.columns)} columns")

        dataframes.append(df)

    combined = pd.concat(dataframes, ignore_index=True)

    return combined


# =============================================================================
# BASIC VALIDATION
# =============================================================================

def audit_structure(df):
    """Audit basic structure and integrity."""

    print("\n" + "=" * 80)
    print("STRUCTURAL AUDIT")
    print("=" * 80)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print(f"Seasons: {df['season'].min()}–{df['season'].max()}")
    print(f"Unique teams: {df['team'].nunique():,}")

    missing = int(df.isna().sum().sum())
    duplicates = int(df.duplicated(["team", "season"]).sum())

    print(f"Missing values: {missing:,}")
    print(f"Duplicate team-season records: {duplicates:,}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")


# =============================================================================
# PERCENTAGE DISTRIBUTIONS
# =============================================================================

def audit_percentage_distributions(df):
    """Summarize the four percentage-PPA fields."""

    print("\n" + "=" * 80)
    print("PERCENTAGE-PPA DISTRIBUTION AUDIT")
    print("=" * 80)

    for column in PERCENT_COLUMNS:
        series = df[column]

        outside_unit = ((series < 0) | (series > 1)).sum()
        outside_two = (series.abs() > 2).sum()

        print(f"\n{column}")
        print("-" * len(column))
        print(f"Minimum: {series.min():.6f}")
        print(f"25th percentile: {series.quantile(0.25):.6f}")
        print(f"Median: {series.median():.6f}")
        print(f"75th percentile: {series.quantile(0.75):.6f}")
        print(f"Maximum: {series.max():.6f}")
        print(f"Outside [0, 1]: {outside_unit:,}")
        print(f"Absolute value > 2: {outside_two:,}")


# =============================================================================
# EXTREME OBSERVATIONS
# =============================================================================

def audit_extreme_observations(df):
    """Display the most extreme observations for each percentage-PPA field."""

    print("\n" + "=" * 80)
    print("EXTREME PERCENTAGE-PPA OBSERVATIONS")
    print("=" * 80)

    display_columns = [
        "season",
        "team",
        "conference",
        *PPA_COLUMNS,
        *PERCENT_COLUMNS,
        *USAGE_COLUMNS,
    ]

    for percent_column in PERCENT_COLUMNS:
        print("\n" + "-" * 80)
        print(f"EXTREMES: {percent_column}")
        print("-" * 80)

        # Most negative values
        print("\nMost negative:")
        negative = (
            df.sort_values(percent_column, ascending=True)
            [display_columns]
            .head(10)
        )

        print(negative.to_string(index=False))

        # Most positive values
        print("\nMost positive:")
        positive = (
            df.sort_values(percent_column, ascending=False)
            [display_columns]
            .head(10)
        )

        print(positive.to_string(index=False))


# =============================================================================
# IMPLIED PPA RATIO ANALYSIS
# =============================================================================

def audit_implied_ratios(df):
    """
    Compare percentage-PPA fields against plausible ratios of returning PPA.

    The returning percentage metrics may be related to returning PPA divided
    by total prior-year PPA. We do not assume the exact formula here.

    Because the processed dataset contains returning PPA rather than the
    underlying prior-year total PPA, this section focuses on internal
    relationships and identifies cases where returning PPA is very small,
    negative, or unusually large relative to total returning PPA.
    """

    print("\n" + "=" * 80)
    print("INTERNAL PPA RELATIONSHIP AUDIT")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Component sum check
    # -------------------------------------------------------------------------

    component_sum = (
        df["returning_passing_ppa"]
        + df["returning_receiving_ppa"]
        + df["returning_rushing_ppa"]
    )

    df = df.copy()

    df["ppa_component_sum"] = component_sum
    df["ppa_component_difference"] = (
        df["returning_total_ppa"] - df["ppa_component_sum"]
    )

    print("\nTotal PPA vs. passing + receiving + rushing PPA")
    print(
        df["ppa_component_difference"]
        .describe()
        .to_string()
    )

    exact_matches = np.isclose(
        df["returning_total_ppa"],
        df["ppa_component_sum"],
        atol=0.001,
    ).sum()

    print(f"\nNear-exact matches: {exact_matches:,} / {len(df):,}")

    # -------------------------------------------------------------------------
    # Negative PPA counts
    # -------------------------------------------------------------------------

    print("\nNegative PPA observations:")

    for column in PPA_COLUMNS:
        count = int((df[column] < 0).sum())
        print(f"  {column}: {count:,}")

    # -------------------------------------------------------------------------
    # Very small absolute PPA
    # -------------------------------------------------------------------------

    print("\nVery small absolute PPA observations:")

    for column in PPA_COLUMNS:
        count = int((df[column].abs() < 1).sum())
        print(f"  |{column}| < 1: {count:,}")

    # -------------------------------------------------------------------------
    # Relationship between total and component PPA
    # -------------------------------------------------------------------------

    print("\nComponent PPA as a share of returning total PPA:")

    for component in [
        "returning_passing_ppa",
        "returning_receiving_ppa",
        "returning_rushing_ppa",
    ]:
        ratio = df[component] / df["returning_total_ppa"].replace(0, np.nan)

        print(f"\n{component} / returning_total_ppa")
        print(ratio.describe().to_string())


# =============================================================================
# EXTREME CASE DIAGNOSTICS
# =============================================================================

def audit_extreme_case_relationships(df):
    """
    Investigate whether extreme percentage values coincide with unusual PPA
    values or component relationships.
    """

    print("\n" + "=" * 80)
    print("EXTREME CASE RELATIONSHIP DIAGNOSTICS")
    print("=" * 80)

    for percent_column in [
        "returning_percent_passing_ppa",
        "returning_percent_rushing_ppa",
    ]:
        print("\n" + "-" * 80)
        print(f"{percent_column}")
        print("-" * 80)

        extreme = df.loc[
            df[percent_column].abs() > 2,
            [
                "season",
                "team",
                "conference",
                "returning_total_ppa",
                "returning_passing_ppa",
                "returning_receiving_ppa",
                "returning_rushing_ppa",
                "returning_percent_ppa",
                "returning_percent_passing_ppa",
                "returning_percent_receiving_ppa",
                "returning_percent_rushing_ppa",
                "returning_usage",
                "returning_passing_usage",
                "returning_receiving_usage",
                "returning_rushing_usage",
            ],
        ].copy()

        if extreme.empty:
            print("No observations found.")
            continue

        print(f"Extreme observations: {len(extreme):,}")

        print("\nSorted by absolute percentage magnitude:")

        extreme["absolute_percent"] = extreme[percent_column].abs()

        extreme = extreme.sort_values(
            "absolute_percent",
            ascending=False,
        ).drop(columns=["absolute_percent"])

        print(extreme.to_string(index=False))


# =============================================================================
# SEASON-LEVEL EXTREME AUDIT
# =============================================================================

def audit_extremes_by_season(df):
    """Determine whether extreme values are concentrated in particular seasons."""

    print("\n" + "=" * 80)
    print("EXTREME VALUES BY SEASON")
    print("=" * 80)

    for column in PERCENT_COLUMNS:
        print(f"\n{column}")

        seasonal = (
            df.assign(
                extreme=(df[column].abs() > 2)
            )
            .groupby("season")
            .agg(
                observations=("extreme", "size"),
                extreme_count=("extreme", "sum"),
            )
        )

        seasonal["extreme_pct"] = (
            seasonal["extreme_count"]
            / seasonal["observations"]
            * 100
        )

        print(seasonal.to_string(float_format=lambda x: f"{x:.2f}"))


# =============================================================================
# USAGE RANGE AUDIT
# =============================================================================

def audit_usage(df):
    """Verify that usage fields behave like bounded proportions."""

    print("\n" + "=" * 80)
    print("USAGE FIELD AUDIT")
    print("=" * 80)

    for column in USAGE_COLUMNS:
        series = df[column]

        outside_unit = int(((series < 0) | (series > 1)).sum())

        print(
            f"{column}: "
            f"min={series.min():.4f}, "
            f"median={series.median():.4f}, "
            f"max={series.max():.4f}, "
            f"outside [0,1]={outside_unit:,}"
        )


# =============================================================================
# FINAL SUMMARY
# =============================================================================

def print_summary(df):
    """Print concise conclusions from the audit."""

    print("\n" + "=" * 80)
    print("AUDIT SUMMARY")
    print("=" * 80)

    structural_issues = (
        int(df.isna().sum().sum())
        + int(df.duplicated(["team", "season"]).sum())
    )

    print("\nStructural integrity:")
    if structural_issues == 0:
        print("  PASSED — no missing values or duplicate team-season records.")
    else:
        print("  REVIEW — structural issues detected.")

    print("\nPercentage-PPA fields:")

    for column in PERCENT_COLUMNS:
        outside_unit = int(((df[column] < 0) | (df[column] > 1)).sum())
        outside_two = int((df[column].abs() > 2).sum())

        print(
            f"  {column}: "
            f"{outside_unit:,} outside [0,1], "
            f"{outside_two:,} with |value| > 2"
        )

    print("\nInterpretation:")
    print(
        "  The percentage-PPA fields have not been clipped or otherwise "
        "modified."
    )
    print(
        "  Extreme values require semantic investigation before these "
        "features are incorporated into the modeling dataset."
    )
    print(
        "  Absolute returning PPA and usage fields should be evaluated "
        "separately from the percentage-PPA fields."
    )

    print("\nNext step:")
    print(
        "  Use the extreme observations and PPA relationships above to "
        "determine whether the percentage values are mathematically valid "
        "consequences of the CFBD PPA calculation."
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("RETURNING PRODUCTION SEMANTIC AUDIT")
    print("=" * 80)
    print()
    print("Purpose:")
    print("  Diagnose unusual CFBD returning-production percentage-PPA values.")
    print("  No data is modified by this script.")

    df = load_processed_data()

    audit_structure(df)
    audit_percentage_distributions(df)
    audit_extreme_observations(df)
    audit_implied_ratios(df)
    audit_extreme_case_relationships(df)
    audit_extremes_by_season(df)
    audit_usage(df)
    print_summary(df)


if __name__ == "__main__":
    main()