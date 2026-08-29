"""
validate_logistic_enhanced_features.py

Focused validation of the enhanced feature set created by
create_logistic_enhanced_features.py.

Purpose
-------
Validate the 8 newly engineered team-level feature families and their
home/away game-level representations before incorporating them into
the next logistic regression experiment.

The validation covers:

1. Structural integrity
2. Missingness and value-range sanity
3. Trend-feature construction
4. Prior-SOS construction and temporal safety
5. Home-vs-away matchup relationships
6. Univariate logistic usefulness

This script does NOT re-audit the existing baseline features.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[6]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "win_probability"
    / "logistic_regression"
    / "enhanced"
)

YEARS = list(range(2015, 2026))

TARGET_COLUMN = "win_home"
GAME_ID_COLUMN = "gameId"

RANDOM_STATE = 42
MIN_N = 100

# -------------------------------------------------------------------------
# New team-level feature families created by the enhanced feature script.
# -------------------------------------------------------------------------

NEW_FEATURES = [
    "pointsForTrend",
    "pointsAgainstTrend",
    "pointDifferentialTrend",
    "totalYardsTrend",
    "netPassingYardsTrend",
    "winPctTrend",
    "priorSOSWinPct",
    "priorSOSPointDiff",
]

# -------------------------------------------------------------------------
# Expected value ranges.
#
# These are deliberately conservative sanity checks rather than strict
# assertions about every possible value.
# -------------------------------------------------------------------------

VALUE_RANGES = {
    "winPctTrend": (-1.0, 1.0),
    "priorSOSWinPct": (0.0, 1.0),
}


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def print_section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def fail(message):
    raise ValueError(f"\nVALIDATION FAILED:\n{message}")


# =============================================================================
# LOADING
# =============================================================================

def load_year(year):
    """
    Load one enhanced feature dataset.
    """

    path = INPUT_DIR / f"logistic_features_{year}.csv"

    if not path.exists():
        fail(
            f"Missing enhanced feature file for {year}:\n"
            f"{path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        fail(f"{year}: enhanced feature dataset is empty.")

    return df


# =============================================================================
# COLUMN HELPERS
# =============================================================================

def home_away_columns():
    """
    Return the expected 16 home/away enhanced columns.
    """

    columns = []

    for feature in NEW_FEATURES:
        columns.append(f"{feature}_home")
        columns.append(f"{feature}_away")

    return columns


def get_new_feature_columns(df):
    """
    Return enhanced columns actually present in the dataframe.
    """

    return [
        column
        for column in home_away_columns()
        if column in df.columns
    ]


# =============================================================================
# STRUCTURAL VALIDATION
# =============================================================================

def structural_validation(df, year):
    """
    Validate basic structure of the enhanced dataset.
    """

    print_section(f"STRUCTURAL VALIDATION — {year}")

    required = [
        GAME_ID_COLUMN,
        TARGET_COLUMN,
    ] + home_away_columns()

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        fail(
            f"{year}: Missing required columns:\n"
            + "\n".join(f"  {column}" for column in missing)
        )

    if df[GAME_ID_COLUMN].duplicated().any():
        duplicate_count = int(
            df[GAME_ID_COLUMN].duplicated().sum()
        )

        fail(
            f"{year}: {duplicate_count} duplicate game IDs found."
        )

    if not set(df[TARGET_COLUMN].dropna().unique()).issubset({0, 1}):
        fail(
            f"{year}: {TARGET_COLUMN} contains values other than 0/1."
        )

    expected_columns = home_away_columns()

    print(f"Rows:                    {len(df):,}")
    print(f"Unique games:             {df[GAME_ID_COLUMN].nunique():,}")
    print(f"Target column:            VALID")
    print(f"Target values:            VALID")
    print(f"New feature columns:      {len(expected_columns)} / {len(expected_columns)}")
    print(f"Game-level grain:         VALID")

    return True


# =============================================================================
# MISSINGNESS / DISTRIBUTION VALIDATION
# =============================================================================

def distribution_validation(df, year):
    """
    Examine missingness and distributions of all new features.
    """

    print_section(f"NEW FEATURE DISTRIBUTIONS — {year}")

    rows = []

    for feature in home_away_columns():

        series = df[feature]

        valid = series.notna() & np.isfinite(series)

        if valid.sum() == 0:
            rows.append(
                {
                    "feature": feature,
                    "n": 0,
                    "missing_pct": 1.0,
                    "mean": np.nan,
                    "std": np.nan,
                    "min": np.nan,
                    "median": np.nan,
                    "max": np.nan,
                }
            )
            continue

        values = series.loc[valid]

        rows.append(
            {
                "feature": feature,
                "n": int(valid.sum()),
                "missing_pct": float(series.isna().mean()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "median": float(values.median()),
                "max": float(values.max()),
            }
        )

    result = pd.DataFrame(rows)

    print(
        result.to_string(
            index=False,
            formatters={
                "missing_pct": "{:.4f}".format,
                "mean": "{:.4f}".format,
                "std": "{:.4f}".format,
                "min": "{:.4f}".format,
                "median": "{:.4f}".format,
                "max": "{:.4f}".format,
            },
        )
    )

    # -------------------------------------------------------------------------
    # Check for completely missing features.
    # -------------------------------------------------------------------------

    completely_missing = result[
        result["n"] == 0
    ]

    if not completely_missing.empty:
        fail(
            f"{year}: Completely missing enhanced features:\n"
            + "\n".join(
                f"  {feature}"
                for feature in completely_missing["feature"]
            )
        )

    # -------------------------------------------------------------------------
    # Check for impossible values in bounded features.
    # -------------------------------------------------------------------------

    for feature, (lower, upper) in VALUE_RANGES.items():

        for suffix in ["_home", "_away"]:

            column = f"{feature}{suffix}"

            values = df[column].dropna()

            if values.empty:
                continue

            invalid = (
                (values < lower)
                | (values > upper)
            )

            if invalid.any():
                fail(
                    f"{year}: {column} contains values outside "
                    f"[{lower}, {upper}]."
                )

    print()
    print("Distribution sanity checks: VALID")

    return result


# =============================================================================
# TREND FEATURE VALIDATION
# =============================================================================

def validate_trend_construction(df, year):
    """
    Verify that trend features correspond to recent form minus
    season-to-date form.

    The engineered trend features should represent:

        recent average - season-to-date average

    using the pregame values already present in the enhanced dataset.
    """

    print_section(f"TREND CONSTRUCTION VALIDATION — {year}")

    mappings = {
        "pointsForTrend": (
            "pointsForAvgLast3",
            "pointsForAvgBefore",
        ),
        "pointsAgainstTrend": (
            "pointsAgainstAvgLast3",
            "pointsAgainstAvgBefore",
        ),
        "pointDifferentialTrend": (
            "pointDifferentialAvgLast3",
            "pointDifferentialAvgBefore",
        ),
        "totalYardsTrend": (
            "totalYardsAvgLast3",
            "totalYardsAvgBefore",
        ),
        "netPassingYardsTrend": (
            "netPassingYardsAvgLast3",
            "netPassingYardsAvgBefore",
        ),
        "winPctTrend": (
            "winPctLast3",
            "winPctBefore",
        ),
    }

    validation_rows = []

    for feature, (recent_col, season_col) in mappings.items():

        for suffix in ["_home", "_away"]:

            trend_col = f"{feature}{suffix}"

            if trend_col not in df.columns:
                fail(
                    f"{year}: Missing trend column {trend_col}."
                )

            recent = df[f"{recent_col}{suffix}"]
            season = df[f"{season_col}{suffix}"]
            observed = df[trend_col]

            expected = recent - season

            valid = (
                recent.notna()
                & season.notna()
                & observed.notna()
                & np.isfinite(recent)
                & np.isfinite(season)
                & np.isfinite(observed)
            )

            if valid.sum() == 0:
                validation_rows.append(
                    {
                        "feature": trend_col,
                        "n_checked": 0,
                        "max_abs_error": np.nan,
                        "status": "NO_VALID_ROWS",
                    }
                )
                continue

            error = (
                observed.loc[valid]
                - expected.loc[valid]
            ).abs()

            max_error = float(error.max())

            status = (
                "VALID"
                if max_error < 1e-8
                else "FAILED"
            )

            validation_rows.append(
                {
                    "feature": trend_col,
                    "n_checked": int(valid.sum()),
                    "max_abs_error": max_error,
                    "status": status,
                }
            )

            if status == "FAILED":
                fail(
                    f"{year}: Trend construction mismatch for "
                    f"{trend_col}. "
                    f"Maximum absolute error = {max_error:.10f}"
                )

    result = pd.DataFrame(validation_rows)

    print(
        result.to_string(
            index=False,
            formatters={
                "max_abs_error": "{:.10f}".format,
            },
        )
    )

    print()
    print("Trend construction: VALID")

    return result


# =============================================================================
# FIRST-GAME / TEMPORAL VALIDATION
# =============================================================================

def validate_first_game_features(df, year):
    """
    Verify that features requiring prior games are missing for teams'
    first games of the season.

    This validation uses the existing gamesBefore field to identify
    first-game observations.
    """

    print_section(f"TEMPORAL / FIRST-GAME VALIDATION — {year}")

    if "gamesBefore_home" not in df.columns:
        fail(
            f"{year}: gamesBefore_home is missing."
        )

    if "gamesBefore_away" not in df.columns:
        fail(
            f"{year}: gamesBefore_away is missing."
        )

    results = []

    for side in ["home", "away"]:

        games_before = df[f"gamesBefore_{side}"]

        first_game = games_before == 0

        if first_game.sum() == 0:
            fail(
                f"{year}: No first-game observations found for {side}."
            )

        for feature in NEW_FEATURES:

            column = f"{feature}_{side}"

            if feature in [
                "pointsForTrend",
                "pointsAgainstTrend",
                "pointDifferentialTrend",
                "totalYardsTrend",
                "netPassingYardsTrend",
                "winPctTrend",
                "priorSOSWinPct",
                "priorSOSPointDiff",
            ]:
                non_missing = (
                    df.loc[first_game, column]
                    .notna()
                    .sum()
                )

                results.append(
                    {
                        "side": side,
                        "feature": column,
                        "first_games": int(first_game.sum()),
                        "non_missing_first_games": int(non_missing),
                    }
                )

                if non_missing > 0:
                    fail(
                        f"{year}: {column} has {non_missing} "
                        f"non-missing first-game values."
                    )

    result = pd.DataFrame(results)

    print(
        result.to_string(index=False)
    )

    print()
    print("First-game prior-feature validation: VALID")

    return result


# =============================================================================
# SOS VALIDATION
# =============================================================================

def validate_sos_features(df, year):
    """
    Validate basic properties of prior SOS features.

    This does not reconstruct the entire SOS calculation from raw data.
    The purpose is to verify that the resulting features behave as
    prior-season/opponent-history features should.
    """

    print_section(f"PRIOR SOS VALIDATION — {year}")

    results = []

    for feature in [
        "priorSOSWinPct",
        "priorSOSPointDiff",
    ]:

        for side in ["home", "away"]:

            column = f"{feature}_{side}"

            series = df[column]

            # First-game values should be missing.
            if "gamesBefore_" + side in df.columns:

                first_game = (
                    df[f"gamesBefore_{side}"] == 0
                )

                first_game_non_missing = (
                    series.loc[first_game]
                    .notna()
                    .sum()
                )

                if first_game_non_missing > 0:
                    fail(
                        f"{year}: {column} contains "
                        f"{first_game_non_missing} non-missing "
                        f"first-game values."
                    )

            results.append(
                {
                    "feature": column,
                    "non_missing": int(series.notna().sum()),
                    "mean": float(series.mean()),
                    "std": float(series.std()),
                }
            )

    result = pd.DataFrame(results)

    print(
        result.to_string(
            index=False,
            formatters={
                "mean": "{:.4f}".format,
                "std": "{:.4f}".format,
            },
        )
    )

    print()
    print("Prior SOS sanity checks: VALID")

    return result


# =============================================================================
# HOME VS AWAY MATCHUP VALIDATION
# =============================================================================

def matchup_feature_validation(df, year):
    """
    Construct home-minus-away differences for the new features and
    evaluate their univariate relationship with home victory.

    These differences are for validation only and are NOT written back
    into the modeling data.
    """

    print_section(f"HOME VS AWAY MATCHUP VALIDATION — {year}")

    y = df[TARGET_COLUMN]

    rows = []

    for feature in NEW_FEATURES:

        home_col = f"{feature}_home"
        away_col = f"{feature}_away"

        difference = (
            df[home_col]
            - df[away_col]
        )

        valid = (
            difference.notna()
            & np.isfinite(difference)
            & y.notna()
        )

        if valid.sum() < MIN_N:
            continue

        x = difference.loc[valid]
        target = y.loc[valid]

        if x.nunique() < 2:
            continue

        scaler = StandardScaler()

        x_scaled = scaler.fit_transform(
            x.to_numpy().reshape(-1, 1)
        )

        model = LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        )

        model.fit(
            x_scaled,
            target,
        )

        coefficient = float(
            model.coef_[0][0]
        )

        probability = model.predict_proba(
            x_scaled
        )[:, 1]

        auc = roc_auc_score(
            target,
            probability,
        )

        loss = log_loss(
            target,
            probability,
        )

        odds_ratio = float(
            np.exp(coefficient)
        )

        rows.append(
            {
                "feature": feature,
                "n": int(valid.sum()),
                "coefficient_per_1sd": coefficient,
                "odds_ratio_per_1sd": odds_ratio,
                "auc": auc,
                "log_loss": loss,
            }
        )

    if not rows:
        print("No matchup features could be evaluated.")
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "auc",
        ascending=False,
    )

    print(
        result.to_string(
            index=False,
            formatters={
                "coefficient_per_1sd": "{:.4f}".format,
                "odds_ratio_per_1sd": "{:.4f}".format,
                "auc": "{:.4f}".format,
                "log_loss": "{:.4f}".format,
            },
        )
    )

    return result


# =============================================================================
# UNIVARIATE FEATURE VALIDATION
# =============================================================================

def univariate_new_feature_validation(df, year):
    """
    Evaluate each new home/away feature individually against win_home.

    This is a diagnostic only.
    """

    print_section(f"UNIVARIATE NEW FEATURE VALIDATION — {year}")

    y = df[TARGET_COLUMN]

    rows = []

    for feature in home_away_columns():

        x = df[feature]

        valid = (
            x.notna()
            & np.isfinite(x)
            & y.notna()
        )

        if valid.sum() < MIN_N:
            continue

        x_valid = x.loc[valid]
        y_valid = y.loc[valid]

        if x_valid.nunique() < 2:
            continue

        scaler = StandardScaler()

        x_scaled = scaler.fit_transform(
            x_valid.to_numpy().reshape(-1, 1)
        )

        model = LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        )

        model.fit(
            x_scaled,
            y_valid,
        )

        coefficient = float(
            model.coef_[0][0]
        )

        probability = model.predict_proba(
            x_scaled
        )[:, 1]

        auc = roc_auc_score(
            y_valid,
            probability,
        )

        loss = log_loss(
            y_valid,
            probability,
        )

        rows.append(
            {
                "feature": feature,
                "n": int(valid.sum()),
                "coefficient_per_1sd": coefficient,
                "odds_ratio_per_1sd": float(
                    np.exp(coefficient)
                ),
                "auc": auc,
                "log_loss": loss,
                "missing_pct": float(
                    x.isna().mean()
                ),
            }
        )

    if not rows:
        print("No new features could be evaluated.")
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "auc",
        ascending=False,
    )

    print(
        result.to_string(
            index=False,
            formatters={
                "coefficient_per_1sd": "{:.4f}".format,
                "odds_ratio_per_1sd": "{:.4f}".format,
                "auc": "{:.4f}".format,
                "log_loss": "{:.4f}".format,
                "missing_pct": "{:.4f}".format,
            },
        )
    )

    return result


# =============================================================================
# CROSS-SEASON SUMMARY
# =============================================================================

def create_summary(all_distribution_results):
    """
    Create a compact cross-season summary of the new features.
    """

    print_section("CROSS-SEASON FEATURE SUMMARY")

    combined = pd.concat(
        all_distribution_results,
        ignore_index=True,
    )

    summary = (
        combined
        .groupby("feature")
        .agg(
            seasons=("feature", "count"),
            avg_missing_pct=("missing_pct", "mean"),
            avg_mean=("mean", "mean"),
            avg_std=("std", "mean"),
        )
        .reset_index()
    )

    print(
        summary.to_string(
            index=False,
            formatters={
                "avg_missing_pct": "{:.4f}".format,
                "avg_mean": "{:.4f}".format,
                "avg_std": "{:.4f}".format,
            },
        )
    )

    return summary


# =============================================================================
# PROCESS ONE YEAR
# =============================================================================

def process_year(year):
    """
    Run all validation checks for one season.
    """

    print_section(f"VALIDATING SEASON {year}")

    df = load_year(year)

    print(f"Loaded: {len(df):,} rows × {len(df.columns):,} columns")

    structural_validation(
        df,
        year,
    )

    distribution = distribution_validation(
        df,
        year,
    )

    validate_trend_construction(
        df,
        year,
    )

    validate_first_game_features(
        df,
        year,
    )

    validate_sos_features(
        df,
        year,
    )

    matchup = matchup_feature_validation(
        df,
        year,
    )

    univariate = univariate_new_feature_validation(
        df,
        year,
    )

    print()
    print(f"{year}: ALL VALIDATION CHECKS PASSED")

    return {
        "year": year,
        "rows": len(df),
        "columns": len(df.columns),
        "distribution": distribution,
        "matchup": matchup,
        "univariate": univariate,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print()
    print("=" * 78)
    print("LOGISTIC REGRESSION ENHANCED FEATURE VALIDATION")
    print("=" * 78)

    print()
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input:        {INPUT_DIR}")
    print(f"Years:        {YEARS[0]}–{YEARS[-1]}")

    if not INPUT_DIR.exists():
        fail(
            f"Enhanced feature directory does not exist:\n"
            f"{INPUT_DIR}"
        )

    all_results = []
    all_distributions = []

    for year in YEARS:

        try:

            result = process_year(year)

            all_results.append(result)

            all_distributions.append(
                result["distribution"]
            )

        except Exception:

            print()
            print("=" * 78)
            print(f"FAILED: {year}")
            print("=" * 78)

            raise

    # -------------------------------------------------------------------------
    # Cross-season summary
    # -------------------------------------------------------------------------

    create_summary(
        all_distributions
    )

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------

    print_section("FINAL VALIDATION SUMMARY")

    summary = pd.DataFrame(
        [
            {
                "year": result["year"],
                "rows": result["rows"],
                "columns": result["columns"],
                "structural": "VALID",
                "trend_construction": "VALID",
                "first_game_validation": "VALID",
                "sos_validation": "VALID",
            }
            for result in all_results
        ]
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 78)
    print("ENHANCED FEATURE VALIDATION COMPLETED SUCCESSFULLY")
    print("=" * 78)
    print()


if __name__ == "__main__":
    main()