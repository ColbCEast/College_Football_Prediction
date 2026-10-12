"""
Create GB Model 4 + relative talent modeling input datasets.

Purpose
-------
Keep the exact 28 predictors from GB Model 4 and add 37 home-minus-away
talent features:
    - 12 returning-production differences
    - 7 recruiting differences
    - 18 transfer-portal differences

Source:
    data/processed/features/full_talent/full_talent_features_{year}.csv

Output:
    data/processed/model_inputs/win_probability/talent/train.csv
    data/processed/model_inputs/win_probability/talent/validation.csv
    data/processed/model_inputs/win_probability/talent/test.csv
    data/processed/model_inputs/win_probability/talent/feature_list.csv
    data/processed/model_inputs/win_probability/talent/season_audit.csv

Temporal split:
    Train:      2015-2022
    Validation: 2023-2024
    Test:       2025

Important:
    - The baseline predictor set is fixed to the exact 28 GB Model 4 features.
    - Every talent feature is calculated as home value minus away value.
    - Transfer data is unavailable before 2021. Transfer differences for
      2015-2020 are NaN; no values are fabricated or imputed.
    - No imputation, scaling, or feature selection is performed here.
    - Existing full_talent datasets are not modified.
"""

from pathlib import Path

import pandas as pd


# =============================================================================
# PROJECT PATHS
# =============================================================================

# Expected location:
#   src/data/transform/create_model_inputs_gb4_talent.py
# parents[3] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "full_talent"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_inputs"
    / "win_probability"
    / "talent"
)


# =============================================================================
# TEMPORAL SPLITS AND COLUMN NAMES
# =============================================================================

TRAIN_YEARS = list(range(2015, 2023))
VALIDATION_YEARS = [2023, 2024]
TEST_YEARS = [2025]
ALL_YEARS = TRAIN_YEARS + VALIDATION_YEARS + TEST_YEARS

TARGET_COLUMN = "win_home"
SEASON_COLUMN = "season"
GAME_ID_COLUMN = "gameId"

METADATA_COLUMNS = [SEASON_COLUMN, GAME_ID_COLUMN]
REQUIRED_COLUMNS = METADATA_COLUMNS + [TARGET_COLUMN]


# =============================================================================
# EXPECTED ROW COUNTS
# =============================================================================

EXPECTED_ROWS = {
    2015: 829,
    2016: 831,
    2017: 834,
    2018: 845,
    2019: 848,
    2020: 542,
    2021: 849,
    2022: 854,
    2023: 868,
    2024: 873,
    2025: 888,
}


# =============================================================================
# EXACT GB MODEL 4 BASELINE FEATURES (28)
# =============================================================================

BASELINE_FEATURES = [
    # Core strength (8)
    "homePregameElo",
    "awayPregameElo",
    "winPctBefore_home",
    "winPctBefore_away",
    "pointDifferentialBefore_home",
    "pointDifferentialBefore_away",
    "pointDifferentialAvgBefore_home",
    "pointDifferentialAvgBefore_away",

    # Recent form (8)
    "pointDifferentialAvgLast3_home",
    "pointDifferentialAvgLast3_away",
    "pointDifferentialAvgLast5_home",
    "pointDifferentialAvgLast5_away",
    "pointsForAvgLast5_home",
    "pointsForAvgLast5_away",
    "pointsAgainstAvgLast5_home",
    "pointsAgainstAvgLast5_away",

    # Offensive efficiency (8)
    "home_pregame_offense_successRate",
    "away_pregame_offense_successRate",
    "home_pregame_offense_ppa",
    "away_pregame_offense_ppa",
    "yardsPerPassAttemptBefore_home",
    "yardsPerPassAttemptBefore_away",
    "yardsPerRushAttemptBefore_home",
    "yardsPerRushAttemptBefore_away",

    # Defensive efficiency (4)
    "home_pregame_defense_successRate",
    "away_pregame_defense_successRate",
    "home_pregame_defense_ppa",
    "away_pregame_defense_ppa",
]


# =============================================================================
# TALENT SOURCE FEATURES
# =============================================================================

RETURNING_FEATURES = [
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
]

RECRUITING_FEATURES = [
    "recruiting_count",
    "recruiting_rated_count",
    "recruiting_avg_rating",
    "recruiting_max_rating",
    "recruiting_5star_count",
    "recruiting_4star_count",
    "recruiting_top_100_count",
]

TRANSFER_FEATURES = [
    "incoming_transfer_count",
    "incoming_rated_player_count",
    "incoming_rating_sum",
    "incoming_rating_mean",
    "incoming_star_rated_player_count",
    "incoming_2_star_count",
    "incoming_3_star_count",
    "incoming_4_star_count",
    "incoming_5_star_count",
    "outgoing_transfer_count",
    "outgoing_rated_player_count",
    "outgoing_rating_sum",
    "outgoing_rating_mean",
    "outgoing_star_rated_player_count",
    "outgoing_2_star_count",
    "outgoing_3_star_count",
    "outgoing_4_star_count",
    "outgoing_5_star_count",
]

TALENT_FEATURE_GROUPS = {
    "returning_production": RETURNING_FEATURES,
    "recruiting": RECRUITING_FEATURES,
    "transfer_portal": TRANSFER_FEATURES,
}

TALENT_FEATURES = [
    f"{feature}_diff"
    for features in TALENT_FEATURE_GROUPS.values()
    for feature in features
]

TRANSFER_FEATURES_DIFF = [
    f"{feature}_diff" for feature in TRANSFER_FEATURES
]

MODEL_FEATURES = BASELINE_FEATURES + TALENT_FEATURES

OUTPUT_COLUMNS = (
    METADATA_COLUMNS
    + MODEL_FEATURES
    + [TARGET_COLUMN]
)


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def require_columns(df, columns, dataset_name):
    """Raise a clear error when required columns are missing."""
    missing = [
        column for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns:\n{missing}"
        )


def validate_unique_key(df, columns, dataset_name):
    """Require uniqueness for the provided key."""
    duplicate_mask = df.duplicated(
        subset=columns,
        keep=False,
    )

    if duplicate_mask.any():
        examples = (
            df.loc[duplicate_mask, columns]
            .sort_values(columns)
            .head(20)
        )

        raise ValueError(
            f"{dataset_name} contains duplicate key values "
            f"for {columns}.\n"
            f"Examples:\n{examples.to_string(index=False)}"
        )


def load_season(year):
    """Load and validate one season's full-talent feature file."""
    path = SOURCE_DIR / f"full_talent_features_{year}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing full-talent feature file for {year}:\n{path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            f"{year}: source file contains no rows:\n{path}"
        )

    require_columns(
        df,
        REQUIRED_COLUMNS,
        f"Full-talent source {year}",
    )

    expected_rows = EXPECTED_ROWS[year]

    if len(df) != expected_rows:
        raise ValueError(
            f"{year}: expected {expected_rows:,} rows, "
            f"found {len(df):,}."
        )

    if df[SEASON_COLUMN].isna().any():
        raise ValueError(
            f"{year}: season contains missing values."
        )

    observed_seasons = sorted(
        df[SEASON_COLUMN].astype(int).unique().tolist()
    )

    if observed_seasons != [year]:
        raise ValueError(
            f"{year}: expected season value [{year}], "
            f"found {observed_seasons}."
        )

    if df[GAME_ID_COLUMN].isna().any():
        raise ValueError(
            f"{year}: gameId contains missing values."
        )

    validate_unique_key(
        df,
        [GAME_ID_COLUMN],
        f"Full-talent source {year}",
    )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"{year}: {TARGET_COLUMN} contains missing values."
        )

    target_values = set(df[TARGET_COLUMN].unique().tolist())

    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"{year}: target must be binary 0/1; "
            f"found {target_values}."
        )

    print(
        f"Loaded {year}: "
        f"{len(df):,} rows × {len(df.columns):,} columns"
    )

    return df


# =============================================================================
# BUILD MODEL INPUT DATASET
# =============================================================================

def build_model_input(df, year):
    """
    Select the exact baseline and create home-minus-away talent differences.

    Returning-production and recruiting columns must exist as home_/away_
    columns in every season. Transfer source data may be absent before 2021.
    """
    require_columns(
        df,
        BASELINE_FEATURES,
        f"Full-talent source {year} baseline predictors",
    )

    output = df[
        METADATA_COLUMNS + BASELINE_FEATURES
    ].copy()

    created_by_group = {}

    for group_name, features in TALENT_FEATURE_GROUPS.items():
        created_by_group[group_name] = []

        for feature in features:
            home_column = f"home_{feature}"
            away_column = f"away_{feature}"
            difference_column = f"{feature}_diff"

            if group_name == "transfer_portal" and year < 2021:
                # Transfer data is unavailable for these seasons.
                # Set every transfer difference to NaN explicitly.
                output[difference_column] = float("nan")
                created_by_group[group_name].append(
                    difference_column
                )
                continue

            require_columns(
                df,
                [home_column, away_column],
                f"Full-talent source {year} {group_name} predictors",
            )

            home_values = pd.to_numeric(
                df[home_column],
                errors="raise",
            )

            away_values = pd.to_numeric(
                df[away_column],
                errors="raise",
            )

            output[difference_column] = (
                home_values - away_values
            )

            created_by_group[group_name].append(
                difference_column
            )

    output[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    # Enforce a stable, explicit schema and column ordering.
    output = output[OUTPUT_COLUMNS]

    if len(output) != len(df):
        raise ValueError(
            f"{year}: row count changed while building model inputs "
            f"({len(df):,} -> {len(output):,})."
        )

    validate_unique_key(
        output,
        [GAME_ID_COLUMN],
        f"Model input {year}",
    )

    # Before 2021, all transfer differences must be missing.
    if year < 2021:
        nonmissing_transfer = (
            output[TRANSFER_FEATURES_DIFF]
            .notna()
            .sum()
            .sum()
        )

        if nonmissing_transfer:
            raise ValueError(
                f"{year}: expected all transfer differences to be NaN, "
                f"found {nonmissing_transfer:,} nonmissing values."
            )

    return output, created_by_group


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("GB MODEL 4 + RELATIVE TALENT MODEL INPUT CREATION")
    print("=" * 80)
    print()

    print(f"Baseline predictors:        {len(BASELINE_FEATURES)}")
    print(f"Returning-production diffs: {len(RETURNING_FEATURES)}")
    print(f"Recruiting diffs:           {len(RECRUITING_FEATURES)}")
    print(f"Transfer-portal diffs:      {len(TRANSFER_FEATURES)}")
    print(f"Total talent predictors:    {len(TALENT_FEATURES)}")
    print(f"Total model predictors:     {len(MODEL_FEATURES)}")
    print(f"Total output columns:       {len(OUTPUT_COLUMNS)}")
    print()
    print("Talent definition: home-team value minus away-team value")
    print("Transfer differences: NaN for 2015-2020; no imputation here.")
    print()

    # Fail immediately if the feature specification is accidentally changed.
    if len(BASELINE_FEATURES) != 28:
        raise AssertionError(
            f"Expected 28 baseline features, "
            f"found {len(BASELINE_FEATURES)}."
        )

    if len(RETURNING_FEATURES) != 12:
        raise AssertionError(
            "Expected 12 returning-production features."
        )

    if len(RECRUITING_FEATURES) != 7:
        raise AssertionError(
            "Expected 7 recruiting features."
        )

    if len(TRANSFER_FEATURES) != 18:
        raise AssertionError(
            "Expected 18 transfer features."
        )

    if len(MODEL_FEATURES) != 65:
        raise AssertionError(
            f"Expected 65 predictors, found {len(MODEL_FEATURES)}."
        )

    if len(set(MODEL_FEATURES)) != len(MODEL_FEATURES):
        raise AssertionError(
            "Model feature list contains duplicate names."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    season_frames = {}
    season_audit_rows = []

    # -------------------------------------------------------------------------
    # Load, validate, and transform each season independently.
    # -------------------------------------------------------------------------

    for year in ALL_YEARS:
        source_df = load_season(year)

        model_df, created_by_group = build_model_input(
            source_df,
            year,
        )

        season_frames[year] = model_df

        season_audit_rows.append(
            {
                "season": year,
                "rows": len(model_df),
                "columns": len(model_df.columns),
                "baseline_feature_count": len(BASELINE_FEATURES),
                "returning_diff_count": len(
                    created_by_group["returning_production"]
                ),
                "recruiting_diff_count": len(
                    created_by_group["recruiting"]
                ),
                "transfer_diff_count": len(
                    created_by_group["transfer_portal"]
                ),
                "transfer_diff_nonmissing_values": int(
                    model_df[TRANSFER_FEATURES_DIFF]
                    .notna()
                    .sum()
                    .sum()
                ),
                "target_missing": int(
                    model_df[TARGET_COLUMN].isna().sum()
                ),
                "duplicate_game_ids": int(
                    model_df[GAME_ID_COLUMN].duplicated().sum()
                ),
            }
        )

        print(
            f"  Built {year}: {len(model_df):,} rows, "
            f"{len(MODEL_FEATURES)} predictors"
        )

    # -------------------------------------------------------------------------
    # Construct temporal splits in chronological year order.
    # -------------------------------------------------------------------------

    train_df = pd.concat(
        [season_frames[year] for year in TRAIN_YEARS],
        ignore_index=True,
    )

    validation_df = pd.concat(
        [season_frames[year] for year in VALIDATION_YEARS],
        ignore_index=True,
    )

    test_df = pd.concat(
        [season_frames[year] for year in TEST_YEARS],
        ignore_index=True,
    )

    # -------------------------------------------------------------------------
    # Validate split membership, schema, and uniqueness.
    # -------------------------------------------------------------------------

    splits = [
        ("train", train_df, TRAIN_YEARS),
        ("validation", validation_df, VALIDATION_YEARS),
        ("test", test_df, TEST_YEARS),
    ]

    for split_name, split_df, expected_years in splits:
        actual_years = sorted(
            split_df[SEASON_COLUMN].astype(int).unique().tolist()
        )

        if actual_years != expected_years:
            raise ValueError(
                f"{split_name}: expected seasons {expected_years}, "
                f"found {actual_years}."
            )

        if list(split_df.columns) != OUTPUT_COLUMNS:
            raise ValueError(
                f"{split_name}: output schema/order mismatch."
            )

        validate_unique_key(
            split_df,
            [GAME_ID_COLUMN],
            f"{split_name.title()} split",
        )

    # -------------------------------------------------------------------------
    # Ensure game IDs do not overlap between temporal splits.
    # -------------------------------------------------------------------------

    train_ids = set(train_df[GAME_ID_COLUMN])
    validation_ids = set(validation_df[GAME_ID_COLUMN])
    test_ids = set(test_df[GAME_ID_COLUMN])

    if train_ids & validation_ids:
        raise ValueError(
            "Game IDs overlap between train and validation."
        )

    if train_ids & test_ids:
        raise ValueError(
            "Game IDs overlap between train and test."
        )

    if validation_ids & test_ids:
        raise ValueError(
            "Game IDs overlap between validation and test."
        )

    # -------------------------------------------------------------------------
    # Save the three modeling datasets.
    # -------------------------------------------------------------------------

    train_df.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False,
    )

    validation_df.to_csv(
        OUTPUT_DIR / "validation.csv",
        index=False,
    )

    test_df.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save the ordered predictor list and feature groups.
    # -------------------------------------------------------------------------

    feature_groups = (
        ["gb_model_4_baseline"] * len(BASELINE_FEATURES)
        + ["returning_production"] * len(RETURNING_FEATURES)
        + ["recruiting"] * len(RECRUITING_FEATURES)
        + ["transfer_portal"] * len(TRANSFER_FEATURES)
    )

    feature_list = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "feature_index": range(len(MODEL_FEATURES)),
            "feature_group": feature_groups,
        }
    )

    feature_list.to_csv(
        OUTPUT_DIR / "feature_list.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save a concise per-season structural audit.
    # -------------------------------------------------------------------------

    audit_df = pd.DataFrame(season_audit_rows)

    audit_df.to_csv(
        OUTPUT_DIR / "season_audit.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Final summary.
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("SPLIT SUMMARY")
    print("=" * 80)

    print(
        f"Train:      {len(train_df):,} rows | "
        f"seasons {TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}"
    )

    print(
        f"Validation: {len(validation_df):,} rows | "
        f"seasons {VALIDATION_YEARS}"
    )

    print(
        f"Test:       {len(test_df):,} rows | "
        f"seasons {TEST_YEARS}"
    )

    print()
    print(f"Output directory:\n  {OUTPUT_DIR}")
    print()
    print("Files saved:")
    print("  train.csv")
    print("  validation.csv")
    print("  test.csv")
    print("  feature_list.csv")
    print("  season_audit.csv")
    print()
    print("All checks passed.")
    print("=" * 80)


if __name__ == "__main__":
    main()