"""
Audit Model 5 Full-Talent Input Datasets

Validates the final model-input datasets used for the full-talent
college football win-probability model.

Checks:
1. Required output files exist
2. Expected row counts
3. Exact common schema across train/validation/test
4. Expected season ranges
5. Game ID uniqueness
6. No game ID overlap across temporal splits
7. Required metadata and target columns
8. Target completeness and binary values
9. Expected predictor count
10. Predictor numeric types
11. No target leakage through predictor names
12. No suspicious outcome-derived feature names
13. No infinite predictor values
14. Missingness summary
15. Transfer feature historical availability
16. No completely missing predictor columns
17. Exact source-to-model-input row/value preservation
18. Exact 2025 test-set preservation

No imputation, scaling, feature selection, or other transformations
are performed by this script.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

YEARS = list(range(2015, 2026))

TRAIN_YEARS = list(range(2015, 2023))
VALIDATION_YEARS = [2023, 2024]
TEST_YEARS = [2025]

EXPECTED_ROWS = {
    "train": 6432,
    "validation": 1741,
    "test": 888,
}

EXPECTED_TOTAL_COLUMNS = 522

METADATA_COLUMNS = [
    "season",
    "gameId",
]

TARGET_COLUMN = "win_home"

TRANSFER_START_YEAR = 2021


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_inputs"
    / "win_probability"
    / "full_talent"
)

SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "full_talent"
)

TRAIN_PATH = MODEL_INPUT_DIR / "train.csv"
VALIDATION_PATH = MODEL_INPUT_DIR / "validation.csv"
TEST_PATH = MODEL_INPUT_DIR / "test.csv"


# =============================================================================
# HELPERS
# =============================================================================

def fail(message):
    """Print an error and terminate the audit."""
    print(f"FAIL: {message}")
    raise AssertionError(message)


def pass_check(message):
    """Print a successful check."""
    print(f"PASS: {message}")


def load_csv(path, label):
    """Load a CSV and verify that it exists."""
    if not path.exists():
        fail(f"{label} file does not exist: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        fail(f"Could not read {label}: {exc}")

    print(f"Loaded {label}: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    return df


def get_predictor_columns(df):
    """Return all model predictors."""
    excluded = set(METADATA_COLUMNS + [TARGET_COLUMN])
    return [column for column in df.columns if column not in excluded]


def check_expected_columns(df, expected_columns, label):
    """Verify exact column order and names."""
    actual = list(df.columns)

    if actual != expected_columns:
        missing = [column for column in expected_columns if column not in actual]
        extra = [column for column in actual if column not in expected_columns]

        print(f"\n{label} schema mismatch.")

        if missing:
            print("Missing columns:")
            for column in missing:
                print(f"  - {column}")

        if extra:
            print("Unexpected columns:")
            for column in extra:
                print(f"  - {column}")

        if not missing and not extra:
            for index, (expected, actual_column) in enumerate(
                zip(expected_columns, actual)
            ):
                if expected != actual_column:
                    print(
                        f"First ordering mismatch at position {index}: "
                        f"expected '{expected}', found '{actual_column}'"
                    )
                    break

        fail(f"{label} does not match the expected schema.")


def compare_dataframes_exact(source_df, model_df, label):
    """
    Compare source and model-input data after aligning rows by gameId.

    The comparison is exact for:
    - columns
    - row count
    - values
    - missingness

    Floating-point values are compared using exact equality after CSV
    round-tripping. This is appropriate because both datasets are read
    from CSV files rather than recomputed independently.
    """

    if "gameId" not in source_df.columns:
        fail(f"{label}: source data does not contain gameId.")

    if "gameId" not in model_df.columns:
        fail(f"{label}: model-input data does not contain gameId.")

    if source_df["gameId"].duplicated().any():
        duplicates = source_df.loc[
            source_df["gameId"].duplicated(keep=False),
            "gameId",
        ].tolist()

        fail(
            f"{label}: source contains duplicate gameId values. "
            f"Examples: {duplicates[:10]}"
        )

    if model_df["gameId"].duplicated().any():
        duplicates = model_df.loc[
            model_df["gameId"].duplicated(keep=False),
            "gameId",
        ].tolist()

        fail(
            f"{label}: model input contains duplicate gameId values. "
            f"Examples: {duplicates[:10]}"
        )

    if set(source_df["gameId"]) != set(model_df["gameId"]):
        missing_from_model = set(source_df["gameId"]) - set(model_df["gameId"])
        missing_from_source = set(model_df["gameId"]) - set(source_df["gameId"])

        if missing_from_model:
            print(
                f"{label}: gameIds missing from model input: "
                f"{list(missing_from_model)[:10]}"
            )

        if missing_from_source:
            print(
                f"{label}: gameIds present only in model input: "
                f"{list(missing_from_source)[:10]}"
            )

        fail(f"{label}: gameId sets do not match.")

    source_aligned = source_df.set_index("gameId").sort_index()
    model_aligned = model_df.set_index("gameId").sort_index()

    if list(source_aligned.columns) != list(model_aligned.columns):
        fail(f"{label}: source and model-input column schemas differ.")

    if source_aligned.shape != model_aligned.shape:
        fail(
            f"{label}: source/model shape mismatch: "
            f"{source_aligned.shape} vs {model_aligned.shape}"
        )

    differences = []

    for column in source_aligned.columns:
        source_series = source_aligned[column]
        model_series = model_aligned[column]

        # Compare missingness first.
        source_missing = source_series.isna()
        model_missing = model_series.isna()

        if not source_missing.equals(model_missing):
            differences.append(
                (
                    column,
                    "missingness",
                    int((source_missing != model_missing).sum()),
                )
            )
            continue

        # Compare non-missing values.
        non_missing = ~source_missing

        if not source_series[non_missing].equals(
            model_series[non_missing]
        ):
            # For numeric columns, use exact array comparison first,
            # then provide a count of differing values.
            if (
                pd.api.types.is_numeric_dtype(source_series)
                and pd.api.types.is_numeric_dtype(model_series)
            ):
                source_values = source_series[non_missing].to_numpy()
                model_values = model_series[non_missing].to_numpy()

                equal = np.array_equal(
                    source_values,
                    model_values,
                    equal_nan=True,
                )

                if not equal:
                    differences.append(
                        (
                            column,
                            "values",
                            int(np.sum(source_values != model_values)),
                        )
                    )
            else:
                differences.append(
                    (
                        column,
                        "values",
                        int(
                            (
                                source_series[non_missing]
                                != model_series[non_missing]
                            ).sum()
                        ),
                    )
                )

    if differences:
        print(f"\n{label}: detected differences.")

        for column, difference_type, count in differences[:20]:
            print(
                f"  - {column}: {difference_type} differences = {count}"
            )

        if len(differences) > 20:
            print(
                f"  ... plus {len(differences) - 20} additional "
                f"differing columns."
            )

        fail(f"{label}: source and model-input values do not match.")

    pass_check(
        f"{label}: source values preserved exactly "
        f"({source_df.shape[0]:,} rows × {source_df.shape[1]:,} columns)."
    )


# =============================================================================
# LOAD DATASETS
# =============================================================================

print("=" * 80)
print("MODEL 5 FULL-TALENT INPUT DATASET AUDIT")
print("=" * 80)

print(f"Project root: {PROJECT_ROOT}")
print(f"Model input directory: {MODEL_INPUT_DIR}")
print(f"Source directory: {SOURCE_DIR}")

print("\n" + "=" * 80)
print("LOADING MODEL INPUT DATASETS")
print("=" * 80)

train = load_csv(TRAIN_PATH, "Train")
validation = load_csv(VALIDATION_PATH, "Validation")
test = load_csv(TEST_PATH, "Test")


# =============================================================================
# FILE / ROW COUNT CHECKS
# =============================================================================

print("\n" + "=" * 80)
print("ROW COUNT CHECK")
print("=" * 80)

datasets = {
    "Train": train,
    "Validation": validation,
    "Test": test,
}

for label, df in datasets.items():
    expected = EXPECTED_ROWS[label.lower()]
    actual = len(df)

    if actual != expected:
        fail(
            f"{label} row count mismatch: "
            f"expected {expected:,}, found {actual:,}."
        )

    pass_check(f"{label}: {actual:,} rows.")


# =============================================================================
# COMMON SCHEMA CHECK
# =============================================================================

print("\n" + "=" * 80)
print("COMMON SCHEMA CHECK")
print("=" * 80)

expected_columns = list(train.columns)

if len(expected_columns) != EXPECTED_TOTAL_COLUMNS:
    fail(
        f"Train contains {len(expected_columns)} columns; "
        f"expected {EXPECTED_TOTAL_COLUMNS}."
    )

for label, df in datasets.items():
    if list(df.columns) != expected_columns:
        fail(
            f"{label} schema does not exactly match the Train schema."
        )

    pass_check(
        f"{label}: exact 522-column schema matches Train."
    )

print(f"Common schema: {EXPECTED_TOTAL_COLUMNS} columns.")


# =============================================================================
# REQUIRED COLUMN CHECK
# =============================================================================

print("\n" + "=" * 80)
print("REQUIRED COLUMN CHECK")
print("=" * 80)

required_columns = METADATA_COLUMNS + [TARGET_COLUMN]

for column in required_columns:
    if column not in expected_columns:
        fail(f"Required column missing: {column}")

    pass_check(f"Required column present: {column}")


# =============================================================================
# PREDICTOR COUNT
# =============================================================================

print("\n" + "=" * 80)
print("PREDICTOR COUNT CHECK")
print("=" * 80)

predictor_columns = get_predictor_columns(train)

expected_predictor_count = (
    EXPECTED_TOTAL_COLUMNS
    - len(METADATA_COLUMNS)
    - 1
)

if len(predictor_columns) != expected_predictor_count:
    fail(
        f"Expected {expected_predictor_count} predictors, "
        f"found {len(predictor_columns)}."
    )

pass_check(
    f"Predictor count: {len(predictor_columns)} "
    f"(522 total - 2 metadata - 1 target)."
)


# =============================================================================
# TEMPORAL SPLIT CHECK
# =============================================================================

print("\n" + "=" * 80)
print("TEMPORAL SPLIT CHECK")
print("=" * 80)

split_expectations = {
    "Train": (train, TRAIN_YEARS),
    "Validation": (validation, VALIDATION_YEARS),
    "Test": (test, TEST_YEARS),
}

for label, (df, expected_seasons) in split_expectations.items():
    actual_seasons = sorted(df["season"].dropna().unique().tolist())

    if actual_seasons != expected_seasons:
        fail(
            f"{label} seasons mismatch: "
            f"expected {expected_seasons}, found {actual_seasons}."
        )

    pass_check(
        f"{label}: seasons {actual_seasons[0]}–{actual_seasons[-1]}."
    )


# =============================================================================
# GAME ID UNIQUENESS
# =============================================================================

print("\n" + "=" * 80)
print("GAME ID UNIQUENESS CHECK")
print("=" * 80)

for label, df in datasets.items():
    duplicate_count = int(df["gameId"].duplicated().sum())

    if duplicate_count > 0:
        duplicate_ids = (
            df.loc[df["gameId"].duplicated(keep=False), "gameId"]
            .drop_duplicates()
            .tolist()
        )

        fail(
            f"{label}: {duplicate_count} duplicate gameId rows. "
            f"Examples: {duplicate_ids[:10]}"
        )

    pass_check(f"{label}: all gameId values are unique.")


# =============================================================================
# CROSS-SPLIT GAME ID OVERLAP
# =============================================================================

print("\n" + "=" * 80)
print("CROSS-SPLIT GAME ID OVERLAP CHECK")
print("=" * 80)

train_ids = set(train["gameId"])
validation_ids = set(validation["gameId"])
test_ids = set(test["gameId"])

train_validation_overlap = train_ids & validation_ids
train_test_overlap = train_ids & test_ids
validation_test_overlap = validation_ids & test_ids

if train_validation_overlap:
    fail(
        "Train/validation gameId overlap detected: "
        f"{list(train_validation_overlap)[:10]}"
    )

if train_test_overlap:
    fail(
        "Train/test gameId overlap detected: "
        f"{list(train_test_overlap)[:10]}"
    )

if validation_test_overlap:
    fail(
        "Validation/test gameId overlap detected: "
        f"{list(validation_test_overlap)[:10]}"
    )

pass_check("No gameId overlap exists between any temporal splits.")


# =============================================================================
# SEASON / GAME ID CONSISTENCY
# =============================================================================

print("\n" + "=" * 80)
print("SEASON / GAME ID CONSISTENCY CHECK")
print("=" * 80)

for label, df in datasets.items():
    season_nulls = int(df["season"].isna().sum())
    game_id_nulls = int(df["gameId"].isna().sum())

    if season_nulls:
        fail(f"{label}: {season_nulls} missing season values.")

    if game_id_nulls:
        fail(f"{label}: {game_id_nulls} missing gameId values.")

    pass_check(f"{label}: season and gameId are complete.")


# =============================================================================
# TARGET CHECK
# =============================================================================

print("\n" + "=" * 80)
print("TARGET CHECK")
print("=" * 80)

for label, df in datasets.items():
    missing_target = int(df[TARGET_COLUMN].isna().sum())

    if missing_target:
        fail(
            f"{label}: {missing_target} missing {TARGET_COLUMN} values."
        )

    unique_target = sorted(df[TARGET_COLUMN].unique().tolist())

    if not set(unique_target).issubset({0, 1}):
        fail(
            f"{label}: target contains non-binary values: "
            f"{unique_target}"
        )

    pass_check(
        f"{label}: target is complete and binary. "
        f"Mean = {df[TARGET_COLUMN].mean():.6f}"
    )


# =============================================================================
# PREDICTOR DATA TYPES
# =============================================================================

print("\n" + "=" * 80)
print("PREDICTOR DATA TYPE CHECK")
print("=" * 80)

for label, df in datasets.items():
    non_numeric = [
        column
        for column in predictor_columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]

    if non_numeric:
        print(f"{label} non-numeric predictors:")
        for column in non_numeric:
            print(f"  - {column}: {df[column].dtype}")

        fail(
            f"{label}: {len(non_numeric)} predictors are non-numeric."
        )

    pass_check(
        f"{label}: all {len(predictor_columns)} predictors are numeric."
    )


# =============================================================================
# TARGET LEAKAGE CHECK
# =============================================================================

print("\n" + "=" * 80)
print("TARGET LEAKAGE CHECK")
print("=" * 80)

target_tokens = [
    "win_home",
    "home_win",
    "away_win",
    "winner",
    "winner_team",
    "winning_team",
    "result",
    "score",
    "points_scored",
    "points_allowed",
    "margin",
    "point_margin",
    "final_score",
    "postgame",
    "post_game",
]

leakage_columns = []

for column in predictor_columns:
    column_lower = column.lower()

    if any(token in column_lower for token in target_tokens):
        leakage_columns.append(column)

if leakage_columns:
    print("Potential leakage-sensitive predictor names:")
    for column in leakage_columns:
        print(f"  - {column}")

    fail(
        "Potential outcome-derived predictor names detected."
    )

pass_check(
    "No outcome-derived predictor names detected."
)


# =============================================================================
# METADATA EXCLUSION CHECK
# =============================================================================

print("\n" + "=" * 80)
print("METADATA / TARGET EXCLUSION CHECK")
print("=" * 80)

for excluded_column in METADATA_COLUMNS + [TARGET_COLUMN]:
    if excluded_column in predictor_columns:
        fail(
            f"Excluded column incorrectly included as predictor: "
            f"{excluded_column}"
        )

pass_check(
    "season, gameId, and win_home are excluded from predictors."
)


# =============================================================================
# INFINITE VALUE CHECK
# =============================================================================

print("\n" + "=" * 80)
print("INFINITE VALUE CHECK")
print("=" * 80)

for label, df in datasets.items():
    numeric_predictors = df[predictor_columns]

    infinite_mask = np.isinf(numeric_predictors.to_numpy())

    infinite_count = int(infinite_mask.sum())

    if infinite_count:
        rows, cols = np.where(infinite_mask)

        examples = [
            predictor_columns[column_index]
            for column_index in cols[:10]
        ]

        print(
            f"{label}: {infinite_count} infinite predictor values."
        )
        print(f"Examples: {examples}")

        fail(
            f"{label}: infinite predictor values detected."
        )

    pass_check(
        f"{label}: no infinite predictor values."
    )


# =============================================================================
# COMPLETELY MISSING PREDICTOR CHECK
# =============================================================================

print("\n" + "=" * 80)
print("COMPLETELY MISSING PREDICTOR CHECK")
print("=" * 80)

combined = pd.concat(
    [train, validation, test],
    axis=0,
    ignore_index=True,
)

completely_missing = [
    column
    for column in predictor_columns
    if combined[column].isna().all()
]

if completely_missing:
    print("Completely missing predictors:")
    for column in completely_missing:
        print(f"  - {column}")

    fail(
        "Predictors that are completely missing across all model "
        "inputs were detected."
    )

pass_check(
    "No predictor is completely missing across the full modeling dataset."
)


# =============================================================================
# MISSINGNESS SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("PREDICTOR MISSINGNESS SUMMARY")
print("=" * 80)

for label, df in datasets.items():
    missing_counts = df[predictor_columns].isna().sum()
    missing_percent = (
        missing_counts / len(df) * 100
    )

    missing_features = pd.DataFrame(
        {
            "missing_count": missing_counts,
            "missing_percent": missing_percent,
        }
    )

    missing_features = missing_features[
        missing_features["missing_count"] > 0
    ].sort_values(
        "missing_percent",
        ascending=False,
    )

    print(f"\n{label}:")

    if missing_features.empty:
        print("  No missing predictor values.")
    else:
        print(
            f"  {len(missing_features)} predictors contain "
            f"missing values."
        )

        print(
            missing_features.head(15).to_string()
        )


# =============================================================================
# TRANSFER FEATURE IDENTIFICATION
# =============================================================================

print("\n" + "=" * 80)
print("TRANSFER FEATURE CHECK")
print("=" * 80)

transfer_columns = [
    column
    for column in predictor_columns
    if (
        "incoming_transfer" in column
        or "outgoing_transfer" in column
        or "incoming_" in column
        and (
            "star" in column
            or "rating" in column
            or "rated_player" in column
        )
        or "outgoing_" in column
        and (
            "star" in column
            or "rating" in column
            or "rated_player" in column
        )
    )
]

# Deduplicate while preserving schema order.
transfer_columns = list(dict.fromkeys(transfer_columns))

if len(transfer_columns) != 36:
    fail(
        f"Expected 36 transfer predictors, found "
        f"{len(transfer_columns)}."
    )

pass_check(
    f"Identified all {len(transfer_columns)} transfer predictors."
)


# =============================================================================
# TRANSFER HISTORICAL AVAILABILITY
# =============================================================================

print("\n" + "=" * 80)
print("TRANSFER HISTORICAL AVAILABILITY CHECK")
print("=" * 80)

for label, df in datasets.items():
    older = df[df["season"] < TRANSFER_START_YEAR]
    newer = df[df["season"] >= TRANSFER_START_YEAR]

    if not older.empty:
        older_non_missing = (
            older[transfer_columns]
            .notna()
            .sum()
            .sum()
        )

        if older_non_missing != 0:
            fail(
                f"{label}: transfer features contain "
                f"{older_non_missing} non-missing values before "
                f"{TRANSFER_START_YEAR}."
            )

        pass_check(
            f"{label}: all transfer predictors are NaN "
            f"before {TRANSFER_START_YEAR}."
        )

    if not newer.empty:
        newer_non_missing = (
            newer[transfer_columns]
            .notna()
            .sum()
            .sum()
        )

        if newer_non_missing == 0:
            fail(
                f"{label}: transfer predictors are completely "
                f"missing for {TRANSFER_START_YEAR}+."
            )

        pass_check(
            f"{label}: transfer predictors contain data "
            f"for {TRANSFER_START_YEAR}+."
        )


# =============================================================================
# TRANSFER FEATURE COMPLETENESS BY SEASON
# =============================================================================

print("\n" + "=" * 80)
print("TRANSFER FEATURE SEASON SUMMARY")
print("=" * 80)

for season in YEARS:
    source_path = SOURCE_DIR / f"full_talent_features_{season}.csv"

    if not source_path.exists():
        fail(
            f"Source full-talent file missing for {season}: "
            f"{source_path}"
        )

    # Find the model-input rows for this season.
    if season in TRAIN_YEARS:
        model_df = train[train["season"] == season]
    elif season in VALIDATION_YEARS:
        model_df = validation[validation["season"] == season]
    else:
        model_df = test[test["season"] == season]

    transfer_non_missing = int(
        model_df[transfer_columns].notna().sum().sum()
    )

    transfer_total = (
        len(model_df) * len(transfer_columns)
    )

    if season < TRANSFER_START_YEAR:
        expected_status = "all NaN"

        if transfer_non_missing != 0:
            fail(
                f"{season}: expected all transfer values to be NaN, "
                f"found {transfer_non_missing} non-missing values."
            )
    else:
        expected_status = "available"

        if transfer_non_missing == 0:
            fail(
                f"{season}: expected transfer values to be available, "
                f"but all values are missing."
            )

    print(
        f"{season}: "
        f"{transfer_non_missing:,}/{transfer_total:,} non-missing "
        f"({transfer_non_missing / transfer_total * 100:.2f}%) "
        f"— {expected_status}"
    )


pass_check(
    "Transfer feature availability matches the historical API era."
)


# =============================================================================
# SOURCE-TO-MODEL-INPUT PRESERVATION
# =============================================================================

print("\n" + "=" * 80)
print("SOURCE-TO-MODEL-INPUT VALUE PRESERVATION")
print("=" * 80)

print(
    "Comparing every model-input season against its corresponding "
    "full-talent source file..."
)

for season in YEARS:
    source_path = SOURCE_DIR / f"full_talent_features_{season}.csv"

    source_df = load_csv(
        source_path,
        f"Source full-talent {season}",
    )

    if season in TRAIN_YEARS:
        model_df = train[train["season"] == season].copy()
        split_label = "Train"
    elif season in VALIDATION_YEARS:
        model_df = validation[validation["season"] == season].copy()
        split_label = "Validation"
    else:
        model_df = test[test["season"] == season].copy()
        split_label = "Test"

    # The model-input schema intentionally contains the union of all
    # full-talent columns. Older seasons therefore need the 36 transfer
    # columns added to them as NaN for comparison.
    source_aligned = source_df.reindex(columns=expected_columns)

    compare_dataframes_exact(
        source_aligned,
        model_df,
        f"{season} ({split_label})",
    )


# =============================================================================
# 2025 TEST-SET PRESERVATION
# =============================================================================

print("\n" + "=" * 80)
print("2025 TEST-SET INTEGRITY CHECK")
print("=" * 80)

source_2025_path = SOURCE_DIR / "full_talent_features_2025.csv"

source_2025 = load_csv(
    source_2025_path,
    "2025 full-talent source",
)

source_2025_aligned = source_2025.reindex(
    columns=expected_columns
)

compare_dataframes_exact(
    source_2025_aligned,
    test,
    "2025 test set",
)

pass_check(
    "2025 test set exactly matches the 2025 full-talent source dataset."
)


# =============================================================================
# TARGET / SOURCE CONSISTENCY
# =============================================================================

print("\n" + "=" * 80)
print("TARGET SOURCE CONSISTENCY CHECK")
print("=" * 80)

for season in YEARS:
    source_path = SOURCE_DIR / f"full_talent_features_{season}.csv"
    source_df = pd.read_csv(source_path)

    if season in TRAIN_YEARS:
        model_df = train[train["season"] == season]
    elif season in VALIDATION_YEARS:
        model_df = validation[validation["season"] == season]
    else:
        model_df = test[test["season"] == season]

    source_target = (
        source_df.set_index("gameId")[TARGET_COLUMN]
        .sort_index()
    )

    model_target = (
        model_df.set_index("gameId")[TARGET_COLUMN]
        .sort_index()
    )

    if not source_target.equals(model_target):
        fail(
            f"{season}: target values differ between source "
            f"and model input."
        )

pass_check(
    "Target values match the source full-talent datasets for every season."
)


# =============================================================================
# FINAL DATASET SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("FINAL MODEL 5 INPUT SUMMARY")
print("=" * 80)

print(
    f"Train:      {len(train):,} rows × {len(train.columns):,} columns"
)
print(
    f"Validation: {len(validation):,} rows × "
    f"{len(validation.columns):,} columns"
)
print(
    f"Test:       {len(test):,} rows × {len(test.columns):,} columns"
)

print(
    f"\nPredictors: {len(predictor_columns):,}"
)

print(
    f"Train seasons:      {sorted(train['season'].unique())}"
)
print(
    f"Validation seasons: {sorted(validation['season'].unique())}"
)
print(
    f"Test seasons:       {sorted(test['season'].unique())}"
)

print(
    f"\nTrain target mean:      {train[TARGET_COLUMN].mean():.6f}"
)
print(
    f"Validation target mean: {validation[TARGET_COLUMN].mean():.6f}"
)
print(
    f"Test target mean:       {test[TARGET_COLUMN].mean():.6f}"
)


# =============================================================================
# FINAL RESULT
# =============================================================================

print("\n" + "=" * 80)
print("FINAL AUDIT RESULT")
print("=" * 80)

print("PASS: Model 5 full-talent input datasets passed all audits.")
print()
print("Verified:")
print("  - Required model-input files exist")
print("  - Expected row counts")
print("  - Exact common 522-column schema")
print("  - Correct temporal splits")
print("  - Unique gameId values")
print("  - No gameId overlap between splits")
print("  - Complete binary target")
print("  - Correct predictor count")
print("  - Numeric predictors")
print("  - No target/outcome-derived predictor names")
print("  - No infinite predictor values")
print("  - No completely missing predictors")
print("  - Transfer-era historical availability")
print("  - Source-to-model-input value preservation")
print("  - Target preservation")
print("  - Exact 2025 test-set preservation")
print()
print("MODEL 5 INPUT DATASETS ARE READY FOR MODELING.")
print("=" * 80)