"""
Create temporal train, validation, and test splits for logistic regression.

Split structure:
    Training:    2015-2022
    Validation:  2023-2024
    Test:        2025

This script does NOT:
    - shuffle the data
    - impute missing values
    - scale features
    - perform feature selection
    - train a model

All preprocessing belongs downstream in the modeling pipeline so that
parameters are learned exclusively from the training data.
"""

from pathlib import Path

import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling"
    / "logistic_regression_data.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling"
)

TRAIN_PATH = OUTPUT_DIR / "logistic_regression_train.csv"
VALIDATION_PATH = OUTPUT_DIR / "logistic_regression_validation.csv"
TEST_PATH = OUTPUT_DIR / "logistic_regression_test.csv"

TRAIN_YEARS = list(range(2015, 2023))
VALIDATION_YEARS = [2023, 2024]
TEST_YEARS = [2025]

TARGET_COLUMN = "win_home"
ID_COLUMN = "gameId"
SEASON_COLUMN = "season"


# ============================================================================
# HELPERS
# ============================================================================

def load_data(path: Path) -> pd.DataFrame:
    """Load the prepared logistic regression dataset."""

    print("\n" + "=" * 70)
    print("LOADING PREPARED MODELING DATA")
    print("=" * 70)

    if not path.exists():
        raise FileNotFoundError(
            f"Prepared modeling dataset not found:\n{path}"
        )

    df = pd.read_csv(path)

    print(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]:,} columns")

    return df


def validate_input_data(df: pd.DataFrame) -> None:
    """Validate the prepared dataset before splitting."""

    print("\n" + "=" * 70)
    print("VALIDATING INPUT DATA")
    print("=" * 70)

    required_columns = {
        SEASON_COLUMN,
        ID_COLUMN,
        TARGET_COLUMN,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Prepared dataset is missing required columns:\n"
            f"{sorted(missing_columns)}"
        )

    # ------------------------------------------------------------------------
    # Season validation
    # ------------------------------------------------------------------------

    seasons = sorted(df[SEASON_COLUMN].dropna().unique())

    expected_seasons = (
        TRAIN_YEARS
        + VALIDATION_YEARS
        + TEST_YEARS
    )

    print(f"Seasons found: {seasons}")

    if seasons != expected_seasons:
        raise ValueError(
            "Unexpected season coverage.\n"
            f"Expected: {expected_seasons}\n"
            f"Found:    {seasons}"
        )

    # ------------------------------------------------------------------------
    # Game ID validation
    # ------------------------------------------------------------------------

    missing_ids = df[ID_COLUMN].isna().sum()
    duplicate_ids = df[ID_COLUMN].duplicated().sum()

    print(f"Missing game IDs:   {missing_ids:,}")
    print(f"Duplicate game IDs: {duplicate_ids:,}")

    if missing_ids > 0:
        raise ValueError(
            f"Found {missing_ids:,} missing game IDs."
        )

    if duplicate_ids > 0:
        raise ValueError(
            f"Found {duplicate_ids:,} duplicate game IDs."
        )

    # ------------------------------------------------------------------------
    # Target validation
    # ------------------------------------------------------------------------

    missing_target = df[TARGET_COLUMN].isna().sum()
    unique_target = sorted(df[TARGET_COLUMN].unique().tolist())

    print(f"Missing target values: {missing_target:,}")
    print(f"Target values: {unique_target}")

    if missing_target > 0:
        raise ValueError(
            f"Found {missing_target:,} missing target values."
        )

    if not set(unique_target).issubset({0, 1}):
        raise ValueError(
            f"Target must contain only 0/1 values. "
            f"Found: {unique_target}"
        )

    # ------------------------------------------------------------------------
    # Row ordering validation
    # ------------------------------------------------------------------------

    if not df[SEASON_COLUMN].is_monotonic_increasing:
        print(
            "\nWARNING: Input data is not sorted by season."
        )
        print(
            "The split will still be deterministic because it is "
            "performed using explicit season filters."
        )

    print("\nInput validation passed.")


def create_splits(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create strict temporal train, validation, and test splits."""

    print("\n" + "=" * 70)
    print("CREATING TEMPORAL SPLITS")
    print("=" * 70)

    train = df[
        df[SEASON_COLUMN].isin(TRAIN_YEARS)
    ].copy()

    validation = df[
        df[SEASON_COLUMN].isin(VALIDATION_YEARS)
    ].copy()

    test = df[
        df[SEASON_COLUMN].isin(TEST_YEARS)
    ].copy()

    return train, validation, test


def validate_split(
    df: pd.DataFrame,
    expected_years: list[int],
    split_name: str,
) -> None:
    """Validate an individual temporal split."""

    seasons = sorted(df[SEASON_COLUMN].unique())

    print(f"\n{split_name} split:")
    print(f"  Rows:    {len(df):,}")
    print(f"  Seasons: {seasons}")

    if seasons != expected_years:
        raise ValueError(
            f"{split_name} contains unexpected seasons.\n"
            f"Expected: {expected_years}\n"
            f"Found:    {seasons}"
        )

    if df[ID_COLUMN].duplicated().any():
        raise ValueError(
            f"Duplicate game IDs found within {split_name}."
        )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"Missing target values found in {split_name}."
        )

    print("  Validation: PASSED")


def validate_split_boundaries(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Ensure temporal boundaries and game separation are correct."""

    print("\n" + "=" * 70)
    print("VALIDATING SPLIT BOUNDARIES")
    print("=" * 70)

    max_train_year = train[SEASON_COLUMN].max()
    min_validation_year = validation[SEASON_COLUMN].min()

    max_validation_year = validation[SEASON_COLUMN].max()
    min_test_year = test[SEASON_COLUMN].min()

    print(
        f"Training ends:       {max_train_year}"
    )
    print(
        f"Validation begins:   {min_validation_year}"
    )
    print(
        f"Validation ends:     {max_validation_year}"
    )
    print(
        f"Test begins:         {min_test_year}"
    )

    if max_train_year >= min_validation_year:
        raise ValueError(
            "Training and validation periods overlap."
        )

    if max_validation_year >= min_test_year:
        raise ValueError(
            "Validation and test periods overlap."
        )

    # ------------------------------------------------------------------------
    # Cross-split game ID validation
    # ------------------------------------------------------------------------

    train_ids = set(train[ID_COLUMN])
    validation_ids = set(validation[ID_COLUMN])
    test_ids = set(test[ID_COLUMN])

    train_validation_overlap = train_ids & validation_ids
    train_test_overlap = train_ids & test_ids
    validation_test_overlap = validation_ids & test_ids

    if train_validation_overlap:
        raise ValueError(
            "Game IDs overlap between training and validation."
        )

    if train_test_overlap:
        raise ValueError(
            "Game IDs overlap between training and test."
        )

    if validation_test_overlap:
        raise ValueError(
            "Game IDs overlap between validation and test."
        )

    print("\nCross-split game ID validation: PASSED")


def print_split_summary(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Print a summary of the resulting splits."""

    print("\n" + "=" * 70)
    print("SPLIT SUMMARY")
    print("=" * 70)

    splits = {
        "Training": train,
        "Validation": validation,
        "Test": test,
    }

    for name, split in splits.items():

        print(f"\n{name}")

        print(
            f"  Rows:       {len(split):,}"
        )

        print(
            f"  Features:   "
            f"{split.shape[1] - 2:,}"
        )

        print(
            f"  Seasons:    "
            f"{split[SEASON_COLUMN].min()}-"
            f"{split[SEASON_COLUMN].max()}"
        )

        target_counts = (
            split[TARGET_COLUMN]
            .value_counts()
            .sort_index()
        )

        target_percentages = (
            split[TARGET_COLUMN]
            .value_counts(normalize=True)
            .sort_index()
            * 100
        )

        print("  Target:")

        for value in target_counts.index:

            label = (
                "Away win"
                if value == 0
                else "Home win"
            )

            print(
                f"    {value} ({label:<9}): "
                f"{target_counts[value]:>6,} "
                f"({target_percentages[value]:>6.2f}%)"
            )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print("\n" + "=" * 70)
    print("CREATING LOGISTIC REGRESSION TEMPORAL SPLITS")
    print("=" * 70)

    print("\nSplit configuration:")
    print(
        f"  Training:   {TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}"
    )
    print(
        f"  Validation:  {VALIDATION_YEARS[0]}-{VALIDATION_YEARS[-1]}"
    )
    print(
        f"  Test:        {TEST_YEARS[0]}"
    )

    # ------------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------------

    df = load_data(INPUT_PATH)

    # ------------------------------------------------------------------------
    # 2. Validate input
    # ------------------------------------------------------------------------

    validate_input_data(df)

    # ------------------------------------------------------------------------
    # 3. Create splits
    # ------------------------------------------------------------------------

    train, validation, test = create_splits(df)

    # ------------------------------------------------------------------------
    # 4. Validate individual splits
    # ------------------------------------------------------------------------

    validate_split(
        train,
        TRAIN_YEARS,
        "Training",
    )

    validate_split(
        validation,
        VALIDATION_YEARS,
        "Validation",
    )

    validate_split(
        test,
        TEST_YEARS,
        "Test",
    )

    # ------------------------------------------------------------------------
    # 5. Validate boundaries and separation
    # ------------------------------------------------------------------------

    validate_split_boundaries(
        train,
        validation,
        test,
    )

    # ------------------------------------------------------------------------
    # 6. Validate total row count
    # ------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATING TOTAL ROW COUNT")
    print("=" * 70)

    total_rows = (
        len(train)
        + len(validation)
        + len(test)
    )

    print(f"Original rows:  {len(df):,}")
    print(f"Split rows:     {total_rows:,}")

    if total_rows != len(df):
        raise ValueError(
            "Split row counts do not sum to the original dataset."
        )

    print("Row-count validation: PASSED")

    # ------------------------------------------------------------------------
    # 7. Print summary
    # ------------------------------------------------------------------------

    print_split_summary(
        train,
        validation,
        test,
    )

    # ------------------------------------------------------------------------
    # 8. Save
    # ------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("SAVING SPLITS")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train.to_csv(
        TRAIN_PATH,
        index=False,
    )

    validation.to_csv(
        VALIDATION_PATH,
        index=False,
    )

    test.to_csv(
        TEST_PATH,
        index=False,
    )

    print(f"\nTraining:")
    print(f"  {TRAIN_PATH}")

    print(f"\nValidation:")
    print(f"  {VALIDATION_PATH}")

    print(f"\nTest:")
    print(f"  {TEST_PATH}")

    # ------------------------------------------------------------------------
    # 9. Final confirmation
    # ------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("TEMPORAL SPLITTING COMPLETE")
    print("=" * 70)

    print("\nFinal split structure:")

    print(
        f"  Training:   {len(train):,} games "
        f"({TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]})"
    )

    print(
        f"  Validation: {len(validation):,} games "
        f"({VALIDATION_YEARS[0]}-{VALIDATION_YEARS[-1]})"
    )

    print(
        f"  Test:       {len(test):,} games "
        f"({TEST_YEARS[0]})"
    )

    print(
        f"\n  Total:      {total_rows:,} games"
    )

    print("\nNo shuffling, imputation, scaling, or modeling "
          "was performed.")


if __name__ == "__main__":
    main()