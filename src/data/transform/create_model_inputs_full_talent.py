"""
Create Model 5 full-talent modeling input datasets.

Source:
    data/processed/features/full_talent/full_talent_features_{year}.csv

Output:
    data/processed/model_inputs/win_probability/full_talent/train.csv
    data/processed/model_inputs/win_probability/full_talent/validation.csv
    data/processed/model_inputs/win_probability/full_talent/test.csv

Temporal split:
    Train:      2015-2022
    Validation: 2023-2024
    Test:       2025

Important:
    - The 2021+ transfer features are added to the common schema.
    - 2015-2020 observations receive NaN for transfer features because
      the source data does not exist for those seasons.
    - No imputation is performed here.
    - No scaling is performed here.
    - No feature selection is performed here.
    - 2025 remains completely separate as the final test set.
"""

from pathlib import Path

import pandas as pd


# =============================================================================
# PROJECT PATHS
# =============================================================================

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
    / "full_talent"
)


# =============================================================================
# TEMPORAL SPLITS
# =============================================================================

TRAIN_YEARS = list(range(2015, 2023))
VALIDATION_YEARS = [2023, 2024]
TEST_YEARS = [2025]

ALL_YEARS = (
    TRAIN_YEARS
    + VALIDATION_YEARS
    + TEST_YEARS
)


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
# MODEL COLUMNS
# =============================================================================

TARGET_COLUMN = "win_home"

METADATA_COLUMNS = [
    "season",
    "gameId",
]

REQUIRED_COLUMNS = [
    "season",
    "gameId",
    TARGET_COLUMN,
]


# =============================================================================
# HELPERS
# =============================================================================

def load_season(year):
    """Load one full-talent season file."""

    path = SOURCE_DIR / f"full_talent_features_{year}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing full-talent feature file for {year}:\n{path}"
        )

    df = pd.read_csv(path)

    print(
        f"Loaded {year}: "
        f"{len(df):,} rows × {len(df.columns):,} columns"
    )

    return df


def validate_season(df, year):
    """Validate basic requirements before combining seasons."""

    expected_rows = EXPECTED_ROWS[year]

    if len(df) != expected_rows:
        raise ValueError(
            f"{year}: expected {expected_rows} rows, "
            f"found {len(df)}"
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{year}: missing required columns: {missing_columns}"
        )

    if df["gameId"].duplicated().any():
        duplicate_count = df["gameId"].duplicated().sum()

        raise ValueError(
            f"{year}: found {duplicate_count} duplicate gameId values"
        )

    invalid_season = (df["season"] != year).sum()

    if invalid_season:
        raise ValueError(
            f"{year}: found {invalid_season} rows with "
            f"season != filename season"
        )

    missing_target = df[TARGET_COLUMN].isna().sum()

    if missing_target:
        raise ValueError(
            f"{year}: found {missing_target} missing target values"
        )

    target_values = set(df[TARGET_COLUMN].unique())

    if not target_values.issubset({0, 1}):
        raise ValueError(
            f"{year}: target contains non-binary values: "
            f"{sorted(target_values)}"
        )


def build_common_schema(frames):
    """
    Build the union of all source columns while preserving the ordering
    from the first full-talent dataset.

    Transfer features introduced in 2021 are therefore added to the
    pre-2021 rows as NaN rather than being dropped.
    """

    common_columns = list(frames[0].columns)

    for df in frames[1:]:
        for column in df.columns:
            if column not in common_columns:
                common_columns.append(column)

    return common_columns


def align_to_schema(df, columns):
    """Return a dataframe containing the complete common schema."""

    return df.reindex(columns=columns)


def combine_years(years, common_columns):
    """Load, validate, align, and combine seasons."""

    frames = []

    for year in years:

        df = load_season(year)

        validate_season(df, year)

        df = align_to_schema(
            df,
            common_columns,
        )

        frames.append(df)

    combined = pd.concat(
        frames,
        axis=0,
        ignore_index=True,
    )

    return combined


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("CREATE MODEL 5 FULL-TALENT INPUT DATASETS")
    print("=" * 80)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Source directory: {SOURCE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    # =========================================================================
    # CREATE OUTPUT DIRECTORY
    # =========================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================================
    # LOAD ALL SOURCE FILES ONCE
    # =========================================================================

    print()
    print("=" * 80)
    print("LOADING ALL SOURCE SEASONS")
    print("=" * 80)

    season_frames = {}

    for year in ALL_YEARS:

        df = load_season(year)

        validate_season(
            df,
            year,
        )

        season_frames[year] = df

    # =========================================================================
    # BUILD COMMON SCHEMA
    # =========================================================================

    print()
    print("=" * 80)
    print("BUILDING COMMON MODELING SCHEMA")
    print("=" * 80)

    common_columns = build_common_schema(
        list(season_frames.values())
    )

    print(
        f"Common schema: {len(common_columns):,} columns"
    )

    print(
        f"2015-2020 source columns: "
        f"{len(season_frames[2015].columns):,}"
    )

    print(
        f"2021-2025 source columns: "
        f"{len(season_frames[2021].columns):,}"
    )

    # =========================================================================
    # IDENTIFY COLUMNS INTRODUCED AFTER 2020
    # =========================================================================

    early_columns = set(
        season_frames[2015].columns
    )

    late_columns = set(
        season_frames[2021].columns
    )

    late_only_columns = sorted(
        late_columns - early_columns
    )

    print()
    print(
        f"Columns introduced in 2021+: "
        f"{len(late_only_columns):,}"
    )

    if late_only_columns:
        print("2021+ columns:")
        for column in late_only_columns:
            print(f"  - {column}")

    # =========================================================================
    # BUILD TEMPORAL DATASETS
    # =========================================================================

    print()
    print("=" * 80)
    print("BUILDING TRAINING DATA")
    print("=" * 80)

    train_df = pd.concat(
        [
            align_to_schema(
                season_frames[year],
                common_columns,
            )
            for year in TRAIN_YEARS
        ],
        axis=0,
        ignore_index=True,
    )

    print(
        f"Train: {len(train_df):,} rows × "
        f"{len(train_df.columns):,} columns"
    )

    print()
    print("=" * 80)
    print("BUILDING VALIDATION DATA")
    print("=" * 80)

    validation_df = pd.concat(
        [
            align_to_schema(
                season_frames[year],
                common_columns,
            )
            for year in VALIDATION_YEARS
        ],
        axis=0,
        ignore_index=True,
    )

    print(
        f"Validation: {len(validation_df):,} rows × "
        f"{len(validation_df.columns):,} columns"
    )

    print()
    print("=" * 80)
    print("BUILDING TEST DATA")
    print("=" * 80)

    test_df = pd.concat(
        [
            align_to_schema(
                season_frames[year],
                common_columns,
            )
            for year in TEST_YEARS
        ],
        axis=0,
        ignore_index=True,
    )

    print(
        f"Test: {len(test_df):,} rows × "
        f"{len(test_df.columns):,} columns"
    )

    # =========================================================================
    # SCHEMA VALIDATION
    # =========================================================================

    print()
    print("=" * 80)
    print("SCHEMA VALIDATION")
    print("=" * 80)

    train_columns = list(train_df.columns)
    validation_columns = list(validation_df.columns)
    test_columns = list(test_df.columns)

    if train_columns != validation_columns:
        raise ValueError(
            "Train and validation column schemas do not match."
        )

    if train_columns != test_columns:
        raise ValueError(
            "Train and test column schemas do not match."
        )

    print(
        f"All datasets contain identical schemas "
        f"({len(common_columns):,} columns)."
    )

    # =========================================================================
    # TEMPORAL VALIDATION
    # =========================================================================

    train_min = train_df["season"].min()
    train_max = train_df["season"].max()

    validation_min = validation_df["season"].min()
    validation_max = validation_df["season"].max()

    test_min = test_df["season"].min()
    test_max = test_df["season"].max()

    if train_min != 2015 or train_max != 2022:
        raise ValueError(
            f"Unexpected training seasons: "
            f"{train_min}-{train_max}"
        )

    if validation_min != 2023 or validation_max != 2024:
        raise ValueError(
            f"Unexpected validation seasons: "
            f"{validation_min}-{validation_max}"
        )

    if test_min != 2025 or test_max != 2025:
        raise ValueError(
            f"Unexpected test seasons: "
            f"{test_min}-{test_max}"
        )

    print("Temporal splits validated.")

    # =========================================================================
    # GAME ID OVERLAP CHECKS
    # =========================================================================

    train_ids = set(train_df["gameId"])
    validation_ids = set(validation_df["gameId"])
    test_ids = set(test_df["gameId"])

    train_validation_overlap = train_ids & validation_ids
    train_test_overlap = train_ids & test_ids
    validation_test_overlap = validation_ids & test_ids

    if train_validation_overlap:
        raise ValueError(
            f"Train/validation gameId overlap detected: "
            f"{len(train_validation_overlap)} games"
        )

    if train_test_overlap:
        raise ValueError(
            f"Train/test gameId overlap detected: "
            f"{len(train_test_overlap)} games"
        )

    if validation_test_overlap:
        raise ValueError(
            f"Validation/test gameId overlap detected: "
            f"{len(validation_test_overlap)} games"
        )

    print("No gameId overlap between temporal splits.")

    # =========================================================================
    # TRANSFER FEATURE MISSINGNESS CHECK
    # =========================================================================

    print()
    print("=" * 80)
    print("TRANSFER FEATURE SCHEMA CHECK")
    print("=" * 80)

    if late_only_columns:

        train_transfer_missing = train_df[
            late_only_columns
        ].isna().all().all()

        early_train = train_df[
            train_df["season"] <= 2020
        ]

        early_transfer_missing = early_train[
            late_only_columns
        ].isna().all().all()

        if not early_transfer_missing:
            raise ValueError(
                "Expected 2021+ transfer columns to be entirely "
                "missing for 2015-2020 rows."
            )

        print(
            "2015-2020 transfer features: "
            "all values are NaN as expected."
        )

        print(
            "2021+ transfer features: "
            "available in the source data."
        )

    # =========================================================================
    # WRITE OUTPUT FILES
    # =========================================================================

    train_path = OUTPUT_DIR / "train.csv"
    validation_path = OUTPUT_DIR / "validation.csv"
    test_path = OUTPUT_DIR / "test.csv"

    train_df.to_csv(
        train_path,
        index=False,
    )

    validation_df.to_csv(
        validation_path,
        index=False,
    )

    test_df.to_csv(
        test_path,
        index=False,
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print()
    print("=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)

    print(
        f"Train:      {len(train_df):,} rows × "
        f"{len(train_df.columns):,} columns"
    )

    print(
        f"Validation: {len(validation_df):,} rows × "
        f"{len(validation_df.columns):,} columns"
    )

    print(
        f"Test:       {len(test_df):,} rows × "
        f"{len(test_df.columns):,} columns"
    )

    print()
    print(
        f"Train seasons:      "
        f"{train_df['season'].min()}-{train_df['season'].max()}"
    )

    print(
        f"Validation seasons: "
        f"{validation_df['season'].min()}-{validation_df['season'].max()}"
    )

    print(
        f"Test seasons:       "
        f"{test_df['season'].min()}-{test_df['season'].max()}"
    )

    print()
    print(
        f"Train target mean:      "
        f"{train_df[TARGET_COLUMN].mean():.6f}"
    )

    print(
        f"Validation target mean: "
        f"{validation_df[TARGET_COLUMN].mean():.6f}"
    )

    print(
        f"Test target mean:       "
        f"{test_df[TARGET_COLUMN].mean():.6f}"
    )

    print()
    print(f"Train output:      {train_path}")
    print(f"Validation output: {validation_path}")
    print(f"Test output:       {test_path}")

    print()
    print("=" * 80)
    print("MODEL 5 INPUT DATASETS CREATED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()