import glob
import os

import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_PATTERN = "data/raw/player/returning/player_returning_*.csv"
OUTPUT_DIRECTORY = "data/processed/player/returning"

EXPECTED_SEASONS = list(range(2015, 2026))

EXPECTED_COLUMNS = [
    "season",
    "team",
    "conference",
    "totalPPA",
    "totalPassingPPA",
    "totalReceivingPPA",
    "totalRushingPPA",
    "percentPPA",
    "percentPassingPPA",
    "percentReceivingPPA",
    "percentRushingPPA",
    "usage",
    "passingUsage",
    "receivingUsage",
    "rushingUsage",
]

RENAME_COLUMNS = {
    "totalPPA": "returning_total_ppa",
    "totalPassingPPA": "returning_passing_ppa",
    "totalReceivingPPA": "returning_receiving_ppa",
    "totalRushingPPA": "returning_rushing_ppa",
    "percentPPA": "returning_percent_ppa",
    "percentPassingPPA": "returning_percent_passing_ppa",
    "percentReceivingPPA": "returning_percent_receiving_ppa",
    "percentRushingPPA": "returning_percent_rushing_ppa",
    "usage": "returning_usage",
    "passingUsage": "returning_passing_usage",
    "receivingUsage": "returning_receiving_usage",
    "rushingUsage": "returning_rushing_usage",
}

TRANSFORMED_COLUMNS = [
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
]

NUMERIC_COLUMNS = [
    "totalPPA",
    "totalPassingPPA",
    "totalReceivingPPA",
    "totalRushingPPA",
    "percentPPA",
    "percentPassingPPA",
    "percentReceivingPPA",
    "percentRushingPPA",
    "usage",
    "passingUsage",
    "receivingUsage",
    "rushingUsage",
]


# =============================================================================
# LOAD
# =============================================================================

def load_year(year):
    filepath = (
        f"data/raw/player/returning/"
        f"player_returning_{year}.csv"
    )

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Raw returning-production file not found: {filepath}"
        )

    return pd.read_csv(filepath)


# =============================================================================
# VALIDATION
# =============================================================================

def validate_raw_structure(df, year):
    actual_columns = list(df.columns)

    if actual_columns != EXPECTED_COLUMNS:
        raise ValueError(
            f"{year}: Unexpected raw schema.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Actual:   {actual_columns}"
        )

    if df["season"].nunique() != 1:
        raise ValueError(
            f"{year}: Expected exactly one season value, "
            f"found {df['season'].unique().tolist()}."
        )

    if df["season"].iloc[0] != year:
        raise ValueError(
            f"{year}: Season column does not match filename."
        )

    if df["team"].isna().any():
        raise ValueError(
            f"{year}: Missing team names detected."
        )

    if df["conference"].isna().any():
        raise ValueError(
            f"{year}: Missing conference values detected."
        )

    duplicate_count = df.duplicated(
        subset=["team", "season"]
    ).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"{year}: Found {duplicate_count} duplicate "
            "team-season records."
        )


def validate_numeric_columns(df, year):
    for column in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(
                f"{year}: Column '{column}' is not numeric."
            )

        if df[column].isna().any():
            raise ValueError(
                f"{year}: Missing values detected in '{column}'."
            )


# =============================================================================
# TRANSFORMATION
# =============================================================================

def transform_year(df, year):
    transformed = df.copy()

    transformed = transformed.rename(
        columns=RENAME_COLUMNS
    )

    transformed = transformed[
        TRANSFORMED_COLUMNS
    ]

    transformed = transformed.sort_values(
        ["season", "team"]
    ).reset_index(drop=True)

    return transformed


# =============================================================================
# TRANSFORMED DATA VALIDATION
# =============================================================================

def validate_transformed_data(df, year):
    if list(df.columns) != TRANSFORMED_COLUMNS:
        raise ValueError(
            f"{year}: Unexpected transformed schema."
        )

    if df["season"].nunique() != 1:
        raise ValueError(
            f"{year}: Multiple seasons found after transformation."
        )

    if df["season"].iloc[0] != year:
        raise ValueError(
            f"{year}: Season does not match expected year."
        )

    duplicate_count = df.duplicated(
        subset=["team", "season"]
    ).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"{year}: Duplicate team-season records "
            "found after transformation."
        )

    if df.isna().any().any():
        raise ValueError(
            f"{year}: Missing values detected after transformation."
        )


# =============================================================================
# SAVE
# =============================================================================

def save_year(df, year):
    os.makedirs(
        OUTPUT_DIRECTORY,
        exist_ok=True
    )

    filepath = (
        f"{OUTPUT_DIRECTORY}/"
        f"returning_production_{year}.csv"
    )

    df.to_csv(
        filepath,
        index=False
    )

    print(
        f"Saved {year}: "
        f"{len(df):,} team records → {filepath}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("RETURNING PRODUCTION TRANSFORMATION")
    print("=" * 80)

    files = sorted(glob.glob(INPUT_PATTERN))

    if not files:
        raise FileNotFoundError(
            f"No files found matching: {INPUT_PATTERN}"
        )

    found_years = sorted(
        int(
            os.path.basename(filepath)
            .replace("player_returning_", "")
            .replace(".csv", "")
        )
        for filepath in files
    )

    if found_years != EXPECTED_SEASONS:
        raise ValueError(
            "Unexpected season coverage.\n"
            f"Expected: {EXPECTED_SEASONS}\n"
            f"Found:    {found_years}"
        )

    print(f"\nFound {len(files)} raw files.")
    print(f"Seasons: {found_years}")

    print("\nProcessing seasons...")

    total_rows = 0

    for year in EXPECTED_SEASONS:

        print(f"\n--- {year} ---")

        # Load
        df = load_year(year)

        print(
            f"Loaded: {len(df):,} rows × "
            f"{len(df.columns)} columns"
        )

        # Validate raw data
        validate_raw_structure(
            df,
            year
        )

        validate_numeric_columns(
            df,
            year
        )

        print("Raw validation: PASSED")

        # Transform
        transformed = transform_year(
            df,
            year
        )

        # Validate transformed data
        validate_transformed_data(
            transformed,
            year
        )

        print("Transformed validation: PASSED")

        # Save
        save_year(
            transformed,
            year
        )

        total_rows += len(transformed)

    print("\n" + "=" * 80)
    print("TRANSFORMATION COMPLETE")
    print("=" * 80)

    print(f"\nSeasons processed: {len(EXPECTED_SEASONS)}")
    print(f"Total team-season records: {total_rows:,}")
    print(f"Output directory: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()