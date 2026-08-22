import os
import pandas as pd


# ============================================================
# EXPECTED DATASET SIZES
# ============================================================

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


# ============================================================
# REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "gameId",
    "season",
    "week",
    "startDate",
    "homeTeam",
    "awayTeam",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_final_features(year):

    path = (
        f"data/processed/final_features/"
        f"final_features_{year}.csv"
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Final feature file does not exist:\n{path}"
        )

    return pd.read_csv(path)


# ============================================================
# VALIDATE ROW COUNT
# ============================================================

def validate_row_count(df, year):

    expected = EXPECTED_ROWS[year]
    actual = len(df)

    print("\nRow count validation:")
    print(f"  Expected rows: {expected}")
    print(f"  Actual rows:   {actual}")

    if actual == expected:
        print("  PASS: Row count matches expected value.")
        return True

    print("  FAIL: Row count does not match expected value.")
    return False


# ============================================================
# VALIDATE GAME IDs
# ============================================================

def validate_game_ids(df):

    print("\nGame ID validation:")

    missing_ids = df["gameId"].isna().sum()

    duplicate_ids = (
        df["gameId"]
        .value_counts()
        .loc[lambda x: x > 1]
    )

    unique_ids = df["gameId"].nunique()

    print(f"  Rows:              {len(df)}")
    print(f"  Unique game IDs:   {unique_ids}")
    print(f"  Missing game IDs:  {missing_ids}")
    print(
        f"  Duplicate game IDs: "
        f"{len(duplicate_ids)}"
    )

    valid = True

    if missing_ids > 0:
        print("  FAIL: Missing game IDs.")
        valid = False

    if len(duplicate_ids) > 0:
        print("  FAIL: Duplicate game IDs.")

        print("\n  Duplicate IDs:")
        print(duplicate_ids)

        valid = False

    if valid:
        print(
            "  PASS: Game IDs are present "
            "and unique."
        )

    return valid


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

def validate_required_columns(df):

    print("\nRequired column validation:")

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        print("  FAIL: Missing required columns:")

        for column in missing_columns:
            print(f"    {column}")

        return False

    print(
        f"  PASS: All {len(REQUIRED_COLUMNS)} "
        f"required columns are present."
    )

    return True


# ============================================================
# VALIDATE GAME INFORMATION
# ============================================================

def validate_game_information(df):

    print("\nGame information validation:")

    checks = {
        "Missing seasons": df["season"].isna().sum(),
        "Missing weeks": df["week"].isna().sum(),
        "Missing start dates": df["startDate"].isna().sum(),
        "Missing home teams": df["homeTeam"].isna().sum(),
        "Missing away teams": df["awayTeam"].isna().sum(),
    }

    valid = True

    for description, count in checks.items():

        print(
            f"  {description}: {count}"
        )

        if count > 0:
            valid = False

    if valid:
        print(
            "  PASS: Required game information "
            "contains no missing values."
        )
    else:
        print(
            "  FAIL: Required game information "
            "contains missing values."
        )

    return valid


# ============================================================
# VALIDATE SEASON
# ============================================================

def validate_season(df, year):

    print("\nSeason validation:")

    seasons = df["season"].dropna().unique()

    print(f"  Expected season: {year}")
    print(f"  Seasons present: {sorted(seasons)}")

    if len(seasons) != 1:
        print(
            "  FAIL: Multiple seasons are present."
        )
        return False

    if seasons[0] != year:
        print(
            "  FAIL: Dataset contains "
            "the wrong season."
        )
        return False

    print("  PASS: Season is correct.")

    return True


# ============================================================
# VALIDATE DUPLICATE COLUMNS
# ============================================================

def validate_duplicate_columns(df):

    print("\nColumn validation:")

    duplicate_columns = (
        df.columns[
            df.columns.duplicated()
        ]
        .tolist()
    )

    print(
        f"  Total columns: {len(df.columns)}"
    )

    print(
        f"  Duplicate column names: "
        f"{len(duplicate_columns)}"
    )

    if duplicate_columns:

        print("  FAIL: Duplicate column names:")

        for column in duplicate_columns:
            print(f"    {column}")

        return False

    print(
        "  PASS: No duplicate column names."
    )

    return True


# ============================================================
# VALIDATE GAME-LEVEL IDENTIFIERS
# ============================================================

def validate_game_identifiers(df):

    print("\nGame identifier validation:")

    identifier_columns = [
        "gameId",
        "season",
        "week",
        "startDate",
        "homeTeam",
        "awayTeam",
    ]

    missing = [
        column
        for column in identifier_columns
        if column not in df.columns
    ]

    if missing:

        print(
            "  FAIL: Missing identifier columns:"
        )

        for column in missing:
            print(f"    {column}")

        return False

    # Check that home and away teams are different
    same_team = (
        df["homeTeam"]
        == df["awayTeam"]
    ).sum()

    print(
        f"  Games with identical home/away teams: "
        f"{same_team}"
    )

    if same_team > 0:

        print(
            "  FAIL: Some games have the same "
            "home and away team."
        )

        return False

    print(
        "  PASS: Game identifiers are valid."
    )

    return True


# ============================================================
# CHECK FOR OBVIOUS DUPLICATE DATA
# ============================================================

def validate_duplicate_rows(df):

    print("\nDuplicate row validation:")

    duplicate_rows = df.duplicated().sum()

    print(
        f"  Completely duplicated rows: "
        f"{duplicate_rows}"
    )

    if duplicate_rows > 0:

        print(
            "  FAIL: Completely duplicated rows "
            "were found."
        )

        return False

    print(
        "  PASS: No completely duplicated rows."
    )

    return True


# ============================================================
# DATASET SUMMARY
# ============================================================

def print_dataset_summary(df):

    print("\nDataset summary:")

    print(
        f"  Rows:    {len(df)}"
    )

    print(
        f"  Columns: {len(df.columns)}"
    )

    print(
        f"  Memory:  "
        f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    )


# ============================================================
# VALIDATE ONE SEASON
# ============================================================

def validate_season_file(year):

    print("\n" + "=" * 70)
    print(
        f"VALIDATING FINAL FEATURES: {year}"
    )
    print("=" * 70)

    passed = True

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    try:

        df = load_final_features(year)

    except Exception as e:

        print("\nFAIL: Could not load dataset.")
        print(e)

        return False

    print("\nLoaded data:")
    print(
        f"  Shape: {df.shape}"
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    if not validate_required_columns(df):
        passed = False

    # --------------------------------------------------------
    # Row count
    # --------------------------------------------------------

    if not validate_row_count(df, year):
        passed = False

    # --------------------------------------------------------
    # Game IDs
    # --------------------------------------------------------

    if not validate_game_ids(df):
        passed = False

    # --------------------------------------------------------
    # Game information
    # --------------------------------------------------------

    if not validate_game_information(df):
        passed = False

    # --------------------------------------------------------
    # Season
    # --------------------------------------------------------

    if not validate_season(df, year):
        passed = False

    # --------------------------------------------------------
    # Duplicate columns
    # --------------------------------------------------------

    if not validate_duplicate_columns(df):
        passed = False

    # --------------------------------------------------------
    # Game identifiers
    # --------------------------------------------------------

    if not validate_game_identifiers(df):
        passed = False

    # --------------------------------------------------------
    # Duplicate rows
    # --------------------------------------------------------

    if not validate_duplicate_rows(df):
        passed = False

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_dataset_summary(df)

    # --------------------------------------------------------
    # Final season result
    # --------------------------------------------------------

    print("\nSeason validation result:")

    if passed:

        print(
            f"  PASS: Final feature dataset "
            f"for {year} is valid."
        )

    else:

        print(
            f"  FAIL: Final feature dataset "
            f"for {year} requires investigation."
        )

    return passed


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    all_passed = True

    for year in range(2015, 2026):

        season_passed = validate_season_file(
            year
        )

        if not season_passed:
            all_passed = False

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL VALIDATION RESULT")
    print("=" * 70)

    if all_passed:

        print(
            "PASS: All final feature datasets "
            "passed structural validation."
        )

        print("\nValidated:")
        print("  - Expected row counts")
        print("  - Unique game IDs")
        print("  - Missing game IDs")
        print("  - Required columns")
        print("  - Game information")
        print("  - Season consistency")
        print("  - Duplicate columns")
        print("  - Home/away team identifiers")
        print("  - Completely duplicated rows")

    else:

        print(
            "FAIL: One or more final feature datasets "
            "require investigation."
        )