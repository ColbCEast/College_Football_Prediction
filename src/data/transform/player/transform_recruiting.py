"""
Transform CFBD recruiting player-level data into team-season features.

Input:
    data/raw/player/recruiting/player_recruiting_{year}.csv

Output:
    data/processed/player/recruiting/recruiting_{year}.csv

Initial recruiting features:
    recruiting_count
    recruiting_rated_count
    recruiting_avg_rating
    recruiting_max_rating
    recruiting_5star_count
    recruiting_4star_count
    recruiting_top_100_count

Notes:
    - Only committed recruits are included in team-level aggregates.
    - Missing recruiting ratings are NOT imputed.
    - Rating-based features use only recruits with a non-missing rating.
    - Top-100 counts use only recruits with a non-missing ranking.
    - Raw player-level recruiting data is never modified.
"""

from pathlib import Path

import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(__file__).resolve().parents[4]

RAW_DIR = ROOT / "data" / "raw" / "player" / "recruiting"
PROCESSED_DIR = ROOT / "data" / "processed" / "player" / "recruiting"

START_YEAR = 2015
END_YEAR = 2025
YEARS = range(START_YEAR, END_YEAR + 1)


# =============================================================================
# EXPECTED COLUMNS
# =============================================================================

REQUIRED_COLUMNS = [
    "id",
    "athleteId",
    "recruitType",
    "year",
    "ranking",
    "name",
    "school",
    "committedTo",
    "position",
    "height",
    "weight",
    "stars",
    "rating",
    "city",
    "stateProvince",
    "country",
    "hometownInfo",
]


# =============================================================================
# TRANSFORMATION
# =============================================================================

def transform_recruiting_year(year):
    """
    Transform one season of player-level recruiting data into
    team-season recruiting features.

    Parameters
    ----------
    year : int
        Recruiting season.

    Returns
    -------
    pd.DataFrame
        Team-season recruiting features.
    """

    input_path = RAW_DIR / f"player_recruiting_{year}.csv"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Recruiting file not found for {year}: {input_path}"
        )

    print("=" * 80)
    print(f"TRANSFORMING RECRUITING DATA - {year}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    df = pd.read_csv(input_path)

    print(f"Raw rows: {len(df):,}")

    # -------------------------------------------------------------------------
    # Validate schema
    # -------------------------------------------------------------------------

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{year}: Missing required columns: {missing_columns}"
        )

    # -------------------------------------------------------------------------
    # Validate year
    # -------------------------------------------------------------------------

    if not (df["year"] == year).all():
        raise ValueError(
            f"{year}: File contains unexpected values in the year column."
        )

    # -------------------------------------------------------------------------
    # Restrict to high-school recruiting
    # -------------------------------------------------------------------------
    # This should currently be 100% HighSchool according to the raw-data audit,
    # but retaining the filter makes the transformation explicit.

    df = df[df["recruitType"] == "HighSchool"].copy()

    # -------------------------------------------------------------------------
    # Keep only committed recruits
    # -------------------------------------------------------------------------
    # Uncommitted recruits cannot be assigned to a team-season.

    committed = df[
        df["committedTo"].notna()
        & (df["committedTo"].astype(str).str.strip() != "")
    ].copy()

    print(f"Committed recruits: {len(committed):,}")
    print(
        f"Uncommitted recruits excluded: "
        f"{len(df) - len(committed):,}"
    )

    if committed.empty:
        raise ValueError(
            f"{year}: No committed recruits found."
        )

    # -------------------------------------------------------------------------
    # Clean numeric fields
    # -------------------------------------------------------------------------

    committed["rating"] = pd.to_numeric(
        committed["rating"],
        errors="coerce"
    )

    committed["stars"] = pd.to_numeric(
        committed["stars"],
        errors="coerce"
    )

    committed["ranking"] = pd.to_numeric(
        committed["ranking"],
        errors="coerce"
    )

    # -------------------------------------------------------------------------
    # Aggregate to team-season
    # -------------------------------------------------------------------------

    grouped = committed.groupby(
        ["year", "committedTo"],
        as_index=False
    )

    recruiting = grouped.agg(
        recruiting_count=("id", "size"),

        recruiting_rated_count=(
            "rating",
            lambda x: x.notna().sum()
        ),

        recruiting_avg_rating=(
            "rating",
            "mean"
        ),

        recruiting_max_rating=(
            "rating",
            "max"
        ),

        recruiting_5star_count=(
            "stars",
            lambda x: (x == 5).sum()
        ),

        recruiting_4star_count=(
            "stars",
            lambda x: (x == 4).sum()
        ),

        recruiting_top_100_count=(
            "ranking",
            lambda x: (x <= 100).sum()
        ),
    )

    # -------------------------------------------------------------------------
    # Rename team column
    # -------------------------------------------------------------------------

    recruiting = recruiting.rename(
        columns={
            "committedTo": "team"
        }
    )

    # -------------------------------------------------------------------------
    # Column ordering
    # -------------------------------------------------------------------------

    recruiting = recruiting[
        [
            "year",
            "team",
            "recruiting_count",
            "recruiting_rated_count",
            "recruiting_avg_rating",
            "recruiting_max_rating",
            "recruiting_5star_count",
            "recruiting_4star_count",
            "recruiting_top_100_count",
        ]
    ]

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    # One row per team-season
    duplicate_team_seasons = recruiting.duplicated(
        subset=["year", "team"]
    ).sum()

    if duplicate_team_seasons > 0:
        raise ValueError(
            f"{year}: Found {duplicate_team_seasons} duplicate "
            f"team-season rows."
        )

    # Counts should be logically consistent
    invalid_rated_counts = (
        recruiting["recruiting_rated_count"]
        > recruiting["recruiting_count"]
    ).sum()

    if invalid_rated_counts > 0:
        raise ValueError(
            f"{year}: Found {invalid_rated_counts} rows where "
            "recruiting_rated_count > recruiting_count."
        )

    # Rating statistics should only exist when rated recruits exist
    invalid_rating_rows = (
        recruiting["recruiting_rated_count"].eq(0)
        & (
            recruiting["recruiting_avg_rating"].notna()
            | recruiting["recruiting_max_rating"].notna()
        )
    ).sum()

    if invalid_rating_rows > 0:
        raise ValueError(
            f"{year}: Found {invalid_rating_rows} rows with rating "
            "statistics despite zero rated recruits."
        )

    # Basic range checks
    if (
        recruiting["recruiting_avg_rating"].notna()
        & ~recruiting["recruiting_avg_rating"].between(0, 1)
    ).any():
        raise ValueError(
            f"{year}: recruiting_avg_rating contains values outside [0, 1]."
        )

    if (
        recruiting["recruiting_max_rating"].notna()
        & ~recruiting["recruiting_max_rating"].between(0, 1)
    ).any():
        raise ValueError(
            f"{year}: recruiting_max_rating contains values outside [0, 1]."
        )

    # Stars and ranking counts cannot be negative
    count_columns = [
        "recruiting_count",
        "recruiting_rated_count",
        "recruiting_5star_count",
        "recruiting_4star_count",
        "recruiting_top_100_count",
    ]

    for column in count_columns:
        if (recruiting[column] < 0).any():
            raise ValueError(
                f"{year}: Negative values found in {column}."
            )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        PROCESSED_DIR / f"recruiting_{year}.csv"
    )

    recruiting.to_csv(
        output_path,
        index=False
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print(f"Team-seasons created: {len(recruiting):,}")
    print(f"Unique teams: {recruiting['team'].nunique():,}")
    print(
        f"Rated recruits: "
        f"{recruiting['recruiting_rated_count'].sum():,}"
    )
    print(
        f"Total committed recruits: "
        f"{recruiting['recruiting_count'].sum():,}"
    )
    print(f"Saved: {output_path}")
    print("Status: PASS")
    print()

    return recruiting


# =============================================================================
# ALL SEASONS
# =============================================================================

def transform_all_years():
    """
    Transform recruiting data for all seasons.
    """

    results = []

    for year in YEARS:
        transformed = transform_recruiting_year(year)
        results.append(transformed)

    combined = pd.concat(
        results,
        ignore_index=True
    )

    # -------------------------------------------------------------------------
    # Cross-season validation
    # -------------------------------------------------------------------------

    duplicate_team_seasons = combined.duplicated(
        subset=["year", "team"]
    ).sum()

    if duplicate_team_seasons > 0:
        raise ValueError(
            "Cross-season validation failed: duplicate team-season rows found."
        )

    print("=" * 80)
    print("RECRUITING TRANSFORMATION COMPLETE")
    print("=" * 80)
    print(f"Seasons: {combined['year'].nunique():,}")
    print(f"Rows: {len(combined):,}")
    print(f"Unique teams: {combined['team'].nunique():,}")
    print(
        f"Total committed recruits represented: "
        f"{combined['recruiting_count'].sum():,}"
    )
    print(
        f"Total rated recruits represented: "
        f"{combined['recruiting_rated_count'].sum():,}"
    )
    print("Duplicate team-seasons: 0")
    print("Status: PASS")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    transform_all_years()