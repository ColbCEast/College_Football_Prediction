from pathlib import Path

import pandas as pd


# =============================================================================
# create_full_talent_features.py
#
# Purpose:
#   Combine the existing game-level final features with:
#       1. Returning production (team-season)
#       2. Recruiting (team-season)
#       3. Transfer portal features (game-team)
#
# Output:
#   data/processed/features/full_talent/
#       full_talent_features_{year}.csv
#
# Important:
#   - Existing final_features files are NOT modified.
#   - Final grain remains one row per game.
#   - Returning/recruiting are merged by team + season/year.
#   - Transfers are merged by gameId + team.
#   - Transfer data is unavailable for 2015-2020 and is therefore left
#     missing for those seasons.
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

BASE_FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "final"
)

RETURNING_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player"
    / "returning"
)

RECRUITING_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player"
    / "recruiting"
)

TRANSFER_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player"
    / "portal"
    / "game_features"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "full_talent"
)


# Seasons to process
SEASONS = range(2015, 2026)


# =============================================================================
# EXPECTED COLUMNS
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


# =============================================================================
# HELPERS
# =============================================================================

def require_columns(df, required_columns, dataset_name):
    """Raise an error if required columns are missing."""

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns:\n"
            f"{missing}"
        )


def validate_unique_key(df, key_columns, dataset_name):
    """Raise an error if the specified key is not unique."""

    duplicate_count = df.duplicated(
        subset=key_columns
    ).sum()

    if duplicate_count > 0:
        duplicate_rows = (
            df.loc[
                df.duplicated(
                    subset=key_columns,
                    keep=False
                ),
                key_columns
            ]
            .sort_values(key_columns)
            .head(20)
        )

        raise ValueError(
            f"{dataset_name} contains {duplicate_count:,} "
            f"duplicate rows for key {key_columns}.\n\n"
            f"Example duplicates:\n{duplicate_rows.to_string(index=False)}"
        )


def load_csv(path, dataset_name):
    """Load a CSV and fail with a useful message if unavailable."""

    if not path.exists():
        raise FileNotFoundError(
            f"{dataset_name} file not found:\n{path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            f"{dataset_name} exists but contains no rows:\n{path}"
        )

    return df


# =============================================================================
# RETURNING PRODUCTION
# =============================================================================

def load_returning(year):
    """
    Load team-season returning production.

    Expected grain:
        one row per (season, team)
    """

    path = (
        RETURNING_DIR
        / f"returning_production_{year}.csv"
    )

    df = load_csv(
        path,
        f"Returning production {year}"
    )

    require_columns(
        df,
        ["season", "team"] + RETURNING_FEATURES,
        f"Returning production {year}"
    )

    validate_unique_key(
        df,
        ["season", "team"],
        f"Returning production {year}"
    )

    # Verify season
    seasons = df["season"].dropna().unique()

    if len(seasons) != 1 or seasons[0] != year:
        raise ValueError(
            f"Returning production {year} contains unexpected "
            f"season values: {seasons}"
        )

    return df[
        ["season", "team"] + RETURNING_FEATURES
    ].copy()


# =============================================================================
# RECRUITING
# =============================================================================

def load_recruiting(year):
    """
    Load team-season recruiting.

    Expected grain:
        one row per (year, team)
    """

    path = (
        RECRUITING_DIR
        / f"recruiting_{year}.csv"
    )

    df = load_csv(
        path,
        f"Recruiting {year}"
    )

    require_columns(
        df,
        ["year", "team"] + RECRUITING_FEATURES,
        f"Recruiting {year}"
    )

    validate_unique_key(
        df,
        ["year", "team"],
        f"Recruiting {year}"
    )

    # Verify year
    years = df["year"].dropna().unique()

    if len(years) != 1 or years[0] != year:
        raise ValueError(
            f"Recruiting {year} contains unexpected "
            f"year values: {years}"
        )

    return df[
        ["year", "team"] + RECRUITING_FEATURES
    ].copy()


# =============================================================================
# TRANSFERS
# =============================================================================

def load_transfer(year):
    """
    Load game-team transfer features.

    Expected grain:
        one row per (gameId, team)

    Transfer data begins in 2021.

    For 2015-2020:
        return None because the CFBD transfer endpoint contains
        no historical records for those seasons.
    """

    if year < 2021:
        return None

    path = (
        TRANSFER_DIR
        / f"transfer_game_features_{year}.csv"
    )

    df = load_csv(
        path,
        f"Transfer game features {year}"
    )

    require_columns(
        df,
        [
            "gameId",
            "season",
            "team",
            "homeAway",
        ] + TRANSFER_FEATURES,
        f"Transfer game features {year}"
    )

    validate_unique_key(
        df,
        ["gameId", "team"],
        f"Transfer game features {year}"
    )

    # Verify season
    seasons = df["season"].dropna().unique()

    if len(seasons) != 1 or seasons[0] != year:
        raise ValueError(
            f"Transfer game features {year} contains unexpected "
            f"season values: {seasons}"
        )

    # Verify home/away values
    unexpected_home_away = set(
        df["homeAway"].dropna().unique()
    ) - {"home", "away"}

    if unexpected_home_away:
        raise ValueError(
            f"Transfer game features {year} contains unexpected "
            f"homeAway values: {unexpected_home_away}"
        )

    return df[
        [
            "gameId",
            "season",
            "team",
            "homeAway",
        ] + TRANSFER_FEATURES
    ].copy()


# =============================================================================
# BUILD RETURNING FEATURES
# =============================================================================

def create_returning_game_features(base_df, returning_df, year):
    """
    Convert team-season returning data into home/away game features.
    """

    teams = (
        base_df[
            ["gameId", "season", "homeTeam", "awayTeam"]
        ]
        .copy()
    )

    home = teams[
        ["gameId", "season", "homeTeam"]
    ].rename(
        columns={
            "homeTeam": "team"
        }
    )

    away = teams[
        ["gameId", "season", "awayTeam"]
    ].rename(
        columns={
            "awayTeam": "team"
        }
    )

    # -------------------------------------------------------------------------
    # Home
    # -------------------------------------------------------------------------

    home = home.merge(
        returning_df,
        on=["season", "team"],
        how="left",
        validate="many_to_one",
    )

    home = home.drop(columns=["team"])

    home = home.rename(
        columns={
            column: f"home_{column}"
            for column in RETURNING_FEATURES
        }
    )

    # -------------------------------------------------------------------------
    # Away
    # -------------------------------------------------------------------------

    away = away.merge(
        returning_df,
        on=["season", "team"],
        how="left",
        validate="many_to_one",
    )

    away = away.drop(columns=["team"])

    away = away.rename(
        columns={
            column: f"away_{column}"
            for column in RETURNING_FEATURES
        }
    )

    # -------------------------------------------------------------------------
    # Combine
    # -------------------------------------------------------------------------

    result = home.merge(
        away,
        on=["gameId", "season"],
        how="inner",
        validate="one_to_one",
    )

    if len(result) != len(base_df):
        raise ValueError(
            f"Returning game-feature construction changed the number "
            f"of games for {year}: "
            f"{len(base_df):,} base games -> {len(result):,} games"
        )

    return result


# =============================================================================
# BUILD RECRUITING FEATURES
# =============================================================================

def create_recruiting_game_features(base_df, recruiting_df, year):
    """
    Convert team-season recruiting data into home/away game features.
    """

    teams = (
        base_df[
            ["gameId", "season", "homeTeam", "awayTeam"]
        ]
        .copy()
    )

    home = teams[
        ["gameId", "season", "homeTeam"]
    ].rename(
        columns={
            "homeTeam": "team"
        }
    )

    away = teams[
        ["gameId", "season", "awayTeam"]
    ].rename(
        columns={
            "awayTeam": "team"
        }
    )

    # -------------------------------------------------------------------------
    # Recruiting uses "year" instead of "season".
    # Rename for consistent merge.
    # -------------------------------------------------------------------------

    recruiting = recruiting_df.rename(
        columns={"year": "season"}
    ).copy()

    # -------------------------------------------------------------------------
    # Home
    # -------------------------------------------------------------------------

    home = home.merge(
        recruiting,
        on=["season", "team"],
        how="left",
        validate="many_to_one",
    )

    home = home.drop(columns=["team"])

    home = home.rename(
        columns={
            column: f"home_{column}"
            for column in RECRUITING_FEATURES
        }
    )

    # -------------------------------------------------------------------------
    # Away
    # -------------------------------------------------------------------------

    away = away.merge(
        recruiting,
        on=["season", "team"],
        how="left",
        validate="many_to_one",
    )

    away = away.drop(columns=["team"])

    away = away.rename(
        columns={
            column: f"away_{column}"
            for column in RECRUITING_FEATURES
        }
    )

    # -------------------------------------------------------------------------
    # Combine
    # -------------------------------------------------------------------------

    result = home.merge(
        away,
        on=["gameId", "season"],
        how="inner",
        validate="one_to_one",
    )

    if len(result) != len(base_df):
        raise ValueError(
            f"Recruiting game-feature construction changed the number "
            f"of games for {year}: "
            f"{len(base_df):,} base games -> {len(result):,} games"
        )

    return result


# =============================================================================
# BUILD TRANSFER FEATURES
# =============================================================================

def create_transfer_game_features(base_df, transfer_df, year):
    """
    Convert game-team transfer data into home/away game features.

    The transfer source already contains:
        gameId
        team
        homeAway

    Therefore, it can be pivoted directly into home/away columns.
    """

    if transfer_df is None:
        return pd.DataFrame(
            index=base_df.index
        ).assign(
            gameId=base_df["gameId"].values
        )

    # -------------------------------------------------------------------------
    # Keep only the columns needed for the final game-level layer.
    # -------------------------------------------------------------------------

    transfer = transfer_df[
        ["gameId", "homeAway"] + TRANSFER_FEATURES
    ].copy()

    # -------------------------------------------------------------------------
    # Separate home and away.
    # -------------------------------------------------------------------------

    home = transfer[
        transfer["homeAway"] == "home"
    ].copy()

    away = transfer[
        transfer["homeAway"] == "away"
    ].copy()

    # -------------------------------------------------------------------------
    # Each game should have exactly one home and one away transfer row.
    # -------------------------------------------------------------------------

    validate_unique_key(
        home,
        ["gameId"],
        f"Transfer home rows {year}"
    )

    validate_unique_key(
        away,
        ["gameId"],
        f"Transfer away rows {year}"
    )

    # -------------------------------------------------------------------------
    # Rename features.
    # -------------------------------------------------------------------------

    home = home.drop(columns=["homeAway"])

    home = home.rename(
        columns={
            column: f"home_{column}"
            for column in TRANSFER_FEATURES
        }
    )

    away = away.drop(columns=["homeAway"])

    away = away.rename(
        columns={
            column: f"away_{column}"
            for column in TRANSFER_FEATURES
        }
    )

    # -------------------------------------------------------------------------
    # Merge home and away.
    # -------------------------------------------------------------------------

    result = home.merge(
        away,
        on="gameId",
        how="outer",
        validate="one_to_one",
    )

    # -------------------------------------------------------------------------
    # Every transfer-feature game should exist in the base data.
    # -------------------------------------------------------------------------

    base_game_ids = set(base_df["gameId"])
    transfer_game_ids = set(result["gameId"])

    unexpected_transfer_games = (
        transfer_game_ids - base_game_ids
    )

    if unexpected_transfer_games:
        raise ValueError(
            f"Transfer data for {year} contains "
            f"{len(unexpected_transfer_games):,} gameIds "
            f"not present in final_features_{year}.csv."
        )

    # -------------------------------------------------------------------------
    # Merge onto base games.
    # -------------------------------------------------------------------------

    result = base_df[
        ["gameId"]
    ].merge(
        result,
        on="gameId",
        how="left",
        validate="one_to_one",
    )

    if len(result) != len(base_df):
        raise ValueError(
            f"Transfer game-feature construction changed the number "
            f"of games for {year}: "
            f"{len(base_df):,} base games -> {len(result):,} games"
        )

    return result


# =============================================================================
# BUILD ONE SEASON
# =============================================================================

def create_full_talent_features(year):
    """Create the combined full-talent game-level feature dataset."""

    print()
    print("=" * 100)
    print(f"PROCESSING {year}")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # Load base features
    # -------------------------------------------------------------------------

    base_path = (
        BASE_FEATURE_DIR
        / f"final_features_{year}.csv"
    )

    base_df = load_csv(
        base_path,
        f"Final features {year}"
    )

    require_columns(
        base_df,
        [
            "gameId",
            "season",
            "homeTeam",
            "awayTeam",
        ],
        f"Final features {year}"
    )

    validate_unique_key(
        base_df,
        ["gameId"],
        f"Final features {year}"
    )

    base_rows = len(base_df)
    base_columns = set(base_df.columns)

    print(f"Base rows:    {base_rows:,}")
    print(f"Base columns: {len(base_df.columns):,}")

    # -------------------------------------------------------------------------
    # Verify base season
    # -------------------------------------------------------------------------

    seasons = base_df["season"].dropna().unique()

    if len(seasons) != 1 or seasons[0] != year:
        raise ValueError(
            f"Final features {year} contains unexpected "
            f"season values: {seasons}"
        )

    # -------------------------------------------------------------------------
    # Load talent sources
    # -------------------------------------------------------------------------

    returning_df = load_returning(year)
    recruiting_df = load_recruiting(year)
    transfer_df = load_transfer(year)

    print(f"Returning rows:  {len(returning_df):,}")
    print(f"Recruiting rows: {len(recruiting_df):,}")

    if transfer_df is None:
        print("Transfer rows:   0 (source unavailable before 2021)")
    else:
        print(f"Transfer rows:   {len(transfer_df):,}")

    # -------------------------------------------------------------------------
    # Create game-level talent features
    # -------------------------------------------------------------------------

    returning_game = create_returning_game_features(
        base_df,
        returning_df,
        year,
    )

    recruiting_game = create_recruiting_game_features(
        base_df,
        recruiting_df,
        year,
    )

    transfer_game = create_transfer_game_features(
        base_df,
        transfer_df,
        year,
    )

    # -------------------------------------------------------------------------
    # Start with the original base data.
    # -------------------------------------------------------------------------

    result = base_df.copy()

    # -------------------------------------------------------------------------
    # Merge returning features.
    # -------------------------------------------------------------------------

    result = result.merge(
        returning_game,
        on=["gameId", "season"],
        how="left",
        validate="one_to_one",
    )

    if len(result) != base_rows:
        raise ValueError(
            f"Returning merge changed row count for {year}."
        )

    # -------------------------------------------------------------------------
    # Merge recruiting features.
    # -------------------------------------------------------------------------

    result = result.merge(
        recruiting_game,
        on=["gameId", "season"],
        how="left",
        validate="one_to_one",
    )

    if len(result) != base_rows:
        raise ValueError(
            f"Recruiting merge changed row count for {year}."
        )

    # -------------------------------------------------------------------------
    # Merge transfer features.
    #
    # For 2015-2020, transfer_game contains only gameId and therefore
    # contributes no transfer feature columns.
    # -------------------------------------------------------------------------

    result = result.merge(
        transfer_game,
        on="gameId",
        how="left",
        validate="one_to_one",
    )

    if len(result) != base_rows:
        raise ValueError(
            f"Transfer merge changed row count for {year}."
        )

    # -------------------------------------------------------------------------
    # Ensure no original base columns were changed or removed.
    # -------------------------------------------------------------------------

    missing_base_columns = [
        column
        for column in base_df.columns
        if column not in result.columns
    ]

    if missing_base_columns:
        raise ValueError(
            f"Original base columns disappeared for {year}:\n"
            f"{missing_base_columns}"
        )

    # -------------------------------------------------------------------------
    # Check gameId uniqueness.
    # -------------------------------------------------------------------------

    validate_unique_key(
        result,
        ["gameId"],
        f"Full talent features {year}"
    )

    # -------------------------------------------------------------------------
    # Verify final row count.
    # -------------------------------------------------------------------------

    if len(result) != base_rows:
        raise ValueError(
            f"Final row count mismatch for {year}: "
            f"{base_rows:,} -> {len(result):,}"
        )

    # -------------------------------------------------------------------------
    # Identify newly added features.
    # -------------------------------------------------------------------------

    added_columns = [
        column
        for column in result.columns
        if column not in base_columns
    ]

    # -------------------------------------------------------------------------
    # Create output directory.
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"full_talent_features_{year}.csv"
    )

    result.to_csv(
        output_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print()
    print("INTEGRATION SUMMARY")
    print("-" * 100)
    print(f"Final rows:             {len(result):,}")
    print(f"Final columns:          {len(result.columns):,}")
    print(f"Added talent columns:   {len(added_columns):,}")
    print(f"Output:                 {output_path}")

    print("\nAdded columns:")

    for column in added_columns:
        print(f"  {column}")

    # -------------------------------------------------------------------------
    # Missingness summary for talent features.
    # -------------------------------------------------------------------------

    if added_columns:
        print("\nTalent-feature missingness:")

        missingness = (
            result[added_columns]
            .isna()
            .mean()
            .sort_values(ascending=False)
        )

        for column, pct in missingness.items():
            print(
                f"  {column:<50} {pct:>7.2%}"
            )

    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 100)
    print("FULL PLAYER-TALENT FEATURE CREATION")
    print("=" * 100)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Output directory: {OUTPUT_DIR}")

    for year in SEASONS:
        create_full_talent_features(year)

    print()
    print("=" * 100)
    print("ALL SEASONS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()