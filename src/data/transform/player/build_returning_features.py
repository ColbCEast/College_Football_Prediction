"""
Build game-level returning-production features for win-probability modeling.

Purpose
-------
Merge team-season returning-production data onto the game-level final feature
dataset, handle teams without CFBD returning-production observations through
season-specific FBS median imputation, create missingness indicators, and
construct returning-production x games-before interaction features.

Inputs
------
data/processed/features/final/final_features_{year}.csv
data/processed/player/returning/returning_production_{year}.csv

Outputs
-------
data/processed/features/win_probability/returning_production/
    returning_features_{year}.csv

Notes
-----
- Returning-production values represent preseason information.
- Missing returning-production observations are imputed using season-specific
  medians calculated from available FBS returning-production observations.
- Missingness indicators identify teams whose returning-production values were
  imputed.
- gamesBefore_home and gamesBefore_away already exist in the final feature
  files and are not recreated here.
- Source percentage-PPA variables are retained without clipping or
  transformation.
- Percentage-PPA variables are not initially included in gamesBefore
  interaction features because of their unusual distributions identified
  during the returning-production audit.
"""

from pathlib import Path

import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[4]

FINAL_FEATURES_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "final"
)

RETURNING_PRODUCTION_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "player"
    / "returning"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "win_probability"
    / "returning_production"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

EXPECTED_SEASONS = list(range(2015, 2026))

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

# Core PPA and usage variables used for the initial gamesBefore interactions.
# Percentage-PPA variables are retained as features but are not initially
# interacted with gamesBefore.
INTERACTION_FEATURES = [
    "returning_total_ppa",
    "returning_passing_ppa",
    "returning_receiving_ppa",
    "returning_rushing_ppa",
    "returning_usage",
    "returning_passing_usage",
    "returning_receiving_usage",
    "returning_rushing_usage",
]

REQUIRED_GAME_COLUMNS = [
    "season",
    "gameId",
    "homeTeam",
    "awayTeam",
    "gamesBefore_home",
    "gamesBefore_away",
]

REQUIRED_RETURNING_COLUMNS = [
    "season",
    "team",
] + RETURNING_FEATURES


# =============================================================================
# VALIDATION HELPERS
# =============================================================================

def validate_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    dataset_name: str,
) -> None:
    """Validate that all required columns are present."""

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{missing_columns}"
        )


def validate_returning_keys(
    returning: pd.DataFrame,
    year: int,
) -> None:
    """Validate uniqueness of returning-production team-season keys."""

    duplicate_mask = returning.duplicated(
        subset=["season", "team"],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = (
            returning.loc[
                duplicate_mask,
                ["season", "team"],
            ]
            .drop_duplicates()
            .sort_values(["season", "team"])
        )

        raise ValueError(
            f"{year}: duplicate returning-production team-season "
            f"keys found:\n{duplicates.to_string(index=False)}"
        )


def validate_games_before(
    games: pd.DataFrame,
    year: int,
) -> None:
    """Validate the existing gamesBefore variables."""

    for column in [
        "gamesBefore_home",
        "gamesBefore_away",
    ]:
        if games[column].isna().any():
            raise ValueError(
                f"{year}: {column} contains missing values."
            )

        if (games[column] < 0).any():
            raise ValueError(
                f"{year}: {column} contains negative values."
            )


# =============================================================================
# RETURNING-PRODUCTION HELPERS
# =============================================================================

def calculate_season_medians(
    returning: pd.DataFrame,
    year: int,
) -> pd.Series:
    """
    Calculate season-specific medians from available returning-production
    observations.
    """

    medians = returning[RETURNING_FEATURES].median()

    if medians.isna().any():
        missing_medians = (
            medians[medians.isna()]
            .index
            .tolist()
        )

        raise ValueError(
            f"{year}: unable to calculate medians for "
            f"{missing_medians}"
        )

    return medians


def merge_team_returning_features(
    games: pd.DataFrame,
    returning: pd.DataFrame,
    team_column: str,
    side: str,
) -> pd.DataFrame:
    """
    Merge returning-production features onto either the home or away team.
    """

    returning_for_merge = returning[
        ["season", "team"] + RETURNING_FEATURES
    ].copy()

    rename_map = {
        feature: f"{side}_{feature}"
        for feature in RETURNING_FEATURES
    }

    returning_for_merge = returning_for_merge.rename(
        columns=rename_map
    )

    team_key = f"{side}_team_key"

    games = games.copy()
    games[team_key] = games[team_column]

    returning_for_merge = returning_for_merge.rename(
        columns={"team": team_key}
    )

    games = games.merge(
        returning_for_merge,
        how="left",
        on=["season", team_key],
        validate="many_to_one",
        indicator=f"_{side}_returning_merge",
    )

    return games


def add_missingness_indicator(
    games: pd.DataFrame,
    side: str,
) -> pd.DataFrame:
    """
    Create a missingness indicator before median imputation.

    1 = no returning-production record was found.
    0 = returning-production record was found.
    """

    merge_indicator = f"_{side}_returning_merge"
    missing_indicator = f"{side}_returning_missing"

    games[missing_indicator] = (
        games[merge_indicator] == "left_only"
    ).astype("int8")

    return games


def impute_returning_features(
    games: pd.DataFrame,
    medians: pd.Series,
    side: str,
) -> pd.DataFrame:
    """Impute missing returning-production values using season medians."""

    for feature in RETURNING_FEATURES:

        column = f"{side}_{feature}"

        missing_mask = games[column].isna()

        if missing_mask.any():
            games.loc[missing_mask, column] = medians[feature]

    return games


# =============================================================================
# INTERACTION FEATURES
# =============================================================================

def add_interaction_features(
    games: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create returning-production x gamesBefore interaction features.

    Percentage-PPA variables are intentionally excluded from these initial
    interactions.
    """

    for side in ["home", "away"]:

        games_before_column = f"gamesBefore_{side}"

        for feature in INTERACTION_FEATURES:

            returning_column = f"{side}_{feature}"

            interaction_column = (
                f"{returning_column}_x_{games_before_column}"
            )

            games[interaction_column] = (
                games[returning_column]
                * games[games_before_column]
            )

    return games


# =============================================================================
# OUTPUT VALIDATION
# =============================================================================

def validate_interactions(
    games: pd.DataFrame,
    year: int,
) -> None:
    """Validate all returning-production interaction calculations."""

    for side in ["home", "away"]:

        games_before_column = f"gamesBefore_{side}"

        for feature in INTERACTION_FEATURES:

            returning_column = f"{side}_{feature}"

            interaction_column = (
                f"{returning_column}_x_{games_before_column}"
            )

            expected = (
                games[returning_column]
                * games[games_before_column]
            )

            if not (
                expected
                .reset_index(drop=True)
                .equals(
                    games[interaction_column]
                    .reset_index(drop=True)
                )
            ):
                raise ValueError(
                    f"{year}: interaction validation failed for "
                    f"{interaction_column}."
                )


def validate_output(
    original_game_ids: pd.Series,
    original_row_count: int,
    games: pd.DataFrame,
    year: int,
) -> None:
    """Perform final structural validation of the output."""

    # -------------------------------------------------------------------------
    # Row-count preservation
    # -------------------------------------------------------------------------

    if len(games) != original_row_count:
        raise ValueError(
            f"{year}: row count changed during feature construction. "
            f"Original: {original_row_count:,}; "
            f"Output: {len(games):,}"
        )

    # -------------------------------------------------------------------------
    # Game ID preservation
    # -------------------------------------------------------------------------

    output_game_ids = games["gameId"]

    if output_game_ids.duplicated().any():
        raise ValueError(
            f"{year}: duplicate gameId values found in output."
        )

    if not (
        original_game_ids
        .reset_index(drop=True)
        .equals(
            output_game_ids
            .reset_index(drop=True)
        )
    ):
        raise ValueError(
            f"{year}: gameId values or ordering changed during "
            f"feature construction."
        )

    # -------------------------------------------------------------------------
    # Required engineered columns
    # -------------------------------------------------------------------------

    required_new_columns = []

    for side in ["home", "away"]:

        required_new_columns.extend(
            f"{side}_{feature}"
            for feature in RETURNING_FEATURES
        )

        required_new_columns.append(
            f"{side}_returning_missing"
        )

        required_new_columns.extend(
            f"{side}_{feature}_x_gamesBefore_{side}"
            for feature in INTERACTION_FEATURES
        )

    missing_new_columns = [
        column
        for column in required_new_columns
        if column not in games.columns
    ]

    if missing_new_columns:
        raise ValueError(
            f"{year}: expected engineered columns are missing: "
            f"{missing_new_columns}"
        )

    # -------------------------------------------------------------------------
    # No missing returning-production values after imputation
    # -------------------------------------------------------------------------

    returning_columns = [
        f"{side}_{feature}"
        for side in ["home", "away"]
        for feature in RETURNING_FEATURES
    ]

    remaining_missing = games[returning_columns].isna().sum()

    if remaining_missing.sum() > 0:
        raise ValueError(
            f"{year}: missing returning-production values remain:\n"
            f"{remaining_missing[remaining_missing > 0]}"
        )

    # -------------------------------------------------------------------------
    # Missingness indicator validation
    # -------------------------------------------------------------------------

    for column in [
        "home_returning_missing",
        "away_returning_missing",
    ]:

        invalid_values = ~games[column].isin([0, 1])

        if invalid_values.any():
            raise ValueError(
                f"{year}: {column} contains values other than 0 or 1."
            )


# =============================================================================
# SEASON PROCESSING
# =============================================================================

def build_returning_features_for_year(
    year: int,
) -> pd.DataFrame:
    """Build returning-production features for one season."""

    final_features_path = (
        FINAL_FEATURES_DIR
        / f"final_features_{year}.csv"
    )

    returning_path = (
        RETURNING_PRODUCTION_DIR
        / f"returning_production_{year}.csv"
    )

    if not final_features_path.exists():
        raise FileNotFoundError(
            f"Final feature file not found:\n"
            f"{final_features_path}"
        )

    if not returning_path.exists():
        raise FileNotFoundError(
            f"Returning-production file not found:\n"
            f"{returning_path}"
        )

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------

    games = pd.read_csv(final_features_path)
    returning = pd.read_csv(returning_path)

    original_row_count = len(games)
    original_game_ids = games["gameId"].copy()

    # -------------------------------------------------------------------------
    # Validate inputs
    # -------------------------------------------------------------------------

    validate_columns(
        games,
        REQUIRED_GAME_COLUMNS,
        f"Final features {year}",
    )

    validate_columns(
        returning,
        REQUIRED_RETURNING_COLUMNS,
        f"Returning production {year}",
    )

    validate_returning_keys(
        returning,
        year,
    )

    validate_games_before(
        games,
        year,
    )

    # -------------------------------------------------------------------------
    # Validate returning-production season
    # -------------------------------------------------------------------------

    returning_seasons = (
        returning["season"]
        .drop_duplicates()
        .tolist()
    )

    if returning_seasons != [year]:
        raise ValueError(
            f"{year}: returning-production file contains unexpected "
            f"seasons: {returning_seasons}"
        )

    # -------------------------------------------------------------------------
    # Calculate season-specific FBS medians
    # -------------------------------------------------------------------------

    medians = calculate_season_medians(
        returning,
        year,
    )

    # -------------------------------------------------------------------------
    # Merge home returning production
    # -------------------------------------------------------------------------

    games = merge_team_returning_features(
        games=games,
        returning=returning,
        team_column="homeTeam",
        side="home",
    )

    # -------------------------------------------------------------------------
    # Merge away returning production
    # -------------------------------------------------------------------------

    games = merge_team_returning_features(
        games=games,
        returning=returning,
        team_column="awayTeam",
        side="away",
    )

    # -------------------------------------------------------------------------
    # Create missingness indicators BEFORE imputation
    # -------------------------------------------------------------------------

    games = add_missingness_indicator(
        games,
        side="home",
    )

    games = add_missingness_indicator(
        games,
        side="away",
    )

    # -------------------------------------------------------------------------
    # Impute missing returning-production values
    # -------------------------------------------------------------------------

    games = impute_returning_features(
        games,
        medians,
        side="home",
    )

    games = impute_returning_features(
        games,
        medians,
        side="away",
    )

    # -------------------------------------------------------------------------
    # Remove temporary merge columns
    # -------------------------------------------------------------------------

    games = games.drop(
        columns=[
            "home_team_key",
            "away_team_key",
            "_home_returning_merge",
            "_away_returning_merge",
        ]
    )

    # -------------------------------------------------------------------------
    # Create interaction features
    # -------------------------------------------------------------------------

    games = add_interaction_features(
        games,
    )

    # -------------------------------------------------------------------------
    # Validate output
    # -------------------------------------------------------------------------

    validate_interactions(
        games,
        year,
    )

    validate_output(
        original_game_ids=original_game_ids,
        original_row_count=original_row_count,
        games=games,
        year=year,
    )

    # -------------------------------------------------------------------------
    # Save output
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"returning_features_{year}.csv"
    )

    games.to_csv(
        output_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    home_imputed = int(
        games["home_returning_missing"].sum()
    )

    away_imputed = int(
        games["away_returning_missing"].sum()
    )

    print(
        f"{year}: "
        f"{len(games):,} games | "
        f"home imputed: {home_imputed:,} | "
        f"away imputed: {away_imputed:,} | "
        f"columns: {len(games.columns):,}"
    )

    print(f"  Output: {output_path}")

    return games


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Build returning-production features for all seasons."""

    print("=" * 80)
    print("BUILD RETURNING-PRODUCTION FEATURES")
    print("=" * 80)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Final features: {FINAL_FEATURES_DIR}")
    print(f"Returning production: {RETURNING_PRODUCTION_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    processed = 0

    for year in EXPECTED_SEASONS:

        build_returning_features_for_year(year)
        processed += 1

    print()
    print("=" * 80)
    print("BUILD COMPLETE")
    print("=" * 80)
    print(f"Seasons processed: {processed}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()