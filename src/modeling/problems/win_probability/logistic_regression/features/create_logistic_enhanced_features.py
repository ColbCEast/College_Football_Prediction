"""
Create enhanced logistic-regression features.

This script extends the existing validated final feature set with:

1. Recent-form / trajectory features
   - Difference between last-3-game average and season-to-date average
   - Calculated from already validated temporal features in game_team_stats

2. Prior strength-of-schedule features
   - Average prior-opponent win percentage
   - Average prior-opponent point differential
   - Opponent strength is measured using information available BEFORE
     the matchup against that opponent.

The existing final_features datasets are NOT modified.

Output:
    data/processed/logistic_enhanced_features/
        logistic_features_{year}.csv

Design:
    game_team_stats
        -> team-game historical features
        -> new team-game features
        -> game-level home/away features
        -> merge onto final_features
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[6]

GAME_TEAM_STATS_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "game_stats"
    / "team_level"
)

FINAL_FEATURES_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "final"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "win_probability"
    / "logistic_regression"
    / "enhanced"
)


YEARS = list(range(2015, 2026))


# =============================================================================
# CONFIGURATION
# =============================================================================

# Continuous recent-form features.
#
# Each tuple is:
#     (new feature name, last-3 column, season-to-date column)
#
# The season-to-date column represents information available BEFORE
# the current game.
TREND_FEATURES = {
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


# SOS features.
#
# These are calculated from opponent information available before
# the current game.
SOS_FEATURES = [
    "priorSOSWinPct",
    "priorSOSPointDiff",
]


# =============================================================================
# UTILITIES
# =============================================================================

def print_section(title):
    """Print a formatted section header."""

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def fail(message):
    """Raise a clear validation error."""

    raise ValueError(f"\nVALIDATION FAILED:\n{message}")


# =============================================================================
# LOAD DATA
# =============================================================================

def load_year_data(year):
    """Load game-team statistics and final features for one season."""

    game_team_path = (
        GAME_TEAM_STATS_DIR
        / f"game_team_stats_{year}.csv"
    )

    final_features_path = (
        FINAL_FEATURES_DIR
        / f"final_features_{year}.csv"
    )

    if not game_team_path.exists():
        fail(
            f"Missing game-team statistics file:\n"
            f"{game_team_path}"
        )

    if not final_features_path.exists():
        fail(
            f"Missing final feature file:\n"
            f"{final_features_path}"
        )

    game_team = pd.read_csv(game_team_path)
    final_features = pd.read_csv(final_features_path)

    return game_team, final_features


# =============================================================================
# BASIC VALIDATION
# =============================================================================

def validate_game_team_structure(df, year):
    """Validate the expected two-team-rows-per-game structure."""

    required = [
        "gameId",
        "season",
        "week",
        "startDate",
        "team",
        "opponent",
        "homeAway",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        fail(
            f"{year}: Missing required game-team columns: "
            f"{missing}"
        )

    if df["gameId"].isna().any():
        fail(
            f"{year}: gameId contains missing values."
        )

    if df[["gameId", "team"]].duplicated().any():
        duplicates = (
            df.loc[
                df[["gameId", "team"]].duplicated(
                    keep=False
                ),
                ["gameId", "team"],
            ]
            .sort_values(["gameId", "team"])
        )

        fail(
            f"{year}: Duplicate gameId/team combinations found.\n"
            f"{duplicates.head(20).to_string(index=False)}"
        )

    game_counts = (
        df.groupby("gameId")
        .size()
    )

    invalid_counts = game_counts[
        game_counts != 2
    ]

    if not invalid_counts.empty:
        fail(
            f"{year}: {len(invalid_counts)} games do not "
            f"have exactly two team rows.\n"
            f"{invalid_counts.head(20)}"
        )

    home_counts = (
        df.groupby("gameId")["homeAway"]
        .apply(lambda x: (x == "home").sum())
    )

    away_counts = (
        df.groupby("gameId")["homeAway"]
        .apply(lambda x: (x == "away").sum())
    )

    if not (home_counts == 1).all():
        fail(
            f"{year}: Some games do not have exactly "
            f"one home team row."
        )

    if not (away_counts == 1).all():
        fail(
            f"{year}: Some games do not have exactly "
            f"one away team row."
        )

    print(
        f"Team-game rows: {len(df):,}"
    )

    print(
        f"Unique games:   {df['gameId'].nunique():,}"
    )

    print(
        "Structure:       VALID"
    )


def validate_final_features(df, year):
    """Validate one-row-per-game final features."""

    if "gameId" not in df.columns:
        fail(
            f"{year}: final_features does not contain gameId."
        )

    if df["gameId"].duplicated().any():
        duplicates = (
            df.loc[
                df["gameId"].duplicated(
                    keep=False
                ),
                "gameId",
            ]
            .value_counts()
        )

        fail(
            f"{year}: final_features contains duplicate gameIds.\n"
            f"{duplicates.head(20)}"
        )

    print(
        f"Final-feature rows: {len(df):,}"
    )

    print(
        f"Final-feature games: {df['gameId'].nunique():,}"
    )

    print(
        "Final-feature grain: VALID"
    )


# =============================================================================
# TEMPORAL VALIDATION
# =============================================================================

def validate_temporal_order(df, year):
    """
    Validate that each team's games can be chronologically ordered.

    startDate is the primary ordering variable.
    gameId provides a deterministic tie-breaker.
    """

    temp = df.copy()

    temp["startDate"] = pd.to_datetime(
        temp["startDate"],
        errors="coerce",
        utc=True,
    )

    if temp["startDate"].isna().any():
        fail(
            f"{year}: startDate contains invalid/missing values."
        )

    temp = temp.sort_values(
        [
            "season",
            "team",
            "startDate",
            "gameId",
        ]
    )

    duplicate_positions = (
        temp.groupby(
            [
                "season",
                "team",
                "startDate",
                "gameId",
            ]
        )
        .size()
    )

    if (duplicate_positions > 1).any():
        fail(
            f"{year}: Duplicate chronological team-game positions found."
        )

    print(
        "Temporal ordering: VALID"
    )

    return temp


# =============================================================================
# TREND FEATURES
# =============================================================================

def create_trend_features(df, year):
    """
    Create recent-form trend features.

    Trend = last-3 average - prior season average.

    Both source columns already represent information available
    before the current game.
    """

    print_section(
        f"CREATING RECENT-FORM FEATURES — {year}"
    )

    result = df.copy()

    created = []

    for new_name, (
        last3_column,
        season_column,
    ) in TREND_FEATURES.items():

        required = [
            last3_column,
            season_column,
        ]

        missing = [
            column
            for column in required
            if column not in result.columns
        ]

        if missing:
            fail(
                f"{year}: Cannot create {new_name}. "
                f"Missing columns: {missing}"
            )

        result[new_name] = (
            result[last3_column]
            - result[season_column]
        )

        created.append(new_name)

    print(
        f"Created {len(created)} trend features:"
    )

    for feature in created:
        print(f"  {feature}")

    return result


# =============================================================================
# OPPONENT STRENGTH
# =============================================================================

def create_opponent_strength_features(df, year):
    """
    Attach each opponent's pregame strength to the current team-game.

    For each game:

        Team A vs Team B

    Team A receives Team B's:

        winPctBefore
        pointDifferentialAvgBefore

    and vice versa.

    These values describe the opponent using information available
    BEFORE the current game.
    """

    print_section(
        f"CREATING OPPONENT PREGAME STRENGTH — {year}"
    )

    required = [
        "gameId",
        "team",
        "opponent",
        "winPctBefore",
        "pointDifferentialAvgBefore",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        fail(
            f"{year}: Missing columns required for opponent strength: "
            f"{missing}"
        )

    opponent_lookup = df[
        [
            "gameId",
            "team",
            "winPctBefore",
            "pointDifferentialAvgBefore",
        ]
    ].copy()

    opponent_lookup = opponent_lookup.rename(
        columns={
            "team": "opponent",
            "winPctBefore": "opponentWinPctBefore",
            "pointDifferentialAvgBefore":
                "opponentPointDiffBefore",
        }
    )

    result = df.merge(
        opponent_lookup,
        on=["gameId", "opponent"],
        how="left",
        validate="one_to_one",
    )

    if len(result) != len(df):
        fail(
            f"{year}: Opponent merge changed row count."
        )

    return result


# =============================================================================
# PRIOR SOS
# =============================================================================

def create_prior_sos(df, year):
    """
    Calculate prior strength of schedule.

    For each team and current game:

        priorSOSWinPct
            = average opponent pregame win percentage
              across prior games

        priorSOSPointDiff
            = average opponent pregame point differential
              across prior games

    IMPORTANT:
        The current opponent is NOT included in the current game's SOS.
        The opponent's strength is first attached to the game and then
        shifted before the cumulative/expanding calculation.
    """

    print_section(
        f"CREATING PRIOR SOS FEATURES — {year}"
    )

    result = df.copy()

    result["startDate"] = pd.to_datetime(
        result["startDate"],
        errors="coerce",
        utc=True,
    )

    result = result.sort_values(
        [
            "season",
            "team",
            "startDate",
            "gameId",
        ]
    ).copy()

    group_columns = [
        "season",
        "team",
    ]

    # Shift opponent strength so the current game cannot contribute
    # to its own SOS.
    result["priorOpponentWinPct"] = (
        result
        .groupby(group_columns)["opponentWinPctBefore"]
        .transform(lambda x: x.shift(1))
    )

    result["priorOpponentPointDiff"] = (
        result
        .groupby(group_columns)["opponentPointDiffBefore"]
        .transform(lambda x: x.shift(1))
    )

    result["priorSOSWinPct"] = (
        result
        .groupby(group_columns)["priorOpponentWinPct"]
        .transform(
            lambda x: x.expanding(
                min_periods=1
            ).mean()
        )
    )

    result["priorSOSPointDiff"] = (
        result
        .groupby(group_columns)["priorOpponentPointDiff"]
        .transform(
            lambda x: x.expanding(
                min_periods=1
            ).mean()
        )
    )

    # Remove intermediate columns.
    result = result.drop(
        columns=[
            "priorOpponentWinPct",
            "priorOpponentPointDiff",
            "opponentWinPctBefore",
            "opponentPointDiffBefore",
        ]
    )

    print(
        "Created:"
    )

    for feature in SOS_FEATURES:
        print(f"  {feature}")

    return result


# =============================================================================
# NEW FEATURE VALIDATION
# =============================================================================

def validate_new_features(df, year):
    """Validate the newly generated team-game features."""

    print_section(
        f"VALIDATING NEW FEATURES — {year}"
    )

    expected = list(
        TREND_FEATURES.keys()
    ) + SOS_FEATURES

    missing = [
        column
        for column in expected
        if column not in df.columns
    ]

    if missing:
        fail(
            f"{year}: New features missing: {missing}"
        )

    print(
        f"New features present: {len(expected)}"
    )

    for feature in expected:

        non_numeric = (
            not pd.api.types.is_numeric_dtype(
                df[feature]
            )
        )

        if non_numeric:
            fail(
                f"{year}: {feature} is not numeric."
            )

    print(
        "Feature types: VALID"
    )

    # First game for every team should not have prior SOS.
    first_games = (
        df.sort_values(
            [
                "season",
                "team",
                "startDate",
                "gameId",
            ]
        )
        .groupby(
            [
                "season",
                "team",
            ],
            as_index=False,
        )
        .head(1)
    )

    if first_games["priorSOSWinPct"].notna().any():
        fail(
            f"{year}: First team-game has non-missing priorSOSWinPct."
        )

    if first_games["priorSOSPointDiff"].notna().any():
        fail(
            f"{year}: First team-game has non-missing priorSOSPointDiff."
        )

    print(
        "First-game SOS validation: VALID"
    )

    # Trend features should be missing when the source values
    # do not yet exist.
    for new_name, (
        last3_column,
        season_column,
    ) in TREND_FEATURES.items():

        source_missing = (
            df[last3_column].isna()
            | df[season_column].isna()
        )

        invalid = (
            source_missing
            & df[new_name].notna()
        )

        if invalid.any():
            fail(
                f"{year}: {new_name} is non-missing when "
                f"its source values are unavailable."
            )

    print(
        "Trend missingness validation: VALID"
    )


# =============================================================================
# CONVERT TO GAME LEVEL
# =============================================================================

def create_game_level_new_features(df, year):
    """
    Convert team-game new features into one row per game.

    Home team values receive _home.
    Away team values receive _away.
    """

    print_section(
        f"CREATING GAME-LEVEL NEW FEATURES — {year}"
    )

    feature_columns = (
        list(TREND_FEATURES.keys())
        + SOS_FEATURES
    )

    home = (
        df[df["homeAway"] == "home"]
        [
            ["gameId"] + feature_columns
        ]
        .copy()
    )

    away = (
        df[df["homeAway"] == "away"]
        [
            ["gameId"] + feature_columns
        ]
        .copy()
    )

    home = home.rename(
        columns={
            feature: f"{feature}_home"
            for feature in feature_columns
        }
    )

    away = away.rename(
        columns={
            feature: f"{feature}_away"
            for feature in feature_columns
        }
    )

    game_features = home.merge(
        away,
        on="gameId",
        how="inner",
        validate="one_to_one",
    )

    expected_rows = df["gameId"].nunique()

    if len(game_features) != expected_rows:
        fail(
            f"{year}: Game-level feature count mismatch.\n"
            f"Expected: {expected_rows:,}\n"
            f"Actual:   {len(game_features):,}"
        )

    if game_features["gameId"].duplicated().any():
        fail(
            f"{year}: Duplicate gameIds in game-level new features."
        )

    print(
        f"Game-level rows: {len(game_features):,}"
    )

    print(
        "Game-level grain: VALID"
    )

    return game_features


# =============================================================================
# MERGE WITH FINAL FEATURES
# =============================================================================

def merge_with_final_features(
    final_features,
    new_game_features,
    year,
):
    """
    Merge new game-level features onto the existing final feature set.

    The source game-team data may contain more games than final_features.
    This is expected when the upstream pipeline filters the final modeling
    population (for example, to games involving an FBS team).

    IMPORTANT:
        New features are calculated using the complete chronological
        game-team history first. Only after feature construction do we
        restrict the new features to gameIds present in final_features.
    """

    print_section(
        f"MERGING WITH FINAL FEATURES — {year}"
    )

    if "gameId" not in final_features.columns:
        fail(
            f"{year}: final_features does not contain gameId."
        )

    if "gameId" not in new_game_features.columns:
        fail(
            f"{year}: New features do not contain gameId."
        )

    # -------------------------------------------------------------------------
    # Identify the modeling population.
    # -------------------------------------------------------------------------

    final_game_ids = set(
        final_features["gameId"]
    )

    new_game_ids = set(
        new_game_features["gameId"]
    )

    only_final = final_game_ids - new_game_ids
    only_new = new_game_ids - final_game_ids

    # Every final-feature game MUST have corresponding enhanced features.
    if only_final:
        fail(
            f"{year}: Games present in final_features but "
            f"missing from new features: "
            f"{len(only_final):,}"
        )

    # Additional games in game_team_stats are allowed.
    if only_new:
        print(
            f"Additional source games excluded from final modeling "
            f"population: {len(only_new):,}"
        )

    # -------------------------------------------------------------------------
    # Restrict enhanced features to the final modeling population.
    #
    # This happens AFTER all temporal features have been calculated.
    # -------------------------------------------------------------------------

    new_game_features = (
        new_game_features[
            new_game_features["gameId"].isin(
                final_game_ids
            )
        ]
        .copy()
    )

    if len(new_game_features) != len(final_features):
        fail(
            f"{year}: After restricting enhanced features to the "
            f"final modeling population, row counts do not match.\n"
            f"Final features:     {len(final_features):,}\n"
            f"Enhanced features:  {len(new_game_features):,}"
        )

    # -------------------------------------------------------------------------
    # Verify one row per game.
    # -------------------------------------------------------------------------

    if new_game_features["gameId"].duplicated().any():
        fail(
            f"{year}: Duplicate gameIds in filtered enhanced features."
        )

    # -------------------------------------------------------------------------
    # Identify new columns.
    # -------------------------------------------------------------------------

    new_columns = [
        column
        for column in new_game_features.columns
        if column != "gameId"
    ]

    collisions = [
        column
        for column in new_columns
        if column in final_features.columns
    ]

    if collisions:
        fail(
            f"{year}: New features would overwrite existing "
            f"columns: {collisions}"
        )

    # -------------------------------------------------------------------------
    # Merge.
    # -------------------------------------------------------------------------

    enhanced = final_features.merge(
        new_game_features,
        on="gameId",
        how="left",
        validate="one_to_one",
    )

    if len(enhanced) != len(final_features):
        fail(
            f"{year}: Merge changed final feature row count.\n"
            f"Before: {len(final_features):,}\n"
            f"After:  {len(enhanced):,}"
        )

    # -------------------------------------------------------------------------
    # Validate that every new feature is present for the final population.
    # -------------------------------------------------------------------------

    for column in new_columns:

        # We expect some early-season observations to legitimately be NaN.
        # Therefore we validate column existence rather than requiring
        # complete non-missingness.
        if column not in enhanced.columns:
            fail(
                f"{year}: New feature {column} disappeared during merge."
            )

    print(
        f"Baseline columns:  {len(final_features.columns):,}"
    )

    print(
        f"New columns:       {len(new_columns):,}"
    )

    print(
        f"Enhanced columns:  {len(enhanced.columns):,}"
    )

    print(
        "One-to-one merge: VALID"
    )

    return enhanced


# =============================================================================
# SAVE
# =============================================================================

def save_enhanced_features(df, year):
    """Save enhanced feature dataset."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / f"logistic_features_{year}.csv"
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved: {output_path}"
    )

    return output_path


# =============================================================================
# PROCESS ONE YEAR
# =============================================================================

def process_year(year):
    """Process one season."""

    print_section(
        f"PROCESSING SEASON {year}"
    )

    game_team, final_features = (
        load_year_data(year)
    )

    print(
        f"Loaded game-team data: "
        f"{game_team.shape[0]:,} rows × "
        f"{game_team.shape[1]:,} columns"
    )

    print(
        f"Loaded final features: "
        f"{final_features.shape[0]:,} rows × "
        f"{final_features.shape[1]:,} columns"
    )

    # -------------------------------------------------------------------------
    # Validate source datasets.
    # -------------------------------------------------------------------------

    validate_game_team_structure(
        game_team,
        year,
    )

    validate_final_features(
        final_features,
        year,
    )

    # -------------------------------------------------------------------------
    # Establish chronological team-game representation.
    # -------------------------------------------------------------------------

    team_games = validate_temporal_order(
        game_team,
        year,
    )

    # -------------------------------------------------------------------------
    # Create recent-form features.
    # -------------------------------------------------------------------------

    team_games = create_trend_features(
        team_games,
        year,
    )

    # -------------------------------------------------------------------------
    # Attach opponent's pregame strength.
    # -------------------------------------------------------------------------

    team_games = create_opponent_strength_features(
        team_games,
        year,
    )

    # -------------------------------------------------------------------------
    # Create prior SOS.
    # -------------------------------------------------------------------------

    team_games = create_prior_sos(
        team_games,
        year,
    )

    # -------------------------------------------------------------------------
    # Validate new features.
    # -------------------------------------------------------------------------

    validate_new_features(
        team_games,
        year,
    )

    # -------------------------------------------------------------------------
    # Return to game-level grain.
    # -------------------------------------------------------------------------

    new_game_features = (
        create_game_level_new_features(
            team_games,
            year,
        )
    )

    # -------------------------------------------------------------------------
    # Merge onto existing final features.
    # -------------------------------------------------------------------------

    enhanced = merge_with_final_features(
        final_features,
        new_game_features,
        year,
    )

    # -------------------------------------------------------------------------
    # Save.
    # -------------------------------------------------------------------------

    output_path = save_enhanced_features(
        enhanced,
        year,
    )

    return {
        "year": year,
        "baseline_rows": len(final_features),
        "enhanced_rows": len(enhanced),
        "baseline_columns": len(
            final_features.columns
        ),
        "enhanced_columns": len(
            enhanced.columns
        ),
        "output": output_path,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run enhanced feature generation for all seasons."""

    print()
    print("=" * 78)
    print("LOGISTIC REGRESSION ENHANCED FEATURE ENGINEERING")
    print("=" * 78)

    print()
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Game-team data: {GAME_TEAM_STATS_DIR}")
    print(f"Final features: {FINAL_FEATURES_DIR}")
    print(f"Output:         {OUTPUT_DIR}")
    print(f"Years:          {YEARS[0]}–{YEARS[-1]}")

    results = []

    for year in YEARS:

        try:

            summary = process_year(year)

            results.append(summary)

        except Exception as exc:

            print()
            print("=" * 78)
            print(f"FAILED: {year}")
            print("=" * 78)
            print(str(exc))
            print()

            raise

    # -------------------------------------------------------------------------
    # Final summary.
    # -------------------------------------------------------------------------

    print_section(
        "FINAL SUMMARY"
    )

    summary_df = pd.DataFrame(
        results
    )

    print(
        summary_df[
            [
                "year",
                "baseline_rows",
                "enhanced_rows",
                "baseline_columns",
                "enhanced_columns",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Enhanced feature generation completed successfully."
    )


if __name__ == "__main__":
    main()