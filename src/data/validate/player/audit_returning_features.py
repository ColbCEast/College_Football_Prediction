"""
Audit game-level returning-production feature layer.

Validates:
1. Original game integrity
2. Returning-production team matching
3. FBS/FCS coverage
4. Missingness indicators
5. Season-specific median imputation
6. Preservation of matched source values
7. Interaction correctness
8. gamesBefore behavior
9. Output structure and column counts

This script is diagnostic only.
It does NOT modify any files.
"""

from pathlib import Path

import numpy as np
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

RETURNING_FEATURES_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "win_probability"
    / "returning_production"
)

EXPECTED_SEASONS = list(range(2015, 2026))


# =============================================================================
# FEATURE DEFINITIONS
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

MISSINGNESS_COLUMNS = [
    "home_returning_missing",
    "away_returning_missing",
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
# HELPERS
# =============================================================================

def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def fail(message):
    raise AssertionError(message)


def assert_equal(actual, expected, message):
    if actual != expected:
        fail(f"{message} | expected={expected}, actual={actual}")


def assert_true(condition, message):
    if not condition:
        fail(message)


def assert_close(actual, expected, message, atol=1e-10):
    if not np.isclose(actual, expected, atol=atol, rtol=1e-10):
        fail(
            f"{message} | expected={expected}, actual={actual}"
        )


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data(year):
    final_path = FINAL_FEATURES_DIR / f"final_features_{year}.csv"
    returning_path = (
        RETURNING_PRODUCTION_DIR
        / f"returning_production_{year}.csv"
    )
    output_path = (
        RETURNING_FEATURES_DIR
        / f"returning_features_{year}.csv"
    )

    assert_true(
        final_path.exists(),
        f"Missing final feature file: {final_path}"
    )

    assert_true(
        returning_path.exists(),
        f"Missing returning-production file: {returning_path}"
    )

    assert_true(
        output_path.exists(),
        f"Missing returning feature file: {output_path}"
    )

    final_df = pd.read_csv(final_path)
    returning_df = pd.read_csv(returning_path)
    output_df = pd.read_csv(output_path)

    return final_df, returning_df, output_df


# =============================================================================
# BASIC STRUCTURE
# =============================================================================

def audit_basic_structure(year, final_df, returning_df, output_df):
    print_section(f"{year} — BASIC STRUCTURE")

    print(f"Original games:          {len(final_df):,}")
    print(f"Returning teams:         {len(returning_df):,}")
    print(f"Returning feature rows:  {len(output_df):,}")
    print(f"Original columns:        {len(final_df.columns):,}")
    print(f"Output columns:          {len(output_df.columns):,}")

    # Row count
    assert_equal(
        len(output_df),
        len(final_df),
        "Output row count changed"
    )

    # Expected column count
    expected_columns = len(final_df.columns) + 24 + 2 + 16

    assert_equal(
        len(output_df.columns),
        expected_columns,
        "Unexpected output column count"
    )

    # Game IDs
    assert_true(
        output_df["gameId"].is_unique,
        "Output contains duplicate gameId values"
    )

    assert_true(
        final_df["gameId"].is_unique,
        "Original final features contain duplicate gameId values"
    )

    # Exact game ID/order preservation
    assert_true(
        output_df["gameId"].equals(final_df["gameId"]),
        "Output gameId values/order do not exactly match original"
    )

    print("✓ Row count preserved")
    print("✓ Expected column count")
    print("✓ gameId uniqueness")
    print("✓ gameId values/order preserved")


# =============================================================================
# ORIGINAL FEATURE PRESERVATION
# =============================================================================

def audit_original_features(final_df, output_df):
    print_section("ORIGINAL FEATURE PRESERVATION")

    original_columns = list(final_df.columns)

    for column in original_columns:
        assert_true(
            column in output_df.columns,
            f"Original column missing from output: {column}"
        )

        original = final_df[column]
        output = output_df[column]

        if pd.api.types.is_numeric_dtype(original):
            equal = np.allclose(
                original.to_numpy(),
                output.to_numpy(),
                equal_nan=True,
            )
        else:
            equal = original.equals(output)

        assert_true(
            equal,
            f"Original feature changed: {column}"
        )

    print(
        f"✓ All {len(original_columns):,} original features preserved exactly"
    )


# =============================================================================
# RETURNING SOURCE VALIDATION
# =============================================================================

def audit_returning_source(year, returning_df):
    print_section(f"{year} — RETURNING SOURCE")

    missing_columns = [
        column
        for column in REQUIRED_RETURNING_COLUMNS
        if column not in returning_df.columns
    ]

    assert_true(
        not missing_columns,
        f"Returning source missing columns: {missing_columns}"
    )

    # Team-season uniqueness
    duplicate_count = returning_df.duplicated(
        subset=["season", "team"]
    ).sum()

    assert_equal(
        duplicate_count,
        0,
        "Returning source contains duplicate team-season records"
    )

    # No missing source values
    missing_values = returning_df[RETURNING_FEATURES].isna().sum().sum()

    assert_equal(
        missing_values,
        0,
        "Returning source contains missing feature values"
    )

    # Correct season
    assert_true(
        (returning_df["season"] == year).all(),
        "Returning source contains unexpected season values"
    )

    print(f"✓ {len(returning_df):,} unique team-season records")
    print("✓ No duplicate team-season records")
    print("✓ No missing returning-production source values")
    print("✓ Correct season")


# =============================================================================
# TEAM MATCHING
# =============================================================================

def audit_team_matching(year, final_df, returning_df, output_df):
    print_section(f"{year} — TEAM MATCHING")

    returning_teams = set(returning_df["team"].dropna().unique())

    home_teams = set(final_df["homeTeam"].dropna().unique())
    away_teams = set(final_df["awayTeam"].dropna().unique())

    game_teams = home_teams | away_teams

    unmatched_game_teams = sorted(
        game_teams - returning_teams
    )

    matched_game_teams = sorted(
        game_teams & returning_teams
    )

    print(f"Unique game teams:       {len(game_teams):,}")
    print(f"Returning teams:         {len(returning_teams):,}")
    print(f"Matched game teams:      {len(matched_game_teams):,}")
    print(f"Unmatched game teams:    {len(unmatched_game_teams):,}")

    # Team-level missingness based on actual source matching
    expected_home_missing = (
        ~final_df["homeTeam"].isin(returning_teams)
    ).astype(int)

    expected_away_missing = (
        ~final_df["awayTeam"].isin(returning_teams)
    ).astype(int)

    assert_true(
        output_df["home_returning_missing"].equals(
            expected_home_missing
        ),
        "home_returning_missing does not match source team coverage"
    )

    assert_true(
        output_df["away_returning_missing"].equals(
            expected_away_missing
        ),
        "away_returning_missing does not match source team coverage"
    )

    print("✓ Home team matching indicators correct")
    print("✓ Away team matching indicators correct")

    # Display unmatched teams for diagnostic purposes
    if unmatched_game_teams:
        print("\nUnmatched game teams:")
        for team in unmatched_game_teams:
            print(f"  - {team}")

    return returning_teams


# =============================================================================
# MATCHED VALUES AND IMPUTATION
# =============================================================================

def audit_feature_values_and_imputation(
    year,
    final_df,
    returning_df,
    output_df,
    returning_teams,
):
    print_section(f"{year} — VALUES & IMPUTATION")

    returning_lookup = returning_df.set_index("team")

    # Season-specific medians
    medians = returning_df[RETURNING_FEATURES].median()

    home_matched = final_df["homeTeam"].isin(returning_teams)
    away_matched = final_df["awayTeam"].isin(returning_teams)

    home_imputed_count = int((~home_matched).sum())
    away_imputed_count = int((~away_matched).sum())

    print(f"Home imputed rows:       {home_imputed_count:,}")
    print(f"Away imputed rows:       {away_imputed_count:,}")

    # -------------------------------------------------------------------------
    # Home side
    # -------------------------------------------------------------------------

    for feature in RETURNING_FEATURES:
        output_column = f"home_{feature}"

        # Matched rows must equal source values
        for team in final_df.loc[home_matched, "homeTeam"].unique():
            source_value = returning_lookup.loc[team, feature]

            mask = final_df["homeTeam"] == team

            actual_values = output_df.loc[mask, output_column]

            assert_true(
                np.allclose(
                    actual_values.to_numpy(),
                    source_value,
                    equal_nan=True,
                ),
                f"Matched home values incorrect for {feature}, team={team}"
            )

        # Unmatched rows must equal season median
        if home_imputed_count > 0:
            imputed_values = output_df.loc[
                ~home_matched,
                output_column
            ]

            assert_true(
                np.allclose(
                    imputed_values.to_numpy(),
                    medians[feature],
                    equal_nan=True,
                ),
                (
                    f"Home imputation incorrect for {feature} | "
                    f"expected median={medians[feature]}"
                ),
            )

    # -------------------------------------------------------------------------
    # Away side
    # -------------------------------------------------------------------------

    for feature in RETURNING_FEATURES:
        output_column = f"away_{feature}"

        # Matched rows must equal source values
        for team in final_df.loc[away_matched, "awayTeam"].unique():
            source_value = returning_lookup.loc[team, feature]

            mask = final_df["awayTeam"] == team

            actual_values = output_df.loc[mask, output_column]

            assert_true(
                np.allclose(
                    actual_values.to_numpy(),
                    source_value,
                    equal_nan=True,
                ),
                f"Matched away values incorrect for {feature}, team={team}"
            )

        # Unmatched rows must equal season median
        if away_imputed_count > 0:
            imputed_values = output_df.loc[
                ~away_matched,
                output_column
            ]

            assert_true(
                np.allclose(
                    imputed_values.to_numpy(),
                    medians[feature],
                    equal_nan=True,
                ),
                (
                    f"Away imputation incorrect for {feature} | "
                    f"expected median={medians[feature]}"
                ),
            )

    print("✓ All matched home values equal source values")
    print("✓ All matched away values equal source values")
    print("✓ All unmatched home values equal season-specific medians")
    print("✓ All unmatched away values equal season-specific medians")

    # No missing values after imputation
    engineered_columns = [
        f"home_{feature}"
        for feature in RETURNING_FEATURES
    ] + [
        f"away_{feature}"
        for feature in RETURNING_FEATURES
    ]

    missing_after_imputation = (
        output_df[engineered_columns]
        .isna()
        .sum()
        .sum()
    )

    assert_equal(
        missing_after_imputation,
        0,
        "Returning-production features contain missing values after imputation"
    )

    print("✓ No missing returning-production values remain")


# =============================================================================
# MISSINGNESS INDICATORS
# =============================================================================

def audit_missingness_indicators(output_df):
    print_section("MISSINGNESS INDICATORS")

    for column in MISSINGNESS_COLUMNS:
        assert_true(
            column in output_df.columns,
            f"Missing indicator not found: {column}"
        )

        unique_values = set(
            output_df[column].dropna().unique()
        )

        assert_true(
            unique_values.issubset({0, 1}),
            f"{column} contains values other than 0/1: {unique_values}"
        )

        missing_count = int(output_df[column].sum())

        print(
            f"{column}: {missing_count:,} rows flagged "
            f"({missing_count / len(output_df):.2%})"
        )

    print("✓ Missingness indicators contain only 0/1")


# =============================================================================
# GAMES BEFORE
# =============================================================================

def audit_games_before(output_df):
    print_section("GAMES-BEFORE VARIABLES")

    for column in ["gamesBefore_home", "gamesBefore_away"]:
        assert_true(
            column in output_df.columns,
            f"Missing required gamesBefore column: {column}"
        )

        assert_true(
            output_df[column].notna().all(),
            f"{column} contains missing values"
        )

        # gamesBefore cannot be negative
        assert_true(
            (output_df[column] >= 0).all(),
            f"{column} contains negative values"
        )

        # gamesBefore should be integer-valued
        assert_true(
            np.allclose(
                output_df[column],
                np.round(output_df[column])
            ),
            f"{column} contains non-integer values"
        )

        print(
            f"{column}: "
            f"min={output_df[column].min():.0f}, "
            f"max={output_df[column].max():.0f}, "
            f"median={output_df[column].median():.1f}, "
            f"zero={int((output_df[column] == 0).sum()):,}"
        )

    # At least some games should occur before a team has played any games
    assert_true(
        (output_df["gamesBefore_home"] == 0).any(),
        "No home games have gamesBefore_home = 0"
    )

    assert_true(
        (output_df["gamesBefore_away"] == 0).any(),
        "No away games have gamesBefore_away = 0"
    )

    print("✓ Both home and away sides contain preseason games")
    print("✓ gamesBefore values are non-negative")
    print("✓ gamesBefore values are integer-valued")


# =============================================================================
# INTERACTION VALIDATION
# =============================================================================

def audit_interactions(output_df):
    print_section("INTERACTION FEATURES")

    interaction_count = 0

    for feature in INTERACTION_FEATURES:
        home_source = f"home_{feature}"
        away_source = f"away_{feature}"

        home_interaction = (
            f"home_{feature}_x_gamesBefore_home"
        )

        away_interaction = (
            f"away_{feature}_x_gamesBefore_away"
        )

        required_columns = [
            home_source,
            away_source,
            home_interaction,
            away_interaction,
        ]

        for column in required_columns:
            assert_true(
                column in output_df.columns,
                f"Missing interaction-related column: {column}"
            )

        expected_home = (
            output_df[home_source]
            * output_df["gamesBefore_home"]
        )

        expected_away = (
            output_df[away_source]
            * output_df["gamesBefore_away"]
        )

        assert_true(
            np.allclose(
                output_df[home_interaction].to_numpy(),
                expected_home.to_numpy(),
                equal_nan=True,
            ),
            f"Incorrect home interaction: {home_interaction}"
        )

        assert_true(
            np.allclose(
                output_df[away_interaction].to_numpy(),
                expected_away.to_numpy(),
                equal_nan=True,
            ),
            f"Incorrect away interaction: {away_interaction}"
        )

        interaction_count += 2

    print(f"✓ All {interaction_count} interaction features calculate correctly")

    # -------------------------------------------------------------------------
    # Explicit zero check for preseason
    # -------------------------------------------------------------------------

    home_zero_games = output_df["gamesBefore_home"] == 0
    away_zero_games = output_df["gamesBefore_away"] == 0

    for feature in INTERACTION_FEATURES:
        home_interaction = (
            f"home_{feature}_x_gamesBefore_home"
        )
        away_interaction = (
            f"away_{feature}_x_gamesBefore_away"
        )

        if home_zero_games.any():
            assert_true(
                (
                    output_df.loc[
                        home_zero_games,
                        home_interaction
                    ] == 0
                ).all(),
                f"{home_interaction} is nonzero when gamesBefore_home=0"
            )

        if away_zero_games.any():
            assert_true(
                (
                    output_df.loc[
                        away_zero_games,
                        away_interaction
                    ] == 0
                ).all(),
                f"{away_interaction} is nonzero when gamesBefore_away=0"
            )

    print("✓ All interactions equal zero when gamesBefore = 0")


# =============================================================================
# COLUMN STRUCTURE
# =============================================================================

def audit_column_structure(final_df, output_df):
    print_section("COLUMN STRUCTURE")

    expected_returning_columns = (
        [
            f"home_{feature}"
            for feature in RETURNING_FEATURES
        ]
        + [
            f"away_{feature}"
            for feature in RETURNING_FEATURES
        ]
        + MISSINGNESS_COLUMNS
        + [
            f"home_{feature}_x_gamesBefore_home"
            for feature in INTERACTION_FEATURES
        ]
        + [
            f"away_{feature}_x_gamesBefore_away"
            for feature in INTERACTION_FEATURES
        ]
    )

    for column in expected_returning_columns:
        assert_true(
            column in output_df.columns,
            f"Expected engineered column missing: {column}"
        )

    # Confirm no unexpected engineered columns
    original_columns = set(final_df.columns)
    engineered_columns = [
        column
        for column in output_df.columns
        if column not in original_columns
    ]

    expected_engineered_set = set(expected_returning_columns)

    unexpected_columns = sorted(
        set(engineered_columns) - expected_engineered_set
    )

    assert_true(
        not unexpected_columns,
        f"Unexpected engineered columns found: {unexpected_columns}"
    )

    assert_equal(
        len(engineered_columns),
        42,
        "Unexpected number of engineered columns"
    )

    print("✓ All expected engineered columns present")
    print("✓ No unexpected engineered columns")
    print("✓ Exactly 42 engineered columns added")


# =============================================================================
# SEASON SUMMARY
# =============================================================================

def print_season_summary(
    year,
    final_df,
    returning_df,
    output_df,
):
    home_imputed = int(
        output_df["home_returning_missing"].sum()
    )

    away_imputed = int(
        output_df["away_returning_missing"].sum()
    )

    unique_game_teams = set(
        final_df["homeTeam"].dropna()
    ) | set(
        final_df["awayTeam"].dropna()
    )

    returning_teams = set(
        returning_df["team"].dropna()
    )

    matched_teams = unique_game_teams & returning_teams
    unmatched_teams = unique_game_teams - returning_teams

    print(
        f"{year}: "
        f"{len(final_df):,} games | "
        f"{len(returning_df):,} returning teams | "
        f"{len(matched_teams):,} matched game teams | "
        f"{len(unmatched_teams):,} unmatched game teams | "
        f"home imputed={home_imputed:,} | "
        f"away imputed={away_imputed:,}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("AUDIT RETURNING-PRODUCTION FEATURES")
    print("=" * 80)

    print(f"Project root:        {PROJECT_ROOT}")
    print(f"Final features:      {FINAL_FEATURES_DIR}")
    print(f"Returning source:    {RETURNING_PRODUCTION_DIR}")
    print(f"Returning features:  {RETURNING_FEATURES_DIR}")

    season_summaries = []

    for year in EXPECTED_SEASONS:
        final_df, returning_df, output_df = load_data(year)

        audit_basic_structure(
            year,
            final_df,
            returning_df,
            output_df,
        )

        audit_original_features(
            final_df,
            output_df,
        )

        audit_returning_source(
            year,
            returning_df,
        )

        returning_teams = audit_team_matching(
            year,
            final_df,
            returning_df,
            output_df,
        )

        audit_feature_values_and_imputation(
            year,
            final_df,
            returning_df,
            output_df,
            returning_teams,
        )

        audit_missingness_indicators(
            output_df,
        )

        audit_games_before(
            output_df,
        )

        audit_interactions(
            output_df,
        )

        audit_column_structure(
            final_df,
            output_df,
        )

        print_season_summary(
            year,
            final_df,
            returning_df,
            output_df,
        )

        season_summaries.append(
            {
                "season": year,
                "games": len(final_df),
                "returning_teams": len(returning_df),
                "output_columns": len(output_df.columns),
                "home_imputed": int(
                    output_df["home_returning_missing"].sum()
                ),
                "away_imputed": int(
                    output_df["away_returning_missing"].sum()
                ),
            }
        )

        print(f"\n✓ {year} audit passed")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print_section("FINAL AUDIT SUMMARY")

    summary_df = pd.DataFrame(season_summaries)

    print(summary_df.to_string(index=False))

    assert_equal(
        len(summary_df),
        len(EXPECTED_SEASONS),
        "Unexpected number of seasons in summary"
    )

    assert_true(
        (summary_df["output_columns"] == 490).all(),
        "Not all output files contain 490 columns"
    )

    print("\n✓ All seasons passed")
    print(f"✓ Seasons audited: {len(summary_df)}")
    print("✓ Original game counts preserved")
    print("✓ Original feature values preserved")
    print("✓ Returning-production matches validated")
    print("✓ Missingness indicators validated")
    print("✓ Season-specific median imputation validated")
    print("✓ No missing engineered values remain")
    print("✓ gamesBefore variables validated")
    print("✓ All returning × gamesBefore interactions validated")
    print("✓ Output column structure validated")

    print("\n" + "=" * 80)
    print("AUDIT COMPLETE — ALL CHECKS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()