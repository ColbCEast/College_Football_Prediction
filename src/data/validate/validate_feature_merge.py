import pandas as pd


# ==============================================================
# KNOWN CFBD DATA COVERAGE EXCEPTIONS
# ==============================================================

# 2016:
# CFBD contains advanced statistics for this game, but the
# corresponding game-level feature dataset does not contain it.
#
# We previously investigated this game and confirmed that the
# CFBD game-level statistics are unavailable even when the game
# is queried individually.
KNOWN_ADVANCED_ONLY_IDS = {
    2016: {
        400868914
    }
}


# Beginning in 2022, the CFBD games endpoint began including
# additional non-FBS games (FCS-FCS, DII, DIII, etc.).
#
# Our game-level feature dataset therefore contains more games
# than the advanced-statistics dataset.
#
# Advanced features are intentionally restricted to games for
# which the appropriate CFBD advanced statistics are available.
EXPECTED_GAME_ONLY_SEASONS = {
    2022,
    2023,
    2024,
    2025
}


# ==============================================================
# LOAD DATA
# ==============================================================

def load_feature_data(year):

    game_features = pd.read_csv(
        f"data/processed/game_features/game_features_{year}.csv"
    )

    advanced_features = pd.read_csv(
        f"data/processed/features/features_{year}.csv"
    )

    return game_features, advanced_features


# ==============================================================
# VALIDATE UNIQUE GAME IDS
# ==============================================================

def validate_unique_game_ids(
    df,
    dataset_name,
    id_column
):
    duplicate_games = (
        df[id_column]
        .value_counts()
        .loc[lambda x: x > 1]
    )

    print(f"\n{dataset_name}:")
    print(f"  Rows: {len(df)}")
    print(
        f"  Unique games: "
        f"{df[id_column].nunique()}"
    )
    print(
        f"  Duplicate game IDs: "
        f"{len(duplicate_games)}"
    )

    if len(duplicate_games) > 0:

        print("  Duplicate game IDs:")
        print(duplicate_games)

    return len(duplicate_games) == 0


# ==============================================================
# VALIDATE GAME ID COVERAGE
# ==============================================================

def validate_game_id_match(
    game_features,
    advanced_features,
    year
):
    """
    Compare game IDs between the two datasets.

    Expected relationship:

        advanced_features ⊆ game_features

    except for explicitly documented CFBD anomalies.

    Beginning in 2022, game-level features may contain
    additional non-FBS games that are not expected to have
    advanced statistics.
    """

    game_ids = set(
        game_features["gameId"]
    )

    advanced_ids = set(
        advanced_features["id"]
    )

    only_game_features = (
        game_ids - advanced_ids
    )

    only_advanced_features = (
        advanced_ids - game_ids
    )

    matching_ids = (
        game_ids & advanced_ids
    )

    print("\nGame ID matching:")
    print(
        f"  Matching game IDs: "
        f"{len(matching_ids)}"
    )
    print(
        f"  Only in game features: "
        f"{len(only_game_features)}"
    )
    print(
        f"  Only in advanced features: "
        f"{len(only_advanced_features)}"
    )

    # ----------------------------------------------------------
    # Game IDs only in game-level features
    # ----------------------------------------------------------

    if only_game_features:

        print(
            "\n  Game IDs only in game features:"
        )

        print(
            f"  Count: "
            f"{len(only_game_features)}"
        )

        if year in EXPECTED_GAME_ONLY_SEASONS:

            print(
                "  EXPECTED: Beginning in 2022, "
                "CFBD includes additional non-FBS games."
            )

        else:

            print(
                "  UNEXPECTED: These game IDs require "
                "investigation."
            )

            print(
                sorted(only_game_features)
            )

    # ----------------------------------------------------------
    # Advanced IDs only in advanced features
    # ----------------------------------------------------------

    if only_advanced_features:

        print(
            "\n  Game IDs only in advanced features:"
        )

        print(
            f"  Count: "
            f"{len(only_advanced_features)}"
        )

        expected_exceptions = (
            KNOWN_ADVANCED_ONLY_IDS.get(
                year,
                set()
            )
        )

        expected_advanced_only = (
            only_advanced_features
            & expected_exceptions
        )

        unexpected_advanced_only = (
            only_advanced_features
            - expected_exceptions
        )

        if expected_advanced_only:

            print(
                "  EXPECTED: Known CFBD source-data "
                "limitation."
            )

            print(
                f"  Known IDs: "
                f"{sorted(expected_advanced_only)}"
            )

        if unexpected_advanced_only:

            print(
                "  UNEXPECTED: These advanced feature "
                "game IDs require investigation."
            )

            print(
                sorted(unexpected_advanced_only)
            )

    # ----------------------------------------------------------
    # Determine whether coverage is valid
    # ----------------------------------------------------------

    expected_exceptions = (
        KNOWN_ADVANCED_ONLY_IDS.get(
            year,
            set()
        )
    )

    unexpected_advanced_only = (
        only_advanced_features
        - expected_exceptions
    )

    unexpected_game_only = set()

    if year not in EXPECTED_GAME_ONLY_SEASONS:

        unexpected_game_only = (
            only_game_features
        )

    coverage_valid = (
        len(unexpected_advanced_only) == 0
        and len(unexpected_game_only) == 0
    )

    if coverage_valid:

        print(
            "\n  PASS: Game ID relationship is "
            "consistent with expected CFBD coverage."
        )

    else:

        print(
            "\n  FAIL: Unexpected game ID "
            "coverage differences detected."
        )

    return (
        only_game_features,
        only_advanced_features,
        coverage_valid
    )


# ==============================================================
# VALIDATE GAME INFORMATION
# ==============================================================

def validate_game_information(
    game_features,
    advanced_features
):
    """
    Compare basic game-level information between
    datasets to make sure matching game IDs refer
    to the same teams.
    """

    game_info = game_features[
        [
            "gameId",
            "homeTeam",
            "awayTeam"
        ]
    ].copy()

    advanced_info = advanced_features[
        [
            "id",
            "homeTeam",
            "awayTeam"
        ]
    ].copy()

    # Rename advanced ID temporarily so both
    # datasets use the same name for comparison.

    advanced_info = advanced_info.rename(
        columns={
            "id": "gameId"
        }
    )

    comparison = game_info.merge(
        advanced_info,
        on="gameId",
        how="outer",
        suffixes=(
            "_game",
            "_advanced"
        ),
        indicator=True
    )

    # Only compare games that exist in both datasets.

    matched = comparison[
        comparison["_merge"] == "both"
    ].copy()

    home_mismatches = matched[
        matched["homeTeam_game"]
        != matched["homeTeam_advanced"]
    ]

    away_mismatches = matched[
        matched["awayTeam_game"]
        != matched["awayTeam_advanced"]
    ]

    print("\nGame information validation:")

    print(
        f"  Home team mismatches: "
        f"{len(home_mismatches)}"
    )

    print(
        f"  Away team mismatches: "
        f"{len(away_mismatches)}"
    )

    if len(home_mismatches) > 0:

        print(
            "\n  Home team mismatches:"
        )

        print(
            home_mismatches
        )

    if len(away_mismatches) > 0:

        print(
            "\n  Away team mismatches:"
        )

        print(
            away_mismatches
        )

    return (
        len(home_mismatches) == 0
        and len(away_mismatches) == 0
    )


# ==============================================================
# VALIDATE ACTUAL MERGE
# ==============================================================

def validate_merge(
    game_features,
    advanced_features,
    year
):
    """
    Validate the actual one-to-one merge.

    The expected relationship is:

        advanced_features ⊆ game_features

    Therefore, every advanced feature row should match
    exactly one game-level feature row.

    We do NOT require every game-level feature row to have
    advanced statistics because:

    1. Beginning in 2022, CFBD includes additional non-FBS
       games in the game-level data.

    2. Some games may legitimately lack advanced statistics.
    """

    merged = game_features.merge(
        advanced_features,
        left_on="gameId",
        right_on="id",
        how="inner",
        suffixes=(
            "_game",
            "_advanced"
        ),
        validate="one_to_one"
    )

    print("\nMerge validation:")

    print(
        f"  Game feature rows: "
        f"{len(game_features)}"
    )

    print(
        f"  Advanced feature rows: "
        f"{len(advanced_features)}"
    )

    print(
        f"  Merged rows: "
        f"{len(merged)}"
    )

    # ----------------------------------------------------------
    # Determine unmatched advanced games
    # ----------------------------------------------------------

    game_ids = set(
        game_features["gameId"]
    )

    advanced_ids = set(
        advanced_features["id"]
    )

    unmatched_advanced = (
        advanced_ids - game_ids
    )

    expected_exceptions = (
        KNOWN_ADVANCED_ONLY_IDS.get(
            year,
            set()
        )
    )

    unexpected_advanced = (
        unmatched_advanced
        - expected_exceptions
    )

    # ----------------------------------------------------------
    # Validate
    # ----------------------------------------------------------

    if len(unexpected_advanced) == 0:

        print(
            "  PASS: All advanced feature rows "
            "matched game-level features, "
            "excluding documented CFBD exceptions."
        )

        if expected_exceptions:

            matched_exceptions = (
                unmatched_advanced
                & expected_exceptions
            )

            if matched_exceptions:

                print(
                    "  Known unmatched advanced "
                    "game IDs:"
                )

                print(
                    f"  {sorted(matched_exceptions)}"
                )

        return merged, True

    # ----------------------------------------------------------
    # Unexpected failure
    # ----------------------------------------------------------

    print(
        "  FAIL: Some advanced feature rows "
        "did not match game-level features."
    )

    print(
        f"  Unexpected unmatched advanced games: "
        f"{len(unexpected_advanced)}"
    )

    print(
        f"  Game IDs: "
        f"{sorted(unexpected_advanced)}"
    )

    return merged, False


# ==============================================================
# VALIDATE OVERLAPPING COLUMNS
# ==============================================================

def validate_columns(
    game_features,
    advanced_features
):
    """
    Check for columns that exist in both datasets
    besides the game ID columns.

    These may need to be handled when creating
    the final dataset.
    """

    game_columns = set(
        game_features.columns
    )

    advanced_columns = set(
        advanced_features.columns
    )

    overlapping_columns = (
        game_columns
        & advanced_columns
    )

    # These identifiers are intentionally different.

    overlapping_columns.discard(
        "gameId"
    )

    overlapping_columns.discard(
        "id"
    )

    print("\nOverlapping columns:")

    print(
        f"  {len(overlapping_columns)} "
        f"columns exist in both datasets."
    )

    if overlapping_columns:

        for column in sorted(
            overlapping_columns
        ):

            print(
                f"  {column}"
            )


# ==============================================================
# MAIN VALIDATION
# ==============================================================

if __name__ == "__main__":

    all_passed = True

    for year in range(
        2015,
        2026
    ):

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"VALIDATING FEATURE MERGE: {year}"
        )

        print(
            "=" * 70
        )

        # ------------------------------------------------------
        # Load data
        # ------------------------------------------------------

        game_features, advanced_features = (
            load_feature_data(year)
        )

        # ------------------------------------------------------
        # 1. Validate unique game IDs
        # ------------------------------------------------------

        game_ids_valid = (
            validate_unique_game_ids(
                game_features,
                "Game-level features",
                "gameId"
            )
        )

        advanced_ids_valid = (
            validate_unique_game_ids(
                advanced_features,
                "Advanced features",
                "id"
            )
        )

        if (
            not game_ids_valid
            or not advanced_ids_valid
        ):

            all_passed = False

        # ------------------------------------------------------
        # 2. Validate matching game IDs
        # ------------------------------------------------------

        (
            only_game,
            only_advanced,
            game_id_match_valid
        ) = validate_game_id_match(
            game_features,
            advanced_features,
            year
        )

        if not game_id_match_valid:

            all_passed = False

        # ------------------------------------------------------
        # 3. Validate game information
        # ------------------------------------------------------

        game_information_valid = (
            validate_game_information(
                game_features,
                advanced_features
            )
        )

        if not game_information_valid:

            all_passed = False

        # ------------------------------------------------------
        # 4. Validate actual merge
        # ------------------------------------------------------

        try:

            (
                merged,
                merge_valid
            ) = validate_merge(
                game_features,
                advanced_features,
                year
            )

            if not merge_valid:

                all_passed = False

        except Exception as e:

            print(
                "\n  MERGE FAILED:"
            )

            print(
                f"  {e}"
            )

            all_passed = False

        # ------------------------------------------------------
        # 5. Check overlapping columns
        # ------------------------------------------------------

        validate_columns(
            game_features,
            advanced_features
        )

    # ==========================================================
    # FINAL RESULT
    # ==========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL VALIDATION RESULT"
    )

    print(
        "=" * 70
    )

    if all_passed:

        print(
            "PASS: All feature datasets "
            "passed merge validation."
        )

        print(
            "\nKnown CFBD coverage differences:"
        )

        print(
            "  2016:"
        )

        print(
            "    - 1 advanced-stat game exists "
            "without game-level feature data."
        )

        print(
            "    - Game ID: 400868914"
        )

        print(
            "    - Confirmed CFBD source-data limitation."
        )

        print(
            "\n  2022-2025:"
        )

        print(
            "    - Game-level data contains additional "
            "non-FBS games."
        )

        print(
            "    - These games are intentionally excluded "
            "from advanced features."
        )

        print(
            "\nNo unexplained game ID mismatches found."
        )

        print(
            "No duplicate game IDs found."
        )

        print(
            "No home-team mismatches found."
        )

        print(
            "No away-team mismatches found."
        )

    else:

        print(
            "FAIL: One or more seasons "
            "require investigation."
        )