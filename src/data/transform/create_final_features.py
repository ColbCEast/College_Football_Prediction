import pandas as pd


# ============================================================
# Configuration
# ============================================================

START_YEAR = 2015
END_YEAR = 2025

GAME_FEATURE_PATH = (
    "data/processed/game_features/game_features_{year}.csv"
)

ADVANCED_FEATURE_PATH = (
    "data/processed/features/features_{year}.csv"
)

OUTPUT_PATH = (
    "data/processed/final_features/final_features_{year}.csv"
)


# ============================================================
# Known CFBD data limitations
# ============================================================

# 2016 contains one advanced-stat record that has no
# corresponding game-level feature record.
#
# This game was investigated separately and confirmed to be
# a CFBD source-data limitation.
KNOWN_ADVANCED_ONLY_IDS = {
    2016: {400868914}
}


# ============================================================
# Load data
# ============================================================

def load_feature_data(year):

    game_features = pd.read_csv(
        GAME_FEATURE_PATH.format(year=year)
    )

    advanced_features = pd.read_csv(
        ADVANCED_FEATURE_PATH.format(year=year)
    )

    return game_features, advanced_features


# ============================================================
# Validate unique game IDs
# ============================================================

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

        raise ValueError(
            f"{dataset_name} contains duplicate "
            f"{id_column} values."
        )


# ============================================================
# Validate game ID relationship
# ============================================================

def validate_game_id_match(
    year,
    game_features,
    advanced_features
):

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

    known_advanced_only = (
        KNOWN_ADVANCED_ONLY_IDS.get(
            year,
            set()
        )
    )

    unexplained_advanced_only = (
        only_advanced_features
        - known_advanced_only
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

    # --------------------------------------------------------
    # Game-level records without advanced features
    #
    # Beginning in 2022, this is expected because CFBD
    # includes additional non-FBS games.
    # --------------------------------------------------------

    if only_game_features:

        if year >= 2022:

            print(
                "\n  EXPECTED: Game-level dataset "
                "contains additional non-FBS games."
            )

        else:

            print(
                "\n  UNEXPECTED: Game-level games "
                "are missing advanced features."
            )

            print(
                sorted(only_game_features)
            )

            raise ValueError(
                f"{year}: Unexpected game IDs only "
                f"in game-level features."
            )

    # --------------------------------------------------------
    # Advanced records without game-level features
    # --------------------------------------------------------

    if known_advanced_only:

        print(
            "\n  EXPECTED: Known CFBD source-data "
            "limitation."
        )

        print(
            "  Known advanced-only IDs:",
            sorted(known_advanced_only)
        )

    if unexplained_advanced_only:

        print(
            "\n  ERROR: Unexplained advanced-only "
            "game IDs:"
        )

        print(
            sorted(unexplained_advanced_only)
        )

        raise ValueError(
            f"{year}: Advanced features contain "
            f"unexpected game IDs."
        )

    print(
        "\n  PASS: Game ID relationship is "
        "consistent with expected CFBD coverage."
    )


# ============================================================
# Validate game information
# ============================================================

def validate_game_information(
    game_features,
    advanced_features
):

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

    advanced_info = advanced_info.rename(
        columns={
            "id": "gameId"
        }
    )

    comparison = game_info.merge(
        advanced_info,
        on="gameId",
        how="inner",
        suffixes=(
            "_game",
            "_advanced"
        ),
        validate="one_to_one"
    )

    home_mismatches = comparison[
        comparison["homeTeam_game"]
        != comparison["homeTeam_advanced"]
    ]

    away_mismatches = comparison[
        comparison["awayTeam_game"]
        != comparison["awayTeam_advanced"]
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

        print(home_mismatches)

        raise ValueError(
            "Home team mismatch detected."
        )

    if len(away_mismatches) > 0:

        print(
            "\n  Away team mismatches:"
        )

        print(away_mismatches)

        raise ValueError(
            "Away team mismatch detected."
        )

    print(
        "  PASS: Game information matches."
    )


# ============================================================
# Remove known advanced-only records
# ============================================================

def remove_known_exceptions(
    year,
    advanced_features
):

    known_ids = KNOWN_ADVANCED_ONLY_IDS.get(
        year,
        set()
    )

    if not known_ids:
        return advanced_features

    original_rows = len(
        advanced_features
    )

    advanced_features = (
        advanced_features[
            ~advanced_features["id"].isin(
                known_ids
            )
        ]
        .copy()
    )

    removed_rows = (
        original_rows
        - len(advanced_features)
    )

    print(
        "\nRemoved known CFBD exceptions:"
    )

    print(
        f"  Rows removed: {removed_rows}"
    )

    print(
        f"  Game IDs: {sorted(known_ids)}"
    )

    return advanced_features


# ============================================================
# Validate final merge
# ============================================================

def create_final_features(
    year,
    game_features,
    advanced_features
):

    # --------------------------------------------------------
    # Remove known CFBD exception
    # --------------------------------------------------------

    advanced_features = (
        remove_known_exceptions(
            year,
            advanced_features
        )
    )

    # --------------------------------------------------------
    # Rename advanced ID to gameId
    # --------------------------------------------------------

    advanced_features = (
        advanced_features.rename(
            columns={
                "id": "gameId"
            }
        )
    )

    # --------------------------------------------------------
    # Identify overlapping columns
    # --------------------------------------------------------

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

    overlapping_columns.discard(
        "gameId"
    )

    print(
        "\nOverlapping columns:"
    )

    print(
        f"  {len(overlapping_columns)}"
        " columns"
    )

    if overlapping_columns:

        for column in sorted(
            overlapping_columns
        ):

            print(
                f"  {column}"
            )

    # --------------------------------------------------------
    # Remove duplicated metadata from advanced features
    #
    # Game-level features are our authoritative source for
    # these columns.
    # --------------------------------------------------------

    advanced_columns_to_drop = (
        sorted(overlapping_columns)
    )

    advanced_features = (
        advanced_features.drop(
            columns=advanced_columns_to_drop
        )
    )

    # --------------------------------------------------------
    # Perform final one-to-one merge
    # --------------------------------------------------------

    final_features = game_features.merge(
        advanced_features,
        on="gameId",
        how="inner",
        validate="one_to_one"
    )

    # --------------------------------------------------------
    # Validate merge row count
    #
    # IMPORTANT:
    #
    # The final dataset should contain only games for which
    # advanced features exist.
    #
    # Therefore, beginning in 2022, this will be smaller than
    # game_features because non-FBS games are intentionally
    # excluded.
    # --------------------------------------------------------

    expected_rows = len(
        advanced_features
    )

    actual_rows = len(
        final_features
    )

    print("\nFinal merge:")

    print(
        f"  Game-level rows: "
        f"{len(game_features)}"
    )

    print(
        f"  Advanced feature rows: "
        f"{len(advanced_features)}"
    )

    print(
        f"  Final feature rows: "
        f"{actual_rows}"
    )

    print(
        f"  Expected final rows: "
        f"{expected_rows}"
    )

    if actual_rows != expected_rows:

        raise ValueError(
            "Final merge did not preserve "
            "all advanced feature rows."
        )

    print(
        "  PASS: Final merge preserved "
        "all advanced feature rows."
    )

    return final_features


# ============================================================
# Validate final dataset
# ============================================================

def validate_final_dataset(
    final_features
):

    print("\nFinal dataset validation:")

    # --------------------------------------------------------
    # Unique game IDs
    # --------------------------------------------------------

    duplicate_games = (
        final_features["gameId"]
        .value_counts()
        .loc[lambda x: x > 1]
    )

    print(
        f"  Rows: {len(final_features)}"
    )

    print(
        f"  Unique games: "
        f"{final_features['gameId'].nunique()}"
    )

    print(
        f"  Duplicate game IDs: "
        f"{len(duplicate_games)}"
    )

    if len(duplicate_games) > 0:

        raise ValueError(
            "Final dataset contains duplicate "
            "game IDs."
        )

    # --------------------------------------------------------
    # Missing game IDs
    # --------------------------------------------------------

    missing_game_ids = (
        final_features["gameId"]
        .isna()
        .sum()
    )

    print(
        f"  Missing game IDs: "
        f"{missing_game_ids}"
    )

    if missing_game_ids > 0:

        raise ValueError(
            "Final dataset contains missing "
            "game IDs."
        )

    print(
        "  PASS: Final dataset is valid."
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    all_years_passed = True

    for year in range(
        START_YEAR,
        END_YEAR + 1
    ):

        print("\n" + "=" * 70)

        print(
            f"CREATING FINAL FEATURES: {year}"
        )

        print("=" * 70)

        try:

            # ------------------------------------------------
            # Load datasets
            # ------------------------------------------------

            game_features, advanced_features = (
                load_feature_data(year)
            )

            print("\nLoaded data:")

            print(
                f"  Game-level features: "
                f"{game_features.shape}"
            )

            print(
                f"  Advanced features: "
                f"{advanced_features.shape}"
            )

            # ------------------------------------------------
            # Validate unique IDs
            # ------------------------------------------------

            validate_unique_game_ids(
                game_features,
                "Game-level features",
                "gameId"
            )

            validate_unique_game_ids(
                advanced_features,
                "Advanced features",
                "id"
            )

            # ------------------------------------------------
            # Validate expected CFBD relationship
            # ------------------------------------------------

            validate_game_id_match(
                year,
                game_features,
                advanced_features
            )

            # ------------------------------------------------
            # Validate teams
            # ------------------------------------------------

            validate_game_information(
                game_features,
                advanced_features
            )

            # ------------------------------------------------
            # Create final dataset
            # ------------------------------------------------

            final_features = (
                create_final_features(
                    year,
                    game_features,
                    advanced_features
                )
            )

            # ------------------------------------------------
            # Validate final dataset
            # ------------------------------------------------

            validate_final_dataset(
                final_features
            )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            output_path = (
                OUTPUT_PATH.format(
                    year=year
                )
            )

            final_features.to_csv(
                output_path,
                index=False
            )

            print(
                "\nSaved final features:"
            )

            print(
                f"  {output_path}"
            )

            print(
                f"  Shape: "
                f"{final_features.shape}"
            )

        except Exception as e:

            all_years_passed = False

            print(
                "\n" + "!" * 70
            )

            print(
                f"FAILED: {year}"
            )

            print(
                f"ERROR: {e}"
            )

            print(
                "!" * 70
            )

    # ========================================================
    # Final result
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    if all_years_passed:

        print(
            "PASS: Final feature datasets "
            "created successfully for all seasons."
        )

    else:

        print(
            "FAIL: One or more seasons "
            "could not be processed."
        )