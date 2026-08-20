import pandas as pd


def load_feature_data(year):
    game_features = pd.read_csv(
        f"data/processed/game_features/game_features_{year}.csv"
    )

    advanced_features = pd.read_csv(
        f"data/processed/features/features_{year}.csv"
    )

    return game_features, advanced_features


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


def validate_game_id_match(
    game_features,
    advanced_features
):
    game_ids = set(game_features["gameId"])
    advanced_ids = set(advanced_features["id"])

    only_game_features = game_ids - advanced_ids
    only_advanced_features = advanced_ids - game_ids
    matching_ids = game_ids & advanced_ids

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

    if only_game_features:
        print(
            "\n  Game IDs only in game features:"
        )
        print(sorted(only_game_features))

    if only_advanced_features:
        print(
            "\n  Game IDs only in advanced features:"
        )
        print(sorted(only_advanced_features))

    return (
        only_game_features,
        only_advanced_features
    )


def validate_game_information(
    game_features,
    advanced_features
):
    """
    Compare basic game-level information between
    datasets to make sure the same games refer
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
        columns={"id": "gameId"}
    )

    comparison = game_info.merge(
        advanced_info,
        on="gameId",
        how="outer",
        suffixes=("_game", "_advanced"),
        indicator=True
    )

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
        print("\n  Home team mismatches:")
        print(home_mismatches)

    if len(away_mismatches) > 0:
        print("\n  Away team mismatches:")
        print(away_mismatches)

    return (
        len(home_mismatches) == 0
        and len(away_mismatches) == 0
    )


def validate_merge(
    game_features,
    advanced_features
):
    """
    Test the actual one-to-one merge without
    saving anything.
    """

    merged = game_features.merge(
        advanced_features,
        left_on="gameId",
        right_on="id",
        how="inner",
        suffixes=("_game", "_advanced"),
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

    expected_rows = len(game_features)

    if len(merged) == expected_rows:
        print(
            "  PASS: Merge preserved "
            "all game feature rows."
        )
        return merged, True

    else:
        print(
            "  FAIL: Merge did not preserve "
            "all game feature rows."
        )
        return merged, False


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

    game_columns = set(game_features.columns)
    advanced_columns = set(advanced_features.columns)

    overlapping_columns = (
        game_columns
        & advanced_columns
    )

    # These are the two different identifiers
    # used by the source datasets.
    overlapping_columns.discard("gameId")
    overlapping_columns.discard("id")

    print("\nOverlapping columns:")
    print(
        f"  {len(overlapping_columns)} "
        f"columns exist in both datasets."
    )

    if overlapping_columns:
        for column in sorted(overlapping_columns):
            print(f"  {column}")


if __name__ == "__main__":

    all_passed = True

    for year in range(2015, 2026):

        print("\n" + "=" * 70)
        print(
            f"VALIDATING FEATURE MERGE: {year}"
        )
        print("=" * 70)

        game_features, advanced_features = (
            load_feature_data(year)
        )

        # --------------------------------------------------
        # 1. Validate unique game IDs
        # --------------------------------------------------

        game_ids_valid = validate_unique_game_ids(
            game_features,
            "Game-level features",
            "gameId"
        )

        advanced_ids_valid = validate_unique_game_ids(
            advanced_features,
            "Advanced features",
            "id"
        )

        if not game_ids_valid or not advanced_ids_valid:
            all_passed = False

        # --------------------------------------------------
        # 2. Validate matching game IDs
        # --------------------------------------------------

        only_game, only_advanced = (
            validate_game_id_match(
                game_features,
                advanced_features
            )
        )

        if only_game or only_advanced:
            all_passed = False

        # --------------------------------------------------
        # 3. Validate teams
        # --------------------------------------------------

        game_information_valid = (
            validate_game_information(
                game_features,
                advanced_features
            )
        )

        if not game_information_valid:
            all_passed = False

        # --------------------------------------------------
        # 4. Validate actual merge
        # --------------------------------------------------

        try:
            merged, merge_valid = validate_merge(
                game_features,
                advanced_features
            )

            if not merge_valid:
                all_passed = False

        except Exception as e:
            print("\n  MERGE FAILED:")
            print(f"  {e}")
            all_passed = False

        # --------------------------------------------------
        # 5. Check overlapping columns
        # --------------------------------------------------

        validate_columns(
            game_features,
            advanced_features
        )

    # ------------------------------------------------------
    # Final result
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL VALIDATION RESULT")
    print("=" * 70)

    if all_passed:
        print(
            "PASS: All feature datasets "
            "passed merge validation."
        )
    else:
        print(
            "FAIL: One or more seasons "
            "require investigation."
        )