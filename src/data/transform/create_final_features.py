import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

GAMES_DIR = Path("data/raw/games")
GAME_FEATURES_DIR = Path("data/processed/game_features")
ADVANCED_FEATURES_DIR = Path("data/processed/features")
FINAL_FEATURES_DIR = Path("data/processed/final_features")


# ============================================================
# LOAD DATA
# ============================================================

def load_data(year):
    """
    Load original games, game-level features, and advanced
    statistics for a season.
    """

    games_path = GAMES_DIR / f"games_{year}.csv"

    game_features_path = (
        GAME_FEATURES_DIR / f"game_features_{year}.csv"
    )

    advanced_features_path = (
        ADVANCED_FEATURES_DIR / f"features_{year}.csv"
    )

    games = pd.read_csv(games_path)
    game_features = pd.read_csv(game_features_path)
    advanced_features = pd.read_csv(advanced_features_path)

    return games, game_features, advanced_features


# ============================================================
# VALIDATE INPUTS
# ============================================================

def validate_input_data(
    games,
    game_features,
    advanced_features,
    year
):

    print(f"\n{'=' * 70}")
    print(f"VALIDATING INPUT DATA: {year}")
    print(f"{'=' * 70}")

    # --------------------------------------------------------
    # Check game IDs
    # --------------------------------------------------------

    if "id" not in games.columns:
        raise ValueError(
            f"Original games for {year} are missing 'id'."
        )

    if "gameId" not in game_features.columns:
        raise ValueError(
            f"Game-level features for {year} are missing 'gameId'."
        )

    if "id" not in advanced_features.columns:
        raise ValueError(
            f"Advanced features for {year} are missing 'id'."
        )

    # --------------------------------------------------------
    # Check classifications
    # --------------------------------------------------------

    required_columns = [
        "homeClassification",
        "awayClassification"
    ]

    missing = [
        column
        for column in required_columns
        if column not in games.columns
    ]

    if missing:
        raise ValueError(
            f"Original games for {year} are missing: {missing}"
        )

    # --------------------------------------------------------
    # Check duplicate IDs
    # --------------------------------------------------------

    game_duplicates = games["id"].duplicated().sum()

    game_feature_duplicates = (
        game_features["gameId"].duplicated().sum()
    )

    advanced_duplicates = (
        advanced_features["id"].duplicated().sum()
    )

    print(f"Original games: {len(games)}")
    print(f"Game-level features: {len(game_features)}")
    print(f"Advanced statistics: {len(advanced_features)}")

    print(f"Duplicate game IDs: {game_duplicates}")
    print(
        f"Duplicate game-level IDs: "
        f"{game_feature_duplicates}"
    )
    print(
        f"Duplicate advanced IDs: "
        f"{advanced_duplicates}"
    )

    if game_duplicates > 0:
        raise ValueError(
            f"Original games contain duplicate IDs for {year}."
        )

    if game_feature_duplicates > 0:
        raise ValueError(
            f"Game-level features contain duplicate gameIds "
            f"for {year}."
        )

    if advanced_duplicates > 0:
        raise ValueError(
            f"Advanced features contain duplicate IDs "
            f"for {year}."
        )


# ============================================================
# CREATE OPTION B GAME UNIVERSE
# ============================================================

def create_option_b_games(games):
    """
    Keep every game involving at least one FBS team.

    FBS-FBS -> KEEP
    FBS-FCS -> KEEP
    FCS-FBS -> KEEP
    FCS-FCS -> DROP
    """

    fbs_mask = (
        (games["homeClassification"] == "fbs")
        |
        (games["awayClassification"] == "fbs")
    )

    option_b_games = games.loc[fbs_mask].copy()

    return option_b_games


# ============================================================
# MERGE GAME-LEVEL FEATURES
# ============================================================

def merge_game_features(option_b_games, game_features, year):
    """
    Merge game-level features onto the Option B game universe.
    """

    game_features_for_merge = game_features.copy()

    merged = option_b_games.merge(
        game_features_for_merge,
        left_on="id",
        right_on="gameId",
        how="left",
        validate="one_to_one",
        suffixes=("", "_game_features")
    )

    # Check that every eligible game received
    # game-level features.
    missing = merged["gameId"].isna().sum()

    if missing > 0:

        missing_games = merged.loc[
            merged["gameId"].isna(),
            [
                "id",
                "homeTeam",
                "awayTeam"
            ]
        ]

        print("\nGames missing game-level features:")
        print(
            missing_games.to_string(index=False)
        )

        raise ValueError(
            f"{missing} Option B games are missing "
            f"game-level features in {year}."
        )

    return merged


# ============================================================
# MERGE ADVANCED FEATURES
# ============================================================

def merge_advanced_features(
    game_features,
    advanced_features,
    year
):
    """
    Merge advanced statistics onto the game-level dataset.
    """

    advanced_for_merge = (
        advanced_features
        .rename(columns={"id": "gameId"})
        .copy()
    )

    merged = game_features.merge(
        advanced_for_merge,
        on="gameId",
        how="left",
        validate="one_to_one",
        suffixes=("", "_advanced")
    )

    return merged


# ============================================================
# VALIDATE OPTION B
# ============================================================

def validate_final_merge(
    option_b_games,
    final_features,
    advanced_features,
    year
):

    print(f"\n{'=' * 70}")
    print(f"VALIDATING FINAL MERGE: {year}")
    print(f"{'=' * 70}")

    # --------------------------------------------------------
    # Row counts
    # --------------------------------------------------------

    print("\nRow counts:")

    print(
        f"  Option B games: "
        f"{len(option_b_games)}"
    )

    print(
        f"  Final features: "
        f"{len(final_features)}"
    )

    if len(final_features) != len(option_b_games):

        raise ValueError(
            f"Final row count does not match Option B "
            f"game count for {year}."
        )

    # --------------------------------------------------------
    # Check FCS-FCS
    # --------------------------------------------------------

    fcs_fcs = (
        (final_features["homeClassification"] != "fbs")
        &
        (final_features["awayClassification"] != "fbs")
    ).sum()

    print(
        f"  FCS-FCS games remaining: "
        f"{fcs_fcs}"
    )

    if fcs_fcs > 0:

        raise ValueError(
            f"FCS-FCS games remain in final dataset "
            f"for {year}."
        )

    # --------------------------------------------------------
    # Check duplicate IDs
    # --------------------------------------------------------

    duplicates = (
        final_features["gameId"]
        .duplicated()
        .sum()
    )

    print(
        f"  Duplicate gameIds: "
        f"{duplicates}"
    )

    if duplicates > 0:

        raise ValueError(
            f"Duplicate gameIds detected in final "
            f"dataset for {year}."
        )

    # --------------------------------------------------------
    # Determine advanced-stat columns
    # --------------------------------------------------------

    advanced_columns = [
        column
        for column in advanced_features.columns
        if column != "id"
    ]

    # --------------------------------------------------------
    # Check missing advanced statistics
    # --------------------------------------------------------

    missing_advanced = (
        final_features[advanced_columns]
        .isna()
        .all(axis=1)
    )

    missing_count = missing_advanced.sum()

    print(
        f"  Games missing advanced statistics: "
        f"{missing_count}"
    )

    if missing_count > 0:

        missing_games = final_features.loc[
            missing_advanced,
            [
                "gameId",
                "homeTeam",
                "awayTeam",
                "homeClassification",
                "awayClassification"
            ]
        ]

        print(
            "\nGames missing advanced statistics:"
        )

        print(
            missing_games.to_string(
                index=False
            )
        )

        raise ValueError(
            f"{missing_count} Option B games are "
            f"missing advanced statistics for {year}."
        )

    # --------------------------------------------------------
    # Success
    # --------------------------------------------------------

    print("\nPASS:")
    print(
        "  All games involve at least one FBS team."
    )
    print(
        "  All games have game-level features."
    )
    print(
        "  All games have advanced statistics."
    )
    print(
        "  No duplicate games were created."
    )


# ============================================================
# SAVE
# ============================================================

def save_final_features(df, year):

    FINAL_FEATURES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        FINAL_FEATURES_DIR
        / f"final_features_{year}.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print("\nSaved:")
    print(f"  {output_path}")


# ============================================================
# PROCESS SEASON
# ============================================================

def process_season(year):

    try:

        # ----------------------------------------------------
        # Load
        # ----------------------------------------------------

        games, game_features, advanced_features = (
            load_data(year)
        )

        # ----------------------------------------------------
        # Validate inputs
        # ----------------------------------------------------

        validate_input_data(
            games,
            game_features,
            advanced_features,
            year
        )

        # ----------------------------------------------------
        # Option B filter
        # ----------------------------------------------------

        option_b_games = create_option_b_games(
            games
        )

        print(
            f"\nOption B eligible games: "
            f"{len(option_b_games)}"
        )

        # ----------------------------------------------------
        # Merge game-level features
        # ----------------------------------------------------

        merged_game_features = merge_game_features(
            option_b_games,
            game_features,
            year
        )

        # ----------------------------------------------------
        # Merge advanced statistics
        # ----------------------------------------------------

        final_features = merge_advanced_features(
            merged_game_features,
            advanced_features,
            year
        )

        # ----------------------------------------------------
        # Validate final dataset
        # ----------------------------------------------------

        validate_final_merge(
            option_b_games,
            final_features,
            advanced_features,
            year
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        save_final_features(
            final_features,
            year
        )

        return True

    except Exception as e:

        print(f"\nERROR processing {year}:")
        print(f"{type(e).__name__}: {e}")

        return False


# ============================================================
# MAIN
# ============================================================

def main():

    years = range(2015, 2026)

    results = {}

    for year in years:

        results[year] = process_season(year)

    print(f"\n{'=' * 70}")
    print("FINAL FEATURE CREATION")
    print(f"{'=' * 70}")

    for year, success in results.items():

        status = "PASS" if success else "FAIL"

        print(f"{year}: {status}")

    if all(results.values()):

        print(
            "\n" + "=" * 70
        )

        print(
            "SUCCESS: FINAL FEATURES CREATED "
            "FOR ALL SEASONS"
        )

        print(
            "=" * 70
        )

    else:

        print(
            "\n" + "=" * 70
        )

        print(
            "FAIL: One or more seasons could "
            "not be processed."
        )

        print(
            "=" * 70
        )


if __name__ == "__main__":
    main()