import pandas as pd


# ============================================================
# PATHS
# ============================================================

ADVANCED_STATS_PATH = "data/processed/features/base/features_{}.csv"
GAME_FEATURES_PATH = "data/processed/features/game_level/game_features_{}.csv"
GAMES_PATH = "data/raw/games/games_{}.csv"


# ============================================================
# VALIDATE ONE SEASON
# ============================================================

def validate_season(year):

    print("\n" + "=" * 70)
    print(f"VALIDATING ADVANCED STAT COVERAGE: {year}")
    print("=" * 70)

    # --------------------------------------------------------
    # Load advanced statistics
    # --------------------------------------------------------

    advanced = pd.read_csv(
        ADVANCED_STATS_PATH.format(year)
    )

    print("\nAdvanced statistics:")
    print(f"  Rows: {len(advanced)}")
    print("  ID column: id")

    if "id" not in advanced.columns:
        raise ValueError(
            "Advanced statistics file is missing 'id'."
        )

    advanced_ids = set(
        pd.to_numeric(
            advanced["id"],
            errors="coerce"
        ).dropna().astype(int)
    )

    # --------------------------------------------------------
    # Load game-level features
    # --------------------------------------------------------

    game_features = pd.read_csv(
        GAME_FEATURES_PATH.format(year)
    )

    print("\nGame-level features:")
    print(f"  Rows: {len(game_features)}")
    print("  ID column: gameId")

    if "gameId" not in game_features.columns:
        raise ValueError(
            "Game-level features file is missing 'gameId'."
        )

    game_features["gameId"] = pd.to_numeric(
        game_features["gameId"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Load original games data
    # --------------------------------------------------------

    games = pd.read_csv(
        GAMES_PATH.format(year)
    )

    print("\nOriginal games:")
    print(f"  Rows: {len(games)}")

    # --------------------------------------------------------
    # Verify required columns
    # --------------------------------------------------------

    required_game_columns = [
        "id",
        "homeClassification",
        "awayClassification",
        "homeTeam",
        "awayTeam"
    ]

    missing_columns = [
        column
        for column in required_game_columns
        if column not in games.columns
    ]

    if missing_columns:

        raise ValueError(
            "Original games file is missing columns: "
            + ", ".join(missing_columns)
        )

    # --------------------------------------------------------
    # Normalize original game IDs
    # --------------------------------------------------------

    games["id"] = pd.to_numeric(
        games["id"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Restrict original games to games in game-level data
    # --------------------------------------------------------

    game_level_ids = set(
        game_features["gameId"].dropna().astype(int)
    )

    games = games[
        games["id"].isin(game_level_ids)
    ].copy()

    print(
        f"  Games represented in game-level features: "
        f"{len(games)}"
    )

    # --------------------------------------------------------
    # Classify games
    # --------------------------------------------------------

    def classify_game(row):

        home = str(
            row["homeClassification"]
        ).lower()

        away = str(
            row["awayClassification"]
        ).lower()

        if home == "fbs" and away == "fbs":
            return "FBS-FBS"

        if (
            home == "fbs" and away == "fcs"
        ) or (
            home == "fcs" and away == "fbs"
        ):
            return "FBS-FCS"

        if home == "fcs" and away == "fcs":
            return "FCS-FCS"

        return "OTHER"

    games["game_type"] = games.apply(
        classify_game,
        axis=1
    )

    # --------------------------------------------------------
    # Determine whether each game has advanced statistics
    # --------------------------------------------------------

    games["has_advanced_stats"] = (
        games["id"].isin(advanced_ids)
    )

    # --------------------------------------------------------
    # Overall classification
    # --------------------------------------------------------

    print("\nGame classification:")

    print(
        games["game_type"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Coverage by game type
    # --------------------------------------------------------

    coverage = (
        games
        .groupby("game_type")["has_advanced_stats"]
        .agg(
            games="count",
            with_stats="sum"
        )
    )

    coverage["without_stats"] = (
        coverage["games"]
        - coverage["with_stats"]
    )

    coverage["coverage_pct"] = (
        coverage["with_stats"]
        / coverage["games"]
        * 100
    )

    print("\nAdvanced statistics coverage by game type:")

    print(coverage.to_string())

    # --------------------------------------------------------
    # Find missing games
    # --------------------------------------------------------

    missing = games[
        ~games["has_advanced_stats"]
    ].copy()

    # --------------------------------------------------------
    # Key validation
    # --------------------------------------------------------

    missing_fbs_fbs = missing[
        missing["game_type"] == "FBS-FBS"
    ]

    missing_fbs_fcs = missing[
        missing["game_type"] == "FBS-FCS"
    ]

    missing_fcs_fcs = missing[
        missing["game_type"] == "FCS-FCS"
    ]

    missing_other = missing[
        missing["game_type"] == "OTHER"
    ]

    print("\n" + "-" * 70)
    print("OPTION B VALIDATION")
    print("-" * 70)

    print(
        f"\nMissing FBS-FBS games: {len(missing_fbs_fbs)}"
    )

    print(
        f"Missing FBS-FCS games: {len(missing_fbs_fcs)}"
    )

    print(
        f"Missing FCS-FCS games: {len(missing_fcs_fcs)}"
    )

    print(
        f"Missing OTHER games:   {len(missing_other)}"
    )

    # --------------------------------------------------------
    # Show missing FBS games
    # --------------------------------------------------------

    missing_fbs = missing[
        missing["game_type"].isin(
            ["FBS-FBS", "FBS-FCS"]
        )
    ]

    if len(missing_fbs) > 0:

        print("\n" + "=" * 70)
        print("MISSING GAMES INVOLVING FBS")
        print("=" * 70)

        display_columns = [
            "id",
            "homeTeam",
            "homeClassification",
            "awayTeam",
            "awayClassification",
            "game_type"
        ]

        if "week" in missing.columns:
            display_columns.insert(1, "week")

        print(
            missing_fbs[
                display_columns
            ].sort_values("id").to_string(index=False)
        )

    # --------------------------------------------------------
    # Show FCS-FCS missing games
    # --------------------------------------------------------

    if len(missing_fcs_fcs) > 0:

        print("\nFCS-FCS games without advanced statistics:")

        display_columns = [
            "id",
            "homeTeam",
            "awayTeam",
            "game_type"
        ]

        if "week" in missing_fcs_fcs.columns:
            display_columns.insert(1, "week")

        print(
            missing_fcs_fcs[
                display_columns
            ].sort_values("id").to_string(index=False)
        )

    # --------------------------------------------------------
    # Determine result
    # --------------------------------------------------------

    if len(missing_fbs) > 0:

        print(
            "\nFAIL: Games involving an FBS team are missing "
            "advanced statistics."
        )

        return False

    elif len(missing_fcs_fcs) > 0:

        print(
            "\nPASS: All games involving FBS teams have "
            "advanced statistics."
        )

        print(
            "The missing advanced statistics are from "
            "FCS-FCS games."
        )

        return True

    else:

        print(
            "\nPASS: Every game has advanced statistics."
        )

        return True


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    years = range(2015, 2026)

    results = {}

    for year in years:

        try:

            results[year] = validate_season(year)

        except Exception as e:

            print(f"\nERROR processing {year}:")
            print(f"{type(e).__name__}: {e}")

            results[year] = False

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL OPTION B VALIDATION")
    print("=" * 70)

    for year, result in results.items():

        print(
            f"{year}: {'PASS' if result else 'FAIL'}"
        )

    print()

    if all(results.values()):

        print(
            "PASS: Option B is validated across all seasons."
        )

        print(
            "Every game involving an FBS team has advanced "
            "statistics."
        )

    else:

        print(
            "FAIL: At least one season contains a game involving "
            "an FBS team without advanced statistics."
        )