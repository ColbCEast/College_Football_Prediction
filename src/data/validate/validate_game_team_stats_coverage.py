import pandas as pd
from pathlib import Path


YEARS = range(2015, 2026)

RAW_GAMES_DIR = Path("data/raw/games")
TEAM_STATS_DIR = Path("data/processed/game_stats/team_level")
GAME_FEATURES_DIR = Path("data/processed/features/game_level")


def validate_year(year):

    print("\n" + "=" * 70)
    print(f"VALIDATING GAME TEAM STATS COVERAGE: {year}")
    print("=" * 70)

    raw_path = RAW_GAMES_DIR / f"games_{year}.csv"
    team_stats_path = (
        TEAM_STATS_DIR / f"game_team_stats_{year}.csv"
    )
    game_features_path = (
        GAME_FEATURES_DIR / f"game_features_{year}.csv"
    )

    raw = pd.read_csv(raw_path)
    team_stats = pd.read_csv(team_stats_path)
    game_features = pd.read_csv(game_features_path)

    # ------------------------------------------------------------
    # Option B games
    # ------------------------------------------------------------

    option_b = raw[
        (raw["homeClassification"] == "fbs") |
        (raw["awayClassification"] == "fbs")
    ].copy()

    option_b_ids = set(
        option_b["id"].astype(int)
    )

    # ------------------------------------------------------------
    # IDs present in game_team_stats
    # ------------------------------------------------------------

    team_stats_ids = set(
        team_stats["gameId"].astype(int)
    )

    # ------------------------------------------------------------
    # IDs present in final game-level features
    # ------------------------------------------------------------

    game_feature_ids = set(
        game_features["gameId"].astype(int)
    )

    # ------------------------------------------------------------
    # Games completely missing from game_team_stats
    # ------------------------------------------------------------

    missing_team_stats = sorted(
        option_b_ids - team_stats_ids
    )

    # ------------------------------------------------------------
    # Games present in team_stats but missing game_features
    # ------------------------------------------------------------

    missing_game_features = sorted(
        team_stats_ids.intersection(option_b_ids)
        - game_feature_ids
    )

    print(f"\nOption B games:              {len(option_b_ids)}")
    print(f"Game-team-stat games:        {len(team_stats_ids)}")
    print(f"Game-level feature games:    {len(game_feature_ids)}")

    print(
        f"\nOption B games missing entirely "
        f"from game_team_stats:       {len(missing_team_stats)}"
    )

    print(
        f"Option B games present in "
        f"team_stats but missing game_features: "
        f"{len(missing_game_features)}"
    )

    # ------------------------------------------------------------
    # Show games missing from game_team_stats
    # ------------------------------------------------------------

    if missing_team_stats:

        missing = option_b[
            option_b["id"]
            .astype(int)
            .isin(missing_team_stats)
        ].copy()

        columns = [
            "id",
            "week",
            "seasonType",
            "homeTeam",
            "homeClassification",
            "awayTeam",
            "awayClassification",
        ]

        print("\n" + "-" * 70)
        print("GAMES MISSING FROM GAME_TEAM_STATS")
        print("-" * 70)

        print(
            missing[columns]
            .sort_values(["week", "homeTeam"])
            .to_string(index=False)
        )

    # ------------------------------------------------------------
    # Check home/away rows for all Option B games
    # ------------------------------------------------------------

    option_b_team_stats = team_stats[
        team_stats["gameId"]
        .astype(int)
        .isin(option_b_ids)
    ].copy()

    location_counts = (
        option_b_team_stats
        .groupby(["gameId", "homeAway"])
        .size()
        .unstack(fill_value=0)
    )

    if "home" not in location_counts.columns:
        location_counts["home"] = 0

    if "away" not in location_counts.columns:
        location_counts["away"] = 0

    incomplete_games = location_counts[
        (location_counts["home"] != 1) |
        (location_counts["away"] != 1)
    ]

    print(
        f"\nOption B games with incomplete "
        f"home/away team stats:        {len(incomplete_games)}"
    )

    if len(incomplete_games) > 0:

        print("\nIncomplete games:")

        print(
            incomplete_games
            .sort_index()
            .to_string()
        )

    # ------------------------------------------------------------
    # Final diagnosis
    # ------------------------------------------------------------

    print("\n" + "-" * 70)
    print("DIAGNOSIS")
    print("-" * 70)

    if missing_team_stats:

        print(
            "\nFAIL: Some Option B games are completely absent "
            "from game_team_stats."
        )

        print(
            "The problem occurs BEFORE create_game_features.py."
        )

    elif missing_game_features:

        print(
            "\nFAIL: Games exist in game_team_stats but are missing "
            "from game_features."
        )

        print(
            "The problem is inside create_game_features.py."
        )

    else:

        print(
            "\nPASS: Every Option B game has complete "
            "game_team_stats coverage and game-level features."
        )


def main():

    for year in YEARS:

        try:
            validate_year(year)

        except Exception as e:

            print(f"\nERROR processing {year}:")
            print(
                f"{type(e).__name__}: {e}"
            )


if __name__ == "__main__":
    main()