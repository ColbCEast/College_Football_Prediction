import pandas as pd


def load_data(year):

    games = pd.read_csv(
        f"data/raw/games/games_{year}.csv"
    )

    advanced_stats = pd.read_csv(
        f"data/processed/features/base/features_{year}.csv"
    )

    game_team_stats = pd.read_csv(
        f"data/raw/game_team_stats/game_team_stats_{year}.csv"
    )

    return games, advanced_stats, game_team_stats


def validate_raw_coverage(
    games,
    advanced_stats,
    game_team_stats,
    year
):

    print("\n" + "=" * 70)
    print(f"VALIDATING RAW GAME-TEAM-STATS COVERAGE: {year}")
    print("=" * 70)

    # --------------------------------------------------
    # Identify Option B games from raw games
    # --------------------------------------------------

    option_b = games[
        (games["homeClassification"] == "fbs") |
        (games["awayClassification"] == "fbs")
    ].copy()

    print(f"\nRaw games:             {len(games)}")
    print(f"Option B games:        {len(option_b)}")
    print(f"Advanced statistics:   {len(advanced_stats)}")
    print(f"Raw game-team-stats:   {len(game_team_stats)}")

    # --------------------------------------------------
    # Unique game IDs in raw game-team-stats
    # --------------------------------------------------

    raw_team_stat_ids = set(
        game_team_stats["gameId"].dropna().unique()
    )

    option_b_ids = set(
        option_b["id"].dropna().unique()
    )

    advanced_ids = set(
        advanced_stats["id"].dropna().unique()
    )

    # --------------------------------------------------
    # Find Option B games missing from raw team stats
    # --------------------------------------------------

    missing_from_raw = option_b[
        ~option_b["id"].isin(raw_team_stat_ids)
    ].copy()

    print(
        "\nOption B games missing from RAW game_team_stats:",
        len(missing_from_raw)
    )

    if len(missing_from_raw) > 0:

        print("\n" + "-" * 70)
        print("GAMES MISSING FROM RAW GAME_TEAM_STATS")
        print("-" * 70)

        columns = [
            "id",
            "week",
            "seasonType",
            "homeTeam",
            "homeClassification",
            "awayTeam",
            "awayClassification"
        ]

        print(
            missing_from_raw[columns].to_string(
                index=False
            )
        )

    # --------------------------------------------------
    # Check whether missing games have advanced stats
    # --------------------------------------------------

    missing_with_advanced = missing_from_raw[
        missing_from_raw["id"].isin(advanced_ids)
    ]

    missing_without_advanced = missing_from_raw[
        ~missing_from_raw["id"].isin(advanced_ids)
    ]

    print("\n" + "-" * 70)
    print("CROSS-CHECK WITH ADVANCED STATISTICS")
    print("-" * 70)

    print(
        "Missing raw team stats + HAS advanced stats:",
        len(missing_with_advanced)
    )

    print(
        "Missing raw team stats + NO advanced stats:",
        len(missing_without_advanced)
    )

    if len(missing_with_advanced) > 0:

        print("\nGames missing from raw team stats but present")
        print("in advanced statistics:")

        columns = [
            "id",
            "week",
            "homeTeam",
            "awayTeam"
        ]

        print(
            missing_with_advanced[columns].to_string(
                index=False
            )
        )

    # --------------------------------------------------
    # Check raw team stats completeness
    # --------------------------------------------------

    option_b_team_stats = game_team_stats[
        game_team_stats["gameId"].isin(option_b_ids)
    ].copy()

    team_counts = (
        option_b_team_stats
        .groupby("gameId")
        .size()
    )

    incomplete_games = option_b[
        option_b["id"].isin(team_counts.index) &
        (
            option_b["id"].map(team_counts) != 2
        )
    ].copy()

    print("\n" + "-" * 70)
    print("HOME/AWAY TEAM-STAT COMPLETENESS")
    print("-" * 70)

    print(
        "Option B games with exactly 2 team-stat rows:",
        (
            option_b["id"].isin(team_counts.index) &
            (option_b["id"].map(team_counts) == 2)
        ).sum()
    )

    print(
        "Option B games with incomplete team stats:",
        len(incomplete_games)
    )

    if len(incomplete_games) > 0:

        print("\nIncomplete games:")

        columns = [
            "id",
            "homeTeam",
            "awayTeam"
        ]

        incomplete_display = incomplete_games[
            columns
        ].copy()

        incomplete_display["teamStatRows"] = (
            incomplete_display["id"]
            .map(team_counts)
            .fillna(0)
            .astype(int)
        )

        print(
            incomplete_display.to_string(
                index=False
            )
        )

    # --------------------------------------------------
    # Final diagnosis
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print(f"DIAGNOSIS: {year}")
    print("=" * 70)

    if len(missing_from_raw) == 0:

        print(
            "\nPASS:"
            "\nEvery Option B game exists in the raw "
            "game_team_stats dataset."
        )

        if len(incomplete_games) == 0:

            print(
                "Every Option B game has both team-stat rows."
            )

    else:

        print(
            "\nFAIL:"
            "\nSome Option B games are missing entirely "
            "from the raw game_team_stats dataset."
        )

        if len(missing_with_advanced) > 0:

            print(
                "\nIMPORTANT:"
                "\nSome missing games HAVE advanced statistics."
                "\nThis strongly indicates a problem with "
                "the game-team-stats API pull."
            )


if __name__ == "__main__":

    results = {}

    for year in range(2015, 2026):

        try:

            games, advanced_stats, game_team_stats = (
                load_data(year)
            )

            validate_raw_coverage(
                games,
                advanced_stats,
                game_team_stats,
                year
            )

        except Exception as e:

            print(
                f"\nERROR processing {year}:"
            )

            print(
                f"{type(e).__name__}: {e}"
            )