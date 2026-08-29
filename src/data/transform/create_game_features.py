import pandas as pd


def load_team_stats(path):
    return pd.read_csv(path)


def create_game_level_data(team_stats):
    home = team_stats[
        team_stats["homeAway"] == "home"
    ].copy()

    away = team_stats[
        team_stats["homeAway"] == "away"
    ].copy()

    print("Home rows: ", len(home))
    print("Away rows: ", len(away))

    return home, away


def validate_games(team_stats):

    game_location_counts = (
        team_stats
        .groupby(["gameId", "homeAway"])
        .size()
        .unstack(fill_value=0)
    )

    print("\nGame validation:")
    print(
        "Total unique games:",
        team_stats["gameId"].nunique()
    )

    print(
        "Games with exactly 1 home row:",
        (game_location_counts["home"] == 1).sum()
    )

    print(
        "Games with exactly 1 away row:",
        (game_location_counts["away"] == 1).sum()
    )

    print(
        "Games with multiple home rows:",
        (game_location_counts["home"] > 1).sum()
    )

    print(
        "Games with multiple away rows:",
        (game_location_counts["away"] > 1).sum()
    )

    print(
        "Games missing home row:",
        (game_location_counts["home"] == 0).sum()
    )

    print(
        "Games missing away row:",
        (game_location_counts["away"] == 0).sum()
    )


def merge_home_away(home, away):
    # Merge the home and away datasets
    # ensuring a 1-1 match in the pipeline
    game_features = home.merge(
        away,
        on = "gameId",
        suffixes = ("_home", "_away"),
        how = "inner",
        validate = "one_to_one")

    return game_features


def clean_game_features(game_features):
    # Game-level information to keep
    game_info = [
        "gameId",
        "season_home",
        "week_home",
        "startDate_home",
        "homeTeam_home",
        "awayTeam_away",
        "conference_home",
        "conference_away",
    ]

    # Columns that are duplicated or unnecessary
    metadata_to_drop = [
        "Unnamed: 0_home",
        "Unnamed: 0_away",

        "season_away",
        "week_away",
        "seasonType_home",
        "seasonType_away",
        "startDate_away",

        "team_home",
        "team_away",
        "homeTeam_away",
        "awayTeam_home",

        "conference_home",
        "conference_away",

        "homeAway_home",
        "homeAway_away",

        "opponent_home",
        "opponent_away",

        "completed_home",
        "completed_away",

        "isHome_home",
        "isAway_home",
        "isHome_away",
        "isAway_away",
    ]

    # Build list of columns to remove
    columns_to_drop = []

    for col in game_features.columns:
        # Remove CSV index columns

        if col.startswith("Unnamed:"):
            columns_to_drop.append(col)
            continue

        # Remove location-mismatched statistics

        if col.startswith("home") and col.endswith("_away"):
            columns_to_drop.append(col)
            continue

        if col.startswith("away") and col.endswith("_home"):
            columns_to_drop.append(col)
            continue

        # Remove duplicated / unnecessary metadata

        if col in metadata_to_drop:
            columns_to_drop.append(col)
            continue

    # Create cleaned dataframe

    cleaned = game_features.drop(
        columns=columns_to_drop
    ).copy()

    # Rename important game-level columns

    # This removes the artificial "_home" / "_away" suffix
    # from columns that are genuinely game-level.

    rename_columns = {
        "season_home": "season",
        "week_home": "week",
        "seasonType_home": "seasonType",
        "startDate_home": "startDate",
        "homeTeam_home": "homeTeam",
        "awayTeam_away": "awayTeam",
    }

    cleaned = cleaned.rename(
        columns=rename_columns
    )

    # Print cleaning summary
    print("\nRemoved columns:")

    for col in columns_to_drop:
        print(f"  {col}")

    print(f"\nRemoved {len(columns_to_drop)} columns")
    print(f"Original shape: {game_features.shape}")
    print(f"Cleaned shape: {cleaned.shape}")

    return cleaned


if __name__ == "__main__":
    for year in range(2015, 2026):
        path = (
            f"data/processed/game_stats/team_level/game_team_stats_{year}.csv"
        )

        team_stats = load_team_stats(path)

        validate_games(team_stats)

        home, away = create_game_level_data(
            team_stats
        )

        game_features = merge_home_away(
            home,
            away
        )

        game_features = clean_game_features(game_features)

        print("\nCleaned game-level shape:")
        print(game_features.shape)

        game_features.to_csv(f"data/processed/features/game_level/game_features_{year}.csv", index = False)