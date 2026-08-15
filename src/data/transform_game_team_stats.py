import pandas as pd


def load_data(year):
    games = pd.read_csv(f"data/raw/games/games_{year}.csv")

    game_team_stats = pd.read_csv(f"data/raw/game_team_stats/game_team_stats_{year}.csv")

    return games, game_team_stats


def merge_game_data(games, game_team_stats):
    game_info = games[
        ["id",
         "startDate",
         "completed",
         "homeTeam",
         "awayTeam",
         "homePoints",
         "awayPoints"]
    ].copy()

    game_info = game_info.rename(columns = {"id": "gameId"})

    team_stats = game_team_stats.merge(
        game_info,
        on = "gameId",
        how = "left"
    )

    team_stats["opponent"] = team_stats.apply(
        lambda row: (
            row["awayTeam"]
            if row["homeAway"] == "home"
            else row["homeTeam"]
        ),
        axis = 1
    )

    return team_stats

def create_game_outcomes(team_stats):
    team_stats["pointsFor"] = team_stats["points"]

    team_stats["pointsAgainst"] = team_stats.apply(
        lambda row: (
            row["awayPoints"]
            if row["homeAway"] == "home" 
            else row["homePoints"]
        ),
        axis = 1
    )

    team_stats["pointDifferential"] = (
        team_stats["pointsFor"] - team_stats["pointsAgainst"]
    )

    team_stats["win"] = (
        team_stats["pointsFor"] > team_stats["pointsAgainst"]
    ).astype(int)

    return team_stats

def sort_games(team_stats):
    team_stats["startDate"] = pd.to_datetime(
        team_stats["startDate"],
        utc = True
    )

    team_stats = team_stats.sort_values(
        ["team", "startDate"]
    ).reset_index(drop = True)

    return team_stats


def create_numeric_features(team_stats):
    # Completion Percentage
    completion = team_stats["completionAttempts"].str.split("-", expand = True)

    team_stats["completions"] = pd.to_numeric(completion[0])
    team_stats["passAttempts"] = pd.to_numeric(completion[1])

    team_stats["completionPct"] = (
        team_stats["completions"] / team_stats["passAttempts"]
    )

    # Third Down Efficiency
    third_down = team_stats["thirdDownEff"].str.split("-", expand = True)

    team_stats["thirdDownConversions"] = pd.to_numeric(third_down[0])
    team_stats["thirdDownAttempts"] = pd.to_numeric(third_down[1])

    team_stats["thirdDownPct"] = (
        team_stats["thirdDownConversions"] / team_stats["thirdDownAttempts"]
    )

    # Fourth Down Efficiency
    fourth_down = team_stats["fourthDownEff"].str.split("-", expand = True)

    team_stats["fourthDownConversions"] = pd.to_numeric(fourth_down[0])
    team_stats["fourthDownAttempts"] = pd.to_numeric(fourth_down[1])

    team_stats["fourthDownPct"] = (
        team_stats["fourthDownConversions"] / team_stats["fourthDownAttempts"]
    )

    # Penalities
    penalties = team_stats["totalPenaltiesYards"].str.split("-", expand = True)

    team_stats["penalties"] = pd.to_numeric(penalties[0])
    team_stats["penaltyYards"] = pd.to_numeric(penalties[1])

    # Possession Time
    possession = team_stats["possessionTime"].str.split(":", expand = True)

    team_stats["possessionSeconds"] = (
        pd.to_numeric(possession[0]) * 60 + pd.to_numeric(possession[1])
    )

    return team_stats


def create_pre_game_stats(team_stats):

    team_stats = team_stats.copy()

    # Number of games played before current game
    team_stats["gamesBefore"] = (
        team_stats
        .groupby("team")
        .cumcount()
    )

    # Wins before current game
    team_stats["winsBefore"] = (
        team_stats
        .groupby("team")["win"]
        .cumsum()
        .sub(team_stats["win"])
    )

    # Statistics for cumulative pre-game calculations
    stat_columns = [
        "pointsFor",
        "pointsAgainst",
        "rushingYards",
        "rushingAttempts",
        "netPassingYards",
        "passAttempts",
        "completions",
        "totalYards",
        "firstDowns",
        "rushingTDs",
        "passingTDs",
        "thirdDownConversions",
        "thirdDownAttempts",
        "fourthDownConversions",
        "fourthDownAttempts",
        "penalties",
        "penaltyYards",
        "possessionSeconds",
        "turnovers",
        "fumblesLost",
        "interceptions",
        "sacks",
        "qbHurries",
        "passesDeflected",
        "tacklesForLoss",
        "defensiveTDs",
    ]

    pre_game_features = {}

    for stat in stat_columns:

        before_column = f"{stat}Before"
        avg_column = f"{stat}AvgBefore"

        cumulative = (
            team_stats
            .groupby("team")[stat]
            .cumsum()
        )

        pre_game_features[before_column] = (
            cumulative
            - team_stats[stat]
        )

        pre_game_features[avg_column] = (
            pre_game_features[before_column]
            / team_stats["gamesBefore"]
        )

    # Point differential
    point_differential = (
        team_stats["pointsFor"]
        - team_stats["pointsAgainst"]
    )

    cumulative_point_differential = (
        point_differential
        .groupby(team_stats["team"])
        .cumsum()
    )

    pre_game_features["pointDifferentialBefore"] = (
        cumulative_point_differential
        - point_differential
    )

    pre_game_features["pointDifferentialAvgBefore"] = (
        pre_game_features["pointDifferentialBefore"]
        / team_stats["gamesBefore"]
    )

    # Rate statistics
    pre_game_features["completionPctBefore"] = (
        pre_game_features["completionsBefore"]
        / pre_game_features["passAttemptsBefore"]
    )

    pre_game_features["thirdDownPctBefore"] = (
        pre_game_features["thirdDownConversionsBefore"]
        / pre_game_features["thirdDownAttemptsBefore"]
    )

    pre_game_features["fourthDownPctBefore"] = (
        pre_game_features["fourthDownConversionsBefore"]
        / pre_game_features["fourthDownAttemptsBefore"]
    )

    pre_game_features["yardsPerRushAttemptBefore"] = (
        pre_game_features["rushingYardsBefore"]
        / pre_game_features["rushingAttemptsBefore"]
    )

    pre_game_features["yardsPerPassAttemptBefore"] = (
        pre_game_features["netPassingYardsBefore"]
        / pre_game_features["passAttemptsBefore"]
    )

    # Win percentage
    pre_game_features["winPctBefore"] = (
        team_stats["winsBefore"]
        / team_stats["gamesBefore"]
    )

    # Add all generated features at once
    team_stats = pd.concat(
        [
            team_stats,
            pd.DataFrame(pre_game_features, index=team_stats.index)
        ],
        axis=1
    )

    return team_stats


if __name__ == "__main__":
    games, game_team_stats = load_data(2025)

    team_stats = merge_game_data(games, game_team_stats)

    team_stats = create_game_outcomes(team_stats)

    team_stats = sort_games(team_stats)

    team_stats = create_numeric_features(team_stats)

    team_stats = create_pre_game_stats(team_stats)

    print(team_stats[
        team_stats["team"] == "Kansas State"
    ][
        ["startDate",
         "week",
         "gamesBefore",
         "winsBefore",
         "winPctBefore",
         "pointsForAvgBefore",
         "rushingYardsAvgBefore",
         "netPassingYardsAvgBefore",
         "totalYardsAvgBefore",
         "completionPctBefore",
         "thirdDownPctBefore",
         "fourthDownPctBefore",
         "yardsPerRushAttemptBefore",
         "yardsPerPassAttemptBefore",]
    ].to_string(index = False))