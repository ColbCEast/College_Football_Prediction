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


def ensure_stat_columns(team_stats):
    # 2015 data seems to have less statistics than 2016 onward.
    # To keep the schema consistent across seasons, add any missing
    # statistics as NaN.

    expected_stats = [
        "rushingTDs",
        "puntReturnYards",
        "puntReturnTDs",
        "puntReturns",
        "passingTDs",
        "kickReturnYards",
        "kickReturnTDs",
        "kickReturns",
        "kickingPoints",
        "interceptionYards",
        "interceptionTDs",
        "passesIntercepted",
        "fumblesRecovered",
        "totalFumbles",
        "possessionTime",
        "interceptions",
        "fumblesLost",
        "turnovers",
        "totalPenaltiesYards",
        "yardsPerRushAttempt",
        "rushingAttempts",
        "rushingYards",
        "yardsPerPass",
        "completionAttempts",
        "netPassingYards",
        "totalYards",
        "fourthDownEff",
        "thirdDownEff",
        "firstDowns",
        "tacklesForLoss",
        "defensiveTDs",
        "tackles",
        "sacks",
        "qbHurries",
        "passesDeflected",
    ]

    for column in expected_stats:
        if column not in team_stats.columns:
            team_stats[column] = pd.NA

    # Columns that should be numeric
    numeric_stats = [
        "points",
        "rushingTDs",
        "puntReturnYards",
        "puntReturnTDs",
        "puntReturns",
        "passingTDs",
        "kickReturnYards",
        "kickReturnTDs",
        "kickReturns",
        "kickingPoints",
        "interceptionYards",
        "interceptionTDs",
        "passesIntercepted",
        "fumblesRecovered",
        "totalFumbles",
        "interceptions",
        "fumblesLost",
        "turnovers",
        "yardsPerRushAttempt",
        "rushingAttempts",
        "rushingYards",
        "yardsPerPass",
        "netPassingYards",
        "totalYards",
        "firstDowns",
        "tacklesForLoss",
        "defensiveTDs",
        "tackles",
        "sacks",
        "qbHurries",
        "passesDeflected",
        "completions",
        "passAttempts",
        "completionPct",
        "thirdDownConversions",
        "thirdDownAttempts",
        "thirdDownPct",
        "fourthDownConversions",
        "fourthDownAttempts",
        "fourthDownPct",
        "penalties",
        "penaltyYards",
        "possessionSeconds",
    ]

    for column in numeric_stats:
        if column in team_stats.columns:
            team_stats[column] = pd.to_numeric(
                team_stats[column],
                errors="coerce"
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


def create_recent_form_stats(team_stats):
    # This function is to create rolling average stats
    # for the last 3 games and the last 5 games
    
    team_stats = team_stats.copy()

    # Ensure that the dataframe is sorted chronologically within team
    team_stats = team_stats.sort_values(
        ["team", "startDate"]
    )

    # Create a list of stats that we'll calculate rolling averages for
    rolling_stats = [
        "pointsFor",
        "pointsAgainst",
        "pointDifferential",
        "rushingYards",
        "netPassingYards",
        "totalYards",
        "sacks",
        "qbHurries",
        "passesDeflected",
        "tacklesForLoss",
        "turnovers"
    ]

    # Create rolling averages, avoiding leakage by shifting
    for window in [3, 5]:
        for stat in rolling_stats:
            team_stats[f"{stat}AvgLast{window}"] = (
                team_stats.groupby("team")[stat]
                .transform(
                    lambda x: 
                    x.shift(1)
                    .rolling(window, min_periods = 1)
                    .mean()
                )
            )

    # Rolling win percentage
    for window in [3, 5]:
        team_stats[f"winPctLast{window}"] = (
            team_stats.groupby("team")["win"]
            .transform(
                lambda x:
                x.shift(1)
                .rolling(window, min_periods = 1)
                .mean()
            )
        )

    return team_stats


def create_location_stats(team_stats):

    team_stats = team_stats.copy()

    team_stats["isHome"] = team_stats["homeAway"] == "home"
    team_stats["isAway"] = team_stats["homeAway"] == "away"

    location_features = {}

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

    for mask, prefix in [
        (team_stats["isHome"], "home"),
        (team_stats["isAway"], "away")
    ]:

        location_indicator = mask.astype(int)

        # Games before this game at this location
        location_games = (
            location_indicator
            .groupby(team_stats["team"])
            .cumsum()
            - location_indicator
        )

        location_features[f"{prefix}GamesBefore"] = location_games

        # Wins before this game at this location
        location_win_values = (
            team_stats["win"] * location_indicator
        )

        location_wins = (
            location_win_values
            .groupby(team_stats["team"])
            .cumsum()
            - location_win_values
        )

        location_features[f"{prefix}WinsBefore"] = location_wins

        # Win percentage
        location_features[f"{prefix}WinPctBefore"] = (
            location_wins / location_games
        )

        # Cumulative statistics and averages
        for stat in stat_columns:

            stat_values = (
                team_stats[stat] * location_indicator
            )

            cumulative = (
                stat_values
                .groupby(team_stats["team"])
                .cumsum()
            )

            cumulative_before = (
                cumulative - stat_values
            )

            location_features[
                f"{prefix}{stat[0].upper()}{stat[1:]}Before"
            ] = cumulative_before

            location_features[
                f"{prefix}{stat[0].upper()}{stat[1:]}AvgBefore"
            ] = (
                cumulative_before / location_games
            )

        # Point differential
        differential = (
            team_stats["pointsFor"]
            - team_stats["pointsAgainst"]
        )

        differential_values = (
            differential * location_indicator
        )

        cumulative_differential = (
            differential_values
            .groupby(team_stats["team"])
            .cumsum()
        )

        differential_before = (
            cumulative_differential
            - differential_values
        )

        location_features[
            f"{prefix}PointDifferentialBefore"
        ] = differential_before

        location_features[
            f"{prefix}PointDifferentialAvgBefore"
        ] = (
            differential_before / location_games
        )

        # Efficiency metrics
        location_features[
            f"{prefix}CompletionPctBefore"
        ] = (
            location_features[f"{prefix}CompletionsBefore"]
            / location_features[f"{prefix}PassAttemptsBefore"]
        )

        location_features[
            f"{prefix}ThirdDownPctBefore"
        ] = (
            location_features[f"{prefix}ThirdDownConversionsBefore"]
            / location_features[f"{prefix}ThirdDownAttemptsBefore"]
        )

        location_features[
            f"{prefix}FourthDownPctBefore"
        ] = (
            location_features[f"{prefix}FourthDownConversionsBefore"]
            / location_features[f"{prefix}FourthDownAttemptsBefore"]
        )

        location_features[
            f"{prefix}YardsPerRushAttemptBefore"
        ] = (
            location_features[f"{prefix}RushingYardsBefore"]
            / location_features[f"{prefix}RushingAttemptsBefore"]
        )

        location_features[
            f"{prefix}YardsPerPassAttemptBefore"
        ] = (
            location_features[f"{prefix}NetPassingYardsBefore"]
            / location_features[f"{prefix}PassAttemptsBefore"]
        )

    # Add all newly created columns at once
    location_features_df = pd.DataFrame(
        location_features,
        index=team_stats.index
    )

    team_stats = pd.concat(
        [team_stats, location_features_df],
        axis=1
    )

    return team_stats

if __name__ == "__main__":
    for year in range (2015, 2026):
        games, game_team_stats = load_data(year)

        team_stats = merge_game_data(games, game_team_stats)

        team_stats = create_game_outcomes(team_stats)

        team_stats = sort_games(team_stats)

        team_stats = create_numeric_features(team_stats)

        team_stats = ensure_stat_columns(team_stats)

        team_stats = create_pre_game_stats(team_stats)

        team_stats = create_recent_form_stats(team_stats)

        team_stats = create_location_stats(team_stats)

        team_stats.to_csv(f"data/processed/game_team_stats/game_team_stats_{year}.csv")

        print(f"{year} data: {team_stats.shape}")