import pandas as pd

model_features = [
    "pregame_offense_ppa",
    "pregame_offense_successRate",
    "pregame_offense_explosiveness",
    "pregame_defense_ppa",
    "pregame_defense_successRate",
    "pregame_defense_explosiveness"
]

def create_matchup_dataset(year):
    """
    Create a matchup-level dataset by combining processed game
    data with pre-game team statistics
    """

    games = pd.read_csv(f"data/processed/games/fbs_games_2015_2025.csv")

    # Filter the season for the purpose of the prototype
    games = games[
        games["season"] == year
    ].copy()

    # Load pre-game statistics
    pregame_stats = pd.read_csv(f"data/processed/game_stats/pregame_stats_{year}.csv")

    # Select only the needed columns
    stats = pregame_stats[["gameId", "team"] + model_features].copy()

    """
    HOME TEAM STATISTICS
    """

    home_stats = stats.rename(
        columns = {
            "team": "homeTeam",
            **{
                feature: f"home_{feature}"
                for feature in model_features
            }
        }
    )

    """
    AWAY TEAM STATISTICS
    """

    away_stats = stats.rename(
        columns = {
            "team": "awayTeam",
            **{
                feature: f"away_{feature}"
                for feature in model_features
            }
        }
    )

    """
    MERGE HOME TEAM STATISTICS
    """

    games = games.merge(
        home_stats,
        left_on = ["id", "homeTeam"],
        right_on = ["gameId", "homeTeam"],
        how = "left"
    )

    """
    MERGE AWAY TEAM STATISTICS
    """

    games = games.merge(
        away_stats,
        left_on = ["id", "awayTeam"],
        right_on = ["gameId", "awayTeam"],
        how = "left"
    )

    # Remove duplicate game ID columns
    games = games.drop(columns = ["gameId_x", "gameId_y"], errors = "ignore")

    return games

if __name__ == "__main__":
    games = create_matchup_dataset(2025)

    games.to_csv("data/processed/features/features_2025.csv", index = False)

    print("Shape: ", games.shape)

    print("\nSelected columns:")
    print(games[["id",
                 "week",
                 "homeTeam",
                 "awayTeam",
                 "home_pregame_offense_ppa",
                 "away_pregame_offense_ppa",
                 "home_pregame_defense_ppa",
                 "away_pregame_defense_ppa"]].head(20))

    print("\nMissing pre-game statistics:")
    print(games[["home_pregame_offense_ppa",
                 "away_pregame_offense_ppa",
                 "home_pregame_defense_ppa",
                 "away_pregame_defense_ppa"]].isna().sum())