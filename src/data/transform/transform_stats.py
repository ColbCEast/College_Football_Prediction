import pandas as pd
import ast

# The CFBD Advanced game statistics import dictionaries of offensive and defensive statistics
# Flatten the dictionaries in order to make those statistics useable within a dataframe.
def flatten_dict(data, prefix = ""):
    """
    Recursively flatten a nested dictionary
    """

    flattened = {}

    for key, value in data.items():
        new_key = f"{prefix}_{key}" if prefix else key

        if isinstance(value, dict):
            flattened.update(flatten_dict(value, new_key))
        else:
            flattened[new_key] = value

    return flattened


def transform_game_stats(year):
    """
    Load raw CFBD game statistics and flatten 
    the nested offense and defense dictionaries
    """

    filepath = f"data/raw/game_stats/game_stats_{year}.csv"

    df = pd.read_csv(filepath)

    # Convert strings back into dictionaries
    df["offense"] = df["offense"].apply(ast.literal_eval)
    df["defense"] = df["defense"].apply(ast.literal_eval)

    # Flatten offense and defense
    offense = df["offense"].apply(
        lambda x: flatten_dict(x, "offense")
    ).apply(pd.Series)

    defense = df["defense"].apply(
        lambda x: flatten_dict(x, "defense")
    ).apply(pd.Series)

    # Remove original dictionary columns
    df = df.drop(columns = ["offense", "defense"])

    # Combine everything
    df = pd.concat([df, offense, defense], axis = 1)

    # Save flattened data frame
    df.to_csv(f"data/processed/game_stats/game_stats_{year}.csv", index = False)

def create_pregame_stats(year):
    """
    Create pre-game cumulative statistics for each team
    
    Each team's statistics for a game are calculated using
    only that team's previous games.
    
    This method is used to prevent data leakage
    """

    filepath = f"data/processed/game_stats/game_stats_{year}.csv"

    df = pd.read_csv(filepath)

    # Load game dates so advanced statistics can be ordered
    # chronologically rather than relying on CFBD week numbers
    games_filepath = f"data/raw/games/games_{year}.csv"

    games = pd.read_csv(games_filepath)

    games = games[["id", "startDate"]].rename(
        columns = {"id": "gameId"}
    )

    df = df.merge(games,
                  on = "gameId",
                  how = "left",
                  validate = "many_to_one")

    df["startDate"] = pd.to_datetime(df["startDate"])

    # Features calculated from games played prior to the current game.

    pregame_features = [
        # Overall offense
        "offense_plays",
        "offense_drives",
        "offense_ppa",
        "offense_totalPPA",
        "offense_successRate",
        "offense_explosiveness",
        "offense_powerSuccess",
        "offense_stuffRate",
        "offense_lineYards",
        "offense_lineYardsTotal",
        "offense_secondLevelYards",
        "offense_secondLevelYardsTotal",
        "offense_openFieldYards",
        "offense_openFieldYardsTotal",

        # Offensive situation
        "offense_standardDowns_ppa",
        "offense_standardDowns_successRate",
        "offense_standardDowns_explosiveness",
        "offense_passingDowns_ppa",
        "offense_passingDowns_successRate",
        "offense_passingDowns_explosiveness",

        # Rushing offense
        "offense_rushingPlays_ppa",
        "offense_rushingPlays_totalPPA",
        "offense_rushingPlays_successRate",
        "offense_rushingPlays_explosiveness",

        # Passing offense
        "offense_passingPlays_ppa",
        "offense_passingPlays_totalPPA",
        "offense_passingPlays_successRate",
        "offense_passingPlays_explosiveness",

        # Overall defense
        "defense_plays",
        "defense_drives",
        "defense_ppa",
        "defense_totalPPA",
        "defense_successRate",
        "defense_explosiveness",
        "defense_powerSuccess",
        "defense_stuffRate",
        "defense_lineYards",
        "defense_lineYardsTotal",
        "defense_secondLevelYards",
        "defense_secondLevelYardsTotal",
        "defense_openFieldYards",
        "defense_openFieldYardsTotal",

        # Defensive situation
        "defense_standardDowns_ppa",
        "defense_standardDowns_successRate",
        "defense_standardDowns_explosiveness",
        "defense_passingDowns_ppa",
        "defense_passingDowns_successRate",
        "defense_passingDowns_explosiveness",

        # Rushing defense
        "defense_rushingPlays_ppa",
        "defense_rushingPlays_totalPPA",
        "defense_rushingPlays_successRate",
        "defense_rushingPlays_explosiveness",

        # Passing defense
        "defense_passingPlays_ppa",
        "defense_passingPlays_totalPPA",
        "defense_passingPlays_successRate",
        "defense_passingPlays_explosiveness"
    ]

    # Sort chronologically within each team
    df = df.sort_values(
        ["team", "startDate", "gameId"]
    ).reset_index(drop = True)

    # Create pre-game averages
    for feature in pregame_features:
        df[f"pregame_{feature}"] = (
            df.groupby(["season", "team"])[feature].transform(
                lambda x: x.expanding().mean().shift(1)
            )
        )

    return df

if __name__ == "__main__":
    for year in range(2015, 2025):
        transform_game_stats(year)

        stats = create_pregame_stats(year)

        stats.to_csv(f"data/processed/game_stats/pregame_stats_{year}.csv", index = False)
        
        print(stats.shape)
        print(stats.head(20))