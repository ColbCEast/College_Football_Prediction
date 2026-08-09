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

    return df


if __name__ == "__main__":

    stats = transform_game_stats(2025)

    stats.to_csv(
        "data/processed/game_stats/game_stats_2025.csv",
        index = False
    )

    print(stats.shape)
    print(stats.head())