import pandas as pd
import glob
import os


def clean_games():
    files = glob.glob("data/raw/games/games_*.csv")

    all_games = []

    for file in files:
        games = pd.read_csv(file)

        # Retain games that involve one or more FBS team
        games = games[
            (games["homeClassification"] == "fbs") |
            (games["awayClassification"] == "fbs")
        ].copy()

        # Retain only completed games
        games = games[games["completed"] == True].copy()

        # Require a final score
        games = games[
            games["homePoints"].notna() &
            games["awayPoints"].notna()
        ].copy()

        all_games.append(games)

    master_games = pd.concat(all_games, ignore_index = True)

    # Sort chronologically
    master_games["startDate"] = pd.to_datetime(master_games["startDate"])

    master_games = master_games.sort_values("startDate").reset_index(drop = True)

    # Output directory
    os.makedirs("data/processed/games", exist_ok = True)

    # Save cleaned dataset
    output_path = "data/processed/games/fbs_games_2015_2025.csv"

    master_games.to_csv(output_path, index = False)

    print(f"Saved {len(master_games):,} games")
    print(f"Output: {output_path}")