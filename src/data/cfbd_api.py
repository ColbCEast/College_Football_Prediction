import os
import requests
import pandas as pd
from dotenv import load_dotenv
import json

load_dotenv()

API_KEY = os.getenv("CFBD_API_KEY")

BASE_URL = "https://api.collegefootballdata.com"

headers = {
    "Authorization": f"Bearer {API_KEY}"
}

def get_games(year):
    params = {
        "year": year,
        "seasonType": "regular"
    }

    response = requests.get(
        f"{BASE_URL}/games",
        headers = headers,
        params = params
    )

    response.raise_for_status()

    return pd.DataFrame(response.json())



def save_games(year):
    games = get_games(year)

    os.makedirs("data/raw/games", exist_ok = True)

    filepath = f"data/raw/games/games_{year}.csv"

    games.to_csv(filepath, index = False)

    print(f"Saved {year}: {len(games):,} games")



def get_team_stats(year):
    url = f"{BASE_URL}/stats/season"

    params = {
        "year": year
    }

    response = requests.get(
        url, 
        headers = headers,
        params = params)

    response.raise_for_status()

    return response.json()



def save_team_stats(year):
    stats = get_team_stats(year)

    df = pd.DataFrame(stats)

    df.to_csv(f"data/raw/stats/team_stats_{year}.csv", index=False)



def get_game_stats(year):
    url = f"{BASE_URL}/stats/game/advanced"

    params = {
        "year": year,
        "seasonType": "regular"
    }

    response = requests.get(
        url,
        headers=headers,
        params=params
    )

    response.raise_for_status()

    return response.json()



def save_game_stats(year):
    stats = get_game_stats(year)

    df = pd.DataFrame(stats)

    df.to_csv(
        f"data/raw/game_stats/game_stats_{year}.csv",
        index=False
    )

    print(f"Saved {year}: {len(stats):,} stats")


if __name__ == "__main__":
    save_game_stats(2025)