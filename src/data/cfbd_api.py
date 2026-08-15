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



def get_game_team_stats(year, week):
    url = f"{BASE_URL}/games/teams"

    params = {
        "year": year,
        "week": week,
        "seasonType": "regular"
    }

    response = requests.get(url,
                headers = headers,
                params = params)

    response.raise_for_status()

    data = response.json()

    rows = []

    for game in data:
        game_id = game.get("id")

        for team in game.get("teams", []):
            row = {
                "gameId": game_id,
                "season": year,
                "week": week,
                "seasonType": "regular",
                "team": team.get("team"),
                "conference": team.get("conference"),
                "homeAway": team.get("homeAway"),
                "points": team.get("points"),
            }

            for stat in team.get("stats", []):
                row[stat["category"]] = stat["stat"]

            rows.append(row)

    return pd.DataFrame(rows)

def get_season_game_team_stats(year, start_week = 1, end_week = 15):
    all_stats = []

    for week in range(start_week, end_week + 1):
        week_stats = get_game_team_stats(year, week)
        all_stats.append(week_stats)

    return pd.concat(all_stats, ignore_index = True)


if __name__ == "__main__":
    stats_2025 = get_season_game_team_stats(2025)

    print(stats_2025.shape)
    print(stats_2025["week"].value_counts().sort_index())
    print(stats_2025["gameId"].nunique())

    stats_2025.to_csv("data/raw/game_team_stats/game_team_stats_2025.csv", index = False)
