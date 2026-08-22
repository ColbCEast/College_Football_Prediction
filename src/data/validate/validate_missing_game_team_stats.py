import requests
import pandas as pd

from src.data.collect.cfbd_api import BASE_URL, headers


def check_game_team_stats(year, week, game_id):

    url = f"{BASE_URL}/games/teams"

    params = {
        "year": year,
        "week": week
    }

    print("=" * 70)
    print("CHECKING CFBD GAME-TEAM-STATS API")
    print("=" * 70)

    print(f"\nYear:     {year}")
    print(f"Week:     {week}")
    print(f"Game ID:  {game_id}")

    response = requests.get(
        url,
        headers=headers,
        params=params
    )

    print("\nAPI request:")
    print(response.url)

    response.raise_for_status()

    data = response.json()

    print(f"\nGames returned by API: {len(data)}")

    matching_games = [
        game
        for game in data
        if game.get("id") == game_id
    ]

    print(
        f"Matching game IDs returned: "
        f"{len(matching_games)}"
    )

    if not matching_games:
        print("\nRESULT:")
        print("The CFBD /games/teams endpoint did NOT return this game.")

        return

    game = matching_games[0]

    print("\nGame found:")
    print(f"  ID:        {game.get('id')}")
    print(f"  Home team: {game.get('homeTeam')}")
    print(f"  Away team: {game.get('awayTeam')}")
    print(f"  Home away: {game.get('homeAway')}")
    print(f"  Teams:     {len(game.get('teams', []))}")

    print("\nTeam statistics:")

    for team in game.get("teams", []):

        print("\n-----------------------------------")

        print(f"Team:     {team.get('team')}")
        print(f"HomeAway: {team.get('homeAway')}")
        print(f"Points:   {team.get('points')}")

        stats = team.get("stats", [])

        print(f"Stats returned: {len(stats)}")

        if stats:
            print("First 10 statistics:")

            for stat in stats[:10]:
                print(
                    f"  {stat.get('category')}: "
                    f"{stat.get('stat')}"
                )


if __name__ == "__main__":

    check_game_team_stats(
        year=2016,
        week=6,
        game_id=400868914
    )