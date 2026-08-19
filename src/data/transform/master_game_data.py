import os
import pandas as pd


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

START_YEAR = 2015
END_YEAR = 2025

INPUT_DIR = "data/processed/game_team_stats"
OUTPUT_DIR = "data/master"
OUTPUT_FILE = "master_game_data.csv"


# ---------------------------------------------------------
# Load season data
# ---------------------------------------------------------

def load_season_data(year):
    """
    Load the cleaned team-game dataset for a single season.

    Each game should have exactly two rows:
        - one home team row
        - one away team row
    """

    filepath = os.path.join(
        INPUT_DIR,
        f"game_team_stats_{year}.csv"
    )

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Could not find {filepath}"
        )

    df = pd.read_csv(filepath)

    unique_games = df["gameId"].nunique()

    print(
        f"Loaded {year}: "
        f"{len(df):,} rows, "
        f"{unique_games:,} unique games"
    )

    return df


# ---------------------------------------------------------
# Validate season data
# ---------------------------------------------------------

def validate_season_data(df, year):
    """
    Validate the cleaned team-game data for a season.

    Each game must have exactly:
        - 1 home row
        - 1 away row
        - 2 total rows
    """

    # -----------------------------------------------------
    # Check required columns
    # -----------------------------------------------------

    required_columns = [
        "gameId",
        "season",
        "homeTeam",
        "awayTeam",
        "homeAway"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{year} is missing required columns: "
            f"{missing_columns}"
        )

    # -----------------------------------------------------
    # Check season
    # -----------------------------------------------------

    seasons = df["season"].dropna().unique()

    if len(seasons) != 1 or seasons[0] != year:
        raise ValueError(
            f"Season mismatch in {year} data. "
            f"Found: {seasons}"
        )

    # -----------------------------------------------------
    # Check missing game IDs
    # -----------------------------------------------------

    if df["gameId"].isna().any():
        raise ValueError(
            f"{year} contains missing gameId values."
        )

    # -----------------------------------------------------
    # Check rows per game
    # -----------------------------------------------------

    game_row_counts = (
        df.groupby("gameId")
        .size()
    )

    games_with_one_row = (
        (game_row_counts == 1).sum()
    )

    games_with_two_rows = (
        (game_row_counts == 2).sum()
    )

    games_with_more_than_two_rows = (
        (game_row_counts > 2).sum()
    )

    print(
        f"  Games with 1 row: "
        f"{games_with_one_row:,}"
    )

    print(
        f"  Games with 2 rows: "
        f"{games_with_two_rows:,}"
    )

    print(
        f"  Games with >2 rows: "
        f"{games_with_more_than_two_rows:,}"
    )

    if games_with_one_row > 0:
        raise ValueError(
            f"{year} contains games with only one row."
        )

    if games_with_more_than_two_rows > 0:
        raise ValueError(
            f"{year} contains games with more than "
            f"two rows."
        )

    # -----------------------------------------------------
    # Check home/away designation
    # -----------------------------------------------------

    home_counts = (
        df.groupby("gameId")["homeAway"]
        .apply(lambda x: (x == "home").sum())
    )

    away_counts = (
        df.groupby("gameId")["homeAway"]
        .apply(lambda x: (x == "away").sum())
    )

    invalid_home = (
        home_counts != 1
    ).sum()

    invalid_away = (
        away_counts != 1
    ).sum()

    print(
        f"  Games with invalid home row count: "
        f"{invalid_home:,}"
    )

    print(
        f"  Games with invalid away row count: "
        f"{invalid_away:,}"
    )

    if invalid_home > 0 or invalid_away > 0:
        raise ValueError(
            f"{year} does not contain exactly one "
            f"home row and one away row for every game."
        )

    # -----------------------------------------------------
    # Check missing team information
    # -----------------------------------------------------

    if df["homeTeam"].isna().any():
        raise ValueError(
            f"{year} contains missing homeTeam values."
        )

    if df["awayTeam"].isna().any():
        raise ValueError(
            f"{year} contains missing awayTeam values."
        )

    # -----------------------------------------------------
    # Print validation summary
    # -----------------------------------------------------

    unique_games = df["gameId"].nunique()

    print(
        f"  Total rows: {len(df):,}"
    )

    print(
        f"  Unique games: {unique_games:,}"
    )

    print(
        f"  Home rows: "
        f"{(df['homeAway'] == 'home').sum():,}"
    )

    print(
        f"  Away rows: "
        f"{(df['homeAway'] == 'away').sum():,}"
    )

    print(f"  ✓ {year} validation passed")


# ---------------------------------------------------------
# Combine seasons
# ---------------------------------------------------------

def combine_seasons(
    start_year=START_YEAR,
    end_year=END_YEAR
):
    """
    Combine all cleaned team-game datasets into one
    master dataset.
    """

    all_seasons = []

    for year in range(start_year, end_year + 1):

        df = load_season_data(year)

        validate_season_data(
            df,
            year
        )

        all_seasons.append(df)

    master = pd.concat(
        all_seasons,
        ignore_index=True
    )

    return master


# ---------------------------------------------------------
# Validate master dataset
# ---------------------------------------------------------

def validate_master_data(master):
    """
    Perform validation checks on the combined master
    team-game dataset.

    The master dataset intentionally contains two rows
    per game: one home row and one away row.
    """

    print("\n" + "=" * 60)
    print("MASTER DATA VALIDATION")
    print("=" * 60)

    # -----------------------------------------------------
    # Shape
    # -----------------------------------------------------

    print(
        f"\nMaster shape: {master.shape}"
    )

    # -----------------------------------------------------
    # Season coverage
    # -----------------------------------------------------

    print("\nRows by season:")

    season_row_counts = (
        master
        .groupby("season")
        .size()
    )

    print(season_row_counts)

    print("\nUnique games by season:")

    season_game_counts = (
        master
        .groupby("season")["gameId"]
        .nunique()
    )

    print(season_game_counts)

    # -----------------------------------------------------
    # Check expected two rows per game
    # -----------------------------------------------------

    game_row_counts = (
        master.groupby("gameId")
        .size()
    )

    games_with_one_row = (
        (game_row_counts == 1).sum()
    )

    games_with_two_rows = (
        (game_row_counts == 2).sum()
    )

    games_with_more_than_two_rows = (
        (game_row_counts > 2).sum()
    )

    print(
        f"\nGames with 1 row: "
        f"{games_with_one_row:,}"
    )

    print(
        f"Games with 2 rows: "
        f"{games_with_two_rows:,}"
    )

    print(
        f"Games with >2 rows: "
        f"{games_with_more_than_two_rows:,}"
    )

    if games_with_one_row > 0:
        raise ValueError(
            "Master dataset contains games with only one row."
        )

    if games_with_more_than_two_rows > 0:
        raise ValueError(
            "Master dataset contains games with more than "
            "two rows."
        )

    # -----------------------------------------------------
    # Home / away validation
    # -----------------------------------------------------

    home_counts = (
        master.groupby("gameId")["homeAway"]
        .apply(lambda x: (x == "home").sum())
    )

    away_counts = (
        master.groupby("gameId")["homeAway"]
        .apply(lambda x: (x == "away").sum())
    )

    bad_home = (
        home_counts != 1
    ).sum()

    bad_away = (
        away_counts != 1
    ).sum()

    print(
        f"\nGames with invalid home row count: "
        f"{bad_home:,}"
    )

    print(
        f"Games with invalid away row count: "
        f"{bad_away:,}"
    )

    if bad_home > 0 or bad_away > 0:
        raise ValueError(
            "Some games do not have exactly one "
            "home row and one away row."
        )

    # -----------------------------------------------------
    # Missing values in key columns
    # -----------------------------------------------------

    key_columns = [
        "gameId",
        "season",
        "week",
        "homeTeam",
        "awayTeam",
        "homeAway"
    ]

    print("\nMissing values in key columns:")

    missing = (
        master[key_columns]
        .isna()
        .sum()
    )

    print(missing)

    if missing.sum() > 0:
        raise ValueError(
            "Master dataset contains missing values "
            "in key columns."
        )

    # -----------------------------------------------------
    # Check home/away team consistency
    # -----------------------------------------------------

    home_team_counts = (
        master
        .groupby("gameId")["homeTeam"]
        .nunique()
    )

    away_team_counts = (
        master
        .groupby("gameId")["awayTeam"]
        .nunique()
    )

    invalid_home_teams = (
        home_team_counts != 1
    ).sum()

    invalid_away_teams = (
        away_team_counts != 1
    ).sum()

    print(
        f"\nGames with invalid home team count: "
        f"{invalid_home_teams:,}"
    )

    print(
        f"Games with invalid away team count: "
        f"{invalid_away_teams:,}"
    )

    if (
        invalid_home_teams > 0
        or invalid_away_teams > 0
    ):
        raise ValueError(
            "Some games do not have exactly one "
            "home team and one away team."
        )

    # -----------------------------------------------------
    # Sort master dataset
    # -----------------------------------------------------

    master.sort_values(
        ["season", "week", "gameId"],
        inplace=True
    )

    master.reset_index(
        drop=True,
        inplace=True
    )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    total_rows = len(master)
    total_games = master["gameId"].nunique()

    print(
        f"\nTotal rows: {total_rows:,}"
    )

    print(
        f"Total unique games: {total_games:,}"
    )

    print(
        f"Expected rows from 2 rows/game: "
        f"{total_games * 2:,}"
    )

    if total_rows != total_games * 2:
        raise ValueError(
            "Master dataset does not contain exactly "
            "two rows per game."
        )

    print("\n✓ Master validation passed")


# ---------------------------------------------------------
# Save master dataset
# ---------------------------------------------------------

def save_master_data(master):
    """
    Save the master team-game dataset.
    """

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    filepath = os.path.join(
        OUTPUT_DIR,
        OUTPUT_FILE
    )

    master.to_csv(
        filepath,
        index=False
    )

    print(
        f"\nSaved master dataset to: {filepath}"
    )

    print(
        f"Final shape: {master.shape}"
    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    master_game_data = combine_seasons()

    validate_master_data(
        master_game_data
    )

    save_master_data(
        master_game_data
    )