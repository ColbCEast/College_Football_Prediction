import pandas as pd


# ============================================================
# Configuration
# ============================================================

YEARS = range(2015, 2026)

BASE_PATH = "data/processed/features/game_level/"


# ============================================================
# Column classification
# ============================================================

# Columns that describe the game itself.
# These are not predictive features, but are useful for
# identifying and organizing observations.

GAME_METADATA = [
    "gameId",
    "season",
    "week",
    "startDate",
    "homeTeam",
    "awayTeam",
]


# These are statistics that describe the game being predicted.
# They must NOT be used as predictor variables.

GAME_OUTCOME_COLUMNS = [
    "homePoints",
    "awayPoints",
    "homeWin",
    "awayWin",
    "homePointDifferential",
    "awayPointDifferential",
]


# These are examples of statistics that should clearly be
# excluded because they describe what happened during the game.
#
# We will also identify additional post-game columns
# programmatically below.

POSTGAME_KEYWORDS = [
    "points",
    "win",
    "pointDifferential",
]


# ============================================================
# Load data
# ============================================================

def load_game_features(year):

    path = f"{BASE_PATH}/game_features_{year}.csv"

    return pd.read_csv(path)


# ============================================================
# Basic validation
# ============================================================

def validate_basic_structure(df, year):

    print("\n" + "=" * 70)
    print(f"YEAR: {year}")
    print("=" * 70)

    print("Shape:", df.shape)

    print(
        "Unique games:",
        df["gameId"].nunique()
    )

    duplicate_games = (
        df["gameId"]
        .duplicated()
        .sum()
    )

    print(
        "Duplicate game IDs:",
        duplicate_games
    )

    missing_game_ids = df["gameId"].isna().sum()

    print(
        "Missing game IDs:",
        missing_game_ids
    )

    return duplicate_games == 0 and missing_game_ids == 0


# ============================================================
# Identify pregame columns
# ============================================================

def identify_pregame_columns(df):

    pregame_columns = []

    for column in df.columns:

        # Explicit pregame naming convention
        if "Before" in column:
            pregame_columns.append(column)

        # Rolling features created from previous games
        elif "Last3" in column:
            pregame_columns.append(column)

        elif "Last5" in column:
            pregame_columns.append(column)

    return sorted(pregame_columns)


# ============================================================
# Identify postgame columns
# ============================================================

def identify_postgame_columns(df):

    postgame_columns = []

    for column in df.columns:

        column_lower = column.lower()

        # Current-game statistics generally do not contain
        # "Before", "Last3", or "Last5".
        #
        # We flag columns that clearly represent game results.

        if column in GAME_OUTCOME_COLUMNS:
            postgame_columns.append(column)

            continue

        # Team-game statistics from the current game
        # frequently use these names.

        postgame_indicators = [
            "points_",
            "win_",
            "pointdifferential_",
        ]

        if any(
            indicator in column_lower
            for indicator in postgame_indicators
        ):

            if (
                "Before" not in column
                and "Last3" not in column
                and "Last5" not in column
            ):
                postgame_columns.append(column)

    return sorted(set(postgame_columns))


# ============================================================
# Validate pregame columns
# ============================================================

def validate_pregame_columns(df, pregame_columns):

    print("\nPregame feature validation:")

    print(
        "Number of pregame columns:",
        len(pregame_columns)
    )

    missing_values = (
        df[pregame_columns]
        .isna()
        .sum()
    )

    missing_columns = (
        missing_values[
            missing_values > 0
        ]
    )

    print(
        "Pregame columns with missing values:",
        len(missing_columns)
    )

    if len(missing_columns) > 0:

        print("\nMissing values:")

        for column, count in missing_columns.items():

            print(
                f"  {column}: {count}"
            )


# ============================================================
# Validate game outcomes
# ============================================================

def validate_outcomes(df):

    print("\nGame outcome validation:")

    for column in GAME_OUTCOME_COLUMNS:

        if column not in df.columns:

            print(
                f"  MISSING: {column}"
            )

        else:

            print(
                f"  {column}: present"
            )


# ============================================================
# Validate feature naming
# ============================================================

def validate_unexpected_columns(
    df,
    pregame_columns,
    postgame_columns
):

    known_columns = set(
        GAME_METADATA
        + GAME_OUTCOME_COLUMNS
        + pregame_columns
        + postgame_columns
    )

    unexpected_columns = sorted(
        set(df.columns) - known_columns
    )

    print("\nUnexpected columns:")

    if not unexpected_columns:

        print("  None")

    else:

        for column in unexpected_columns:

            print(
                f"  {column}"
            )

    return unexpected_columns


# ============================================================
# Compare schemas across seasons
# ============================================================

def compare_schemas(all_data):

    print("\n" + "=" * 70)
    print("SCHEMA COMPARISON")
    print("=" * 70)

    years = list(all_data.keys())

    reference_year = years[0]

    reference_columns = set(
        all_data[reference_year].columns
    )

    print(
        f"Reference year: {reference_year}"
    )

    schema_match = True

    for year in years[1:]:

        columns = set(
            all_data[year].columns
        )

        missing = sorted(
            reference_columns - columns
        )

        extra = sorted(
            columns - reference_columns
        )

        if missing or extra:

            schema_match = False

            print(
                f"\n{year} differs from {reference_year}:"
            )

            if missing:

                print("  Missing:")
                for column in missing:
                    print(f"    {column}")

            if extra:

                print("  Extra:")
                for column in extra:
                    print(f"    {column}")

    if schema_match:

        print(
            "\nAll years have identical column schemas."
        )

    return schema_match


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    all_data = {}

    overall_valid = True

    for year in YEARS:

        df = load_game_features(year)

        all_data[year] = df

        structure_valid = (
            validate_basic_structure(
                df,
                year
            )
        )

        if not structure_valid:

            overall_valid = False

        pregame_columns = (
            identify_pregame_columns(df)
        )

        postgame_columns = (
            identify_postgame_columns(df)
        )

        validate_pregame_columns(
            df,
            pregame_columns
        )

        validate_outcomes(df)

        unexpected_columns = (
            validate_unexpected_columns(
                df,
                pregame_columns,
                postgame_columns
            )
        )

        if unexpected_columns:

            overall_valid = False

        print(
            "\nPregame columns:",
            len(pregame_columns)
        )

        print(
            "Postgame columns:",
            len(postgame_columns)
        )

    schema_valid = compare_schemas(
        all_data
    )

    if not schema_valid:

        overall_valid = False

    print("\n" + "=" * 70)
    print("FINAL VALIDATION RESULT")
    print("=" * 70)

    if overall_valid:

        print(
            "PASS: Game feature datasets passed validation."
        )

    else:

        print(
            "REVIEW REQUIRED: One or more validation checks failed."
        )