import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

YEAR = 2025

PREGAME_STATS_PATH = Path(
    f"data/processed/game_stats/game_level/pregame_stats_{YEAR}.csv"
)

GAME_STATS_PATH = Path(
    f"data/processed/game_stats/game_level/game_stats_{YEAR}.csv"
)

GAMES_PATH = Path(
    f"data/raw/games/games_{YEAR}.csv"
)


# ============================================================
# Load data
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

pregame_df = pd.read_csv(PREGAME_STATS_PATH)
game_stats_df = pd.read_csv(GAME_STATS_PATH)
games_df = pd.read_csv(GAMES_PATH)

print(f"\nPregame stats shape: {pregame_df.shape}")
print(f"Game stats shape:    {game_stats_df.shape}")
print(f"Games shape:         {games_df.shape}")


# ============================================================
# Identify pregame columns
# ============================================================

pregame_columns = [
    col
    for col in pregame_df.columns
    if col.startswith("pregame_")
]

print(
    f"\nPregame feature columns: {len(pregame_columns)}"
)


# ============================================================
# 1. Required columns
# ============================================================

print("\n" + "=" * 70)
print("1. REQUIRED COLUMNS")
print("=" * 70)

required_columns = [
    "gameId",
    "season",
    "seasonType",
    "week",
    "team",
    "opponent",
    "startDate",
]

missing_required = [
    col
    for col in required_columns
    if col not in pregame_df.columns
]

if missing_required:

    print("✗ Missing required columns:")
    for col in missing_required:
        print(f"  {col}")

else:

    print("✓ All required columns exist")


# ============================================================
# 2. Duplicate game/team records
# ============================================================

print("\n" + "=" * 70)
print("2. DUPLICATE GAME/TEAM CHECK")
print("=" * 70)

duplicate_mask = pregame_df.duplicated(
    subset=["gameId", "team"],
    keep=False
)

duplicate_rows = pregame_df[duplicate_mask]

duplicate_keys = (
    pregame_df.loc[
        duplicate_mask,
        ["gameId", "team"]
    ]
    .drop_duplicates()
)

print(
    f"Duplicate rows:              "
    f"{len(duplicate_rows):,}"
)

print(
    f"Duplicate game/team pairs:   "
    f"{len(duplicate_keys):,}"
)

if len(duplicate_keys) == 0:

    print("✓ No duplicate game/team records")

else:

    print("✗ Duplicate game/team records found")

    print(
        duplicate_keys.head(20).to_string(
            index=False
        )
    )


# ============================================================
# 3. Missing dates
# ============================================================

print("\n" + "=" * 70)
print("3. DATE CHECK")
print("=" * 70)

pregame_df["startDate"] = pd.to_datetime(
    pregame_df["startDate"],
    errors="coerce"
)

missing_dates = pregame_df["startDate"].isna().sum()

print(
    f"Missing startDate values: "
    f"{missing_dates:,}"
)

if missing_dates == 0:

    print("✓ Every game/team row has a valid startDate")

else:

    print("✗ Missing or invalid startDate values found")


# ============================================================
# 4. Game ID → date validation
# ============================================================

print("\n" + "=" * 70)
print("4. GAME DATE MERGE VALIDATION")
print("=" * 70)

games_lookup = games_df[
    ["id", "startDate"]
].copy()

games_lookup = games_lookup.rename(
    columns={"id": "gameId"}
)

games_lookup["startDate"] = pd.to_datetime(
    games_lookup["startDate"],
    errors="coerce"
)

# Check for duplicate game IDs in the games dataset
duplicate_game_ids = (
    games_lookup["gameId"]
    .duplicated()
    .sum()
)

print(
    f"Duplicate game IDs in games data: "
    f"{duplicate_game_ids:,}"
)

# Merge the expected date
validation_df = pregame_df[
    ["gameId", "team", "startDate"]
].merge(
    games_lookup,
    on="gameId",
    how="left",
    suffixes=("_pregame", "_games"),
    validate="many_to_one"
)

missing_lookup_dates = (
    validation_df["startDate_games"]
    .isna()
    .sum()
)

date_mismatches = (
    validation_df["startDate_pregame"]
    != validation_df["startDate_games"]
).sum()

print(
    f"Game IDs without matching game date: "
    f"{missing_lookup_dates:,}"
)

print(
    f"Date mismatches: "
    f"{date_mismatches:,}"
)

if (
    missing_lookup_dates == 0
    and date_mismatches == 0
):

    print("✓ Game dates match source games data")

else:

    print("✗ Game date validation failed")


# ============================================================
# 5. Chronological ordering
# ============================================================

print("\n" + "=" * 70)
print("5. CHRONOLOGICAL ORDERING")
print("=" * 70)

ordered_df = pregame_df.sort_values(
    ["team", "startDate", "gameId"]
).reset_index(drop=True)

original_ids = pregame_df[
    ["gameId", "team"]
].reset_index(drop=True)

ordered_ids = ordered_df[
    ["gameId", "team"]
].reset_index(drop=True)

is_already_sorted = original_ids.equals(
    ordered_ids
)

if is_already_sorted:

    print(
        "✓ Data is already sorted by "
        "team → startDate → gameId"
    )

else:

    print(
        "⚠ Data is not stored in chronological order."
    )

    print(
        "This does not necessarily indicate leakage, "
        "but the ordering should be corrected."
    )


# ============================================================
# 6. Chronology errors within each team
# ============================================================

print("\n" + "=" * 70)
print("6. TEAM CHRONOLOGY CHECK")
print("=" * 70)

chronology_errors = []

for team, group in pregame_df.groupby("team"):

    dates = group["startDate"].values

    if not pd.Series(dates).is_monotonic_increasing:

        chronology_errors.append(team)


print(
    f"Teams with chronology errors: "
    f"{len(chronology_errors)}"
)

if len(chronology_errors) == 0:

    print("✓ All team games are chronologically ordered")

else:

    print("✗ Teams with chronology errors:")

    for team in chronology_errors[:20]:
        print(f"  {team}")


# ============================================================
# 7. First-game pregame values
# ============================================================

print("\n" + "=" * 70)
print("7. FIRST-GAME PREGAME VALUES")
print("=" * 70)

first_game_rows = (
    pregame_df
    .sort_values(
        ["team", "startDate", "gameId"]
    )
    .groupby("team", as_index=False)
    .nth(0)
    .reset_index(drop = True)
)

first_game_non_null = (
    first_game_rows[pregame_columns]
    .notna()
    .sum()
)

problem_columns = first_game_non_null[
    first_game_non_null > 0
]

print(
    f"Teams checked: "
    f"{len(first_game_rows):,}"
)

print(
    f"Pregame columns with first-game values: "
    f"{len(problem_columns)}"
)

if len(problem_columns) == 0:

    print(
        "✓ All first-game pregame statistics are NaN"
    )

else:

    print(
        "✗ Some first-game pregame statistics "
        "are not NaN:"
    )

    for col in problem_columns.index:
        print(
            f"  {col}: "
            f"{problem_columns[col]} non-null"
        )


# ============================================================
# 8. Second-game validation
# ============================================================

print("\n" + "=" * 70)
print("8. SECOND-GAME VALIDATION")
print("=" * 70)

print(
    "Checking that each team's second-game "
    "pregame value equals the first game's "
    "actual statistic."
)

second_game_errors = []

# Identify the source statistic for each pregame feature
for pregame_col in pregame_columns:

    source_col = pregame_col.replace(
        "pregame_",
        "",
        1
    )

    if source_col not in pregame_df.columns:
        continue

    for team, group in (
        pregame_df
        .sort_values(
            ["team", "startDate", "gameId"]
        )
        .groupby("team")
    ):

        if len(group) < 2:
            continue

        first = group.iloc[0]
        second = group.iloc[1]

        expected = first[source_col]
        actual = second[pregame_col]

        # Both missing is correct
        if pd.isna(expected) and pd.isna(actual):
            continue

        # One missing and one not missing is incorrect
        if pd.isna(expected) != pd.isna(actual):

            second_game_errors.append(
                {
                    "team": team,
                    "gameId": second["gameId"],
                    "feature": pregame_col,
                    "expected": expected,
                    "actual": actual,
                }
            )

            continue

        # Numerical comparison
        if not np.isclose(
            float(expected),
            float(actual),
            rtol=1e-9,
            atol=1e-9
        ):

            second_game_errors.append(
                {
                    "team": team,
                    "gameId": second["gameId"],
                    "feature": pregame_col,
                    "expected": expected,
                    "actual": actual,
                }
            )


print(
    f"Second-game mismatches: "
    f"{len(second_game_errors):,}"
)

if len(second_game_errors) == 0:

    print(
        "✓ Second-game pregame values are correct"
    )

else:

    print("✗ Second-game mismatches found")

    print(
        pd.DataFrame(second_game_errors)
        .head(20)
        .to_string(index=False)
    )


# ============================================================
# 9. Third-game cumulative average validation
# ============================================================

print("\n" + "=" * 70)
print("9. THIRD-GAME CUMULATIVE AVERAGE VALIDATION")
print("=" * 70)

print(
    "Checking that each team's third-game "
    "pregame value equals the mean of the "
    "first two games."
)

third_game_errors = []

for pregame_col in pregame_columns:

    source_col = pregame_col.replace(
        "pregame_",
        "",
        1
    )

    if source_col not in pregame_df.columns:
        continue

    for team, group in (
        pregame_df
        .sort_values(
            ["team", "startDate", "gameId"]
        )
        .groupby("team")
    ):

        if len(group) < 3:
            continue

        first_two = group.iloc[:2]
        third = group.iloc[2]

        expected = first_two[source_col].mean()
        actual = third[pregame_col]

        # If both are NaN, correct
        if pd.isna(expected) and pd.isna(actual):
            continue

        # Missing mismatch
        if pd.isna(expected) != pd.isna(actual):

            third_game_errors.append(
                {
                    "team": team,
                    "gameId": third["gameId"],
                    "feature": pregame_col,
                    "expected": expected,
                    "actual": actual,
                }
            )

            continue

        if not np.isclose(
            float(expected),
            float(actual),
            rtol=1e-9,
            atol=1e-9
        ):

            third_game_errors.append(
                {
                    "team": team,
                    "gameId": third["gameId"],
                    "feature": pregame_col,
                    "expected": expected,
                    "actual": actual,
                }
            )


print(
    f"Third-game mismatches: "
    f"{len(third_game_errors):,}"
)

if len(third_game_errors) == 0:

    print(
        "✓ Third-game cumulative averages are correct"
    )

else:

    print("✗ Third-game mismatches found")

    print(
        pd.DataFrame(third_game_errors)
        .head(20)
        .to_string(index=False)
    )


# ============================================================
# 10. Check that all pregame columns have the correct source
# ============================================================

print("\n" + "=" * 70)
print("10. PREGAME COLUMN SOURCE CHECK")
print("=" * 70)

missing_source_columns = []

for pregame_col in pregame_columns:

    source_col = pregame_col.replace(
        "pregame_",
        "",
        1
    )

    if source_col not in pregame_df.columns:

        missing_source_columns.append(
            (pregame_col, source_col)
        )


print(
    f"Pregame columns without source columns: "
    f"{len(missing_source_columns)}"
)

if len(missing_source_columns) == 0:

    print(
        "✓ Every pregame feature has a corresponding "
        "current-game source statistic"
    )

else:

    print("✗ Missing source columns:")

    for pregame_col, source_col in missing_source_columns:
        print(
            f"  {pregame_col} → {source_col}"
        )


# ============================================================
# 11. Missingness summary
# ============================================================

print("\n" + "=" * 70)
print("11. PREGAME FEATURE MISSINGNESS")
print("=" * 70)

missing_summary = (
    pregame_df[pregame_columns]
    .isna()
    .sum()
    .sort_values(ascending=False)
)

print(
    missing_summary.to_string()
)


# ============================================================
# Final summary
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)

checks = {
    "Required columns": len(missing_required) == 0,
    "Duplicate game/team records": len(duplicate_keys) == 0,
    "Missing dates": missing_dates == 0,
    "Game date merge": (
        missing_lookup_dates == 0
        and date_mismatches == 0
    ),
    "Team chronology": len(chronology_errors) == 0,
    "First-game NaNs": len(problem_columns) == 0,
    "Second-game values": len(second_game_errors) == 0,
    "Third-game averages": len(third_game_errors) == 0,
    "Pregame source columns": (
        len(missing_source_columns) == 0
    ),
}

print()

for check, passed in checks.items():

    if passed:
        print(f"✓ {check}")

    else:
        print(f"✗ {check}")


failed_checks = [
    check
    for check, passed in checks.items()
    if not passed
]

print()

if len(failed_checks) == 0:

    print(
        "✓ ALL VALIDATION CHECKS PASSED"
    )

    print(
        f"\nThe {YEAR} advanced pregame statistics "
        "appear to be correctly calculated without "
        "current-game leakage."
    )

else:

    print(
        f"✗ {len(failed_checks)} validation check(s) failed:"
    )

    for check in failed_checks:
        print(f"  - {check}")

    print(
        "\nDo not process all seasons until the "
        "failed checks have been investigated."
    )