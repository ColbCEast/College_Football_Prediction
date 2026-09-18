"""
College Football Recruiting Data Audit
=======================================

Audits raw CFBD recruiting data across seasons 2015-2025.

Input:
    data/raw/player/recruiting/player_recruiting_{year}.csv

Checks:
    1. File existence
    2. Row counts by season
    3. Column consistency
    4. Data types
    5. recruitType distribution
    6. Duplicate rows
    7. Duplicate non-null athlete IDs
    8. Commitment rate
    9. Unique committed teams
    10. Missing ranking/stars/rating
    11. Rating distributions
    12. Stars distributions
    13. Position distributions
    14. Basic value/range checks
    15. Cross-season structural consistency

This script validates raw data only.
It does not transform or modify any files.
"""

from pathlib import Path

import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parents[4]

RECRUITING_DIR = ROOT / "data" / "raw" / "player" / "recruiting"

START_YEAR = 2015
END_YEAR = 2025
YEARS = range(START_YEAR, END_YEAR + 1)


# =============================================================================
# EXPECTED SCHEMA
# =============================================================================

EXPECTED_COLUMNS = [
    "id",
    "athleteId",
    "recruitType",
    "year",
    "ranking",
    "name",
    "school",
    "committedTo",
    "position",
    "height",
    "weight",
    "stars",
    "rating",
    "city",
    "stateProvince",
    "country",
    "hometownInfo",
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_section(title):
    """Print a formatted audit section header."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_subsection(title):
    """Print a formatted subsection header."""
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


# =============================================================================
# LOAD DATA
# =============================================================================

print("=" * 80)
print("COLLEGE FOOTBALL RECRUITING DATA AUDIT")
print("=" * 80)

print(f"\nProject root:")
print(f"  {ROOT}")

print(f"\nRecruiting directory:")
print(f"  {RECRUITING_DIR}")

print(f"\nAudit period:")
print(f"  {START_YEAR}-{END_YEAR}")


datasets = {}
missing_files = []

for year in YEARS:
    filepath = RECRUITING_DIR / f"player_recruiting_{year}.csv"

    if not filepath.exists():
        missing_files.append(year)
        continue

    df = pd.read_csv(filepath)
    datasets[year] = df


# =============================================================================
# 1. FILE AVAILABILITY
# =============================================================================

print_section("1. FILE AVAILABILITY")

print(f"Expected files: {len(list(YEARS))}")
print(f"Files found:    {len(datasets)}")
print(f"Files missing:  {len(missing_files)}")

if missing_files:
    print(f"\nMissing seasons:")
    for year in missing_files:
        print(f"  {year}")
else:
    print("\nPASS: All expected recruiting files were found.")


# Stop if files are missing because the remaining audit would be incomplete.
if missing_files:
    print("\nAUDIT STATUS: FAIL")
    print("Cannot complete the full multi-season audit.")
    raise SystemExit(1)


# =============================================================================
# 2. ROW COUNTS BY SEASON
# =============================================================================

print_section("2. ROW COUNTS BY SEASON")

row_counts = []

for year, df in datasets.items():
    row_counts.append({
        "season": year,
        "rows": len(df),
        "columns": len(df.columns),
    })

row_counts_df = pd.DataFrame(row_counts)

print(row_counts_df.to_string(index=False))

print(
    f"\nTotal recruiting records: "
    f"{row_counts_df['rows'].sum():,}"
)


# =============================================================================
# 3. COLUMN CONSISTENCY
# =============================================================================

print_section("3. COLUMN CONSISTENCY")

reference_columns = list(datasets[START_YEAR].columns)

print(f"Reference season: {START_YEAR}")
print(f"Reference columns: {len(reference_columns)}")

schema_pass = True

for year, df in datasets.items():

    columns = list(df.columns)

    missing_columns = [
        col for col in reference_columns
        if col not in columns
    ]

    extra_columns = [
        col for col in columns
        if col not in reference_columns
    ]

    if missing_columns or extra_columns:
        schema_pass = False

        print_subsection(f"{year}")

        if missing_columns:
            print("Missing columns:")
            for col in missing_columns:
                print(f"  - {col}")

        if extra_columns:
            print("Extra columns:")
            for col in extra_columns:
                print(f"  + {col}")


if schema_pass:
    print("\nPASS: Column structure is consistent across all seasons.")


# Compare against explicitly expected schema.
expected_schema_pass = True

for year, df in datasets.items():

    missing_expected = [
        col for col in EXPECTED_COLUMNS
        if col not in df.columns
    ]

    extra_columns = [
        col for col in df.columns
        if col not in EXPECTED_COLUMNS
    ]

    if missing_expected or extra_columns:
        expected_schema_pass = False

        print_subsection(f"{year} vs expected schema")

        if missing_expected:
            print("Missing expected columns:")
            for col in missing_expected:
                print(f"  - {col}")

        if extra_columns:
            print("Unexpected columns:")
            for col in extra_columns:
                print(f"  + {col}")

if expected_schema_pass:
    print("PASS: All seasons match the expected CFBD recruiting schema.")


# =============================================================================
# 4. DATA TYPES
# =============================================================================

print_section("4. DATA TYPES")

dtype_reference = datasets[START_YEAR].dtypes.astype(str)

dtype_pass = True

for year, df in datasets.items():

    current_dtypes = df.dtypes.astype(str)

    differences = {}

    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            continue

        if current_dtypes[col] != dtype_reference[col]:
            differences[col] = (
                dtype_reference[col],
                current_dtypes[col]
            )

    if differences:
        dtype_pass = False

        print_subsection(f"{year}")

        for col, (reference, current) in differences.items():
            print(
                f"{col}: "
                f"{reference} -> {current}"
            )

if dtype_pass:
    print(
        "PASS: No unexpected dtype differences relative to "
        f"{START_YEAR}."
    )
else:
    print(
        "\nNOTE: Some dtype differences exist. "
        "These may be caused by missing values and are not "
        "necessarily data-quality problems."
    )


# =============================================================================
# 5. RECRUIT TYPE DISTRIBUTION
# =============================================================================

print_section("5. RECRUIT TYPE DISTRIBUTION")

recruit_type_rows = []

for year, df in datasets.items():

    counts = (
        df["recruitType"]
        .value_counts(dropna=False)
        .sort_index()
    )

    for recruit_type, count in counts.items():
        recruit_type_rows.append({
            "season": year,
            "recruitType": recruit_type,
            "count": count,
            "percent": count / len(df) * 100,
        })

recruit_type_df = pd.DataFrame(recruit_type_rows)

print(
    recruit_type_df.to_string(
        index=False,
        formatters={"percent": "{:.2f}".format}
    )
)

unique_recruit_types = set(
    recruit_type_df["recruitType"].dropna().unique()
)

print(f"\nRecruit types observed: {sorted(unique_recruit_types)}")


# =============================================================================
# 6. DUPLICATE ROWS
# =============================================================================

print_section("6. DUPLICATE ROWS")

duplicate_pass = True

for year, df in datasets.items():

    duplicates = df.duplicated().sum()

    print(f"{year}: {duplicates:,}")

    if duplicates > 0:
        duplicate_pass = False


if duplicate_pass:
    print("\nPASS: No duplicate rows found.")


# =============================================================================
# 7. DUPLICATE ATHLETE IDS
# =============================================================================

print_section("7. DUPLICATE NON-MISSING ATHLETE IDS")

athlete_id_results = []

athlete_id_pass = True

for year, df in datasets.items():

    athlete_ids = df["athleteId"].dropna()

    duplicate_ids = athlete_ids.duplicated().sum()

    athlete_id_results.append({
        "season": year,
        "non_missing": len(athlete_ids),
        "unique": athlete_ids.nunique(),
        "duplicate_ids": duplicate_ids,
    })

    if duplicate_ids > 0:
        athlete_id_pass = False

athlete_id_df = pd.DataFrame(athlete_id_results)

print(athlete_id_df.to_string(index=False))

if athlete_id_pass:
    print(
        "\nPASS: No duplicate non-null athlete IDs found."
    )
else:
    print(
        "\nWARNING: Duplicate non-null athlete IDs were found."
    )


# =============================================================================
# 8. COMMITMENT RATE
# =============================================================================

print_section("8. COMMITMENT RATE")

commitment_results = []

for year, df in datasets.items():

    committed = df["committedTo"].notna().sum()
    uncommitted = df["committedTo"].isna().sum()

    commitment_results.append({
        "season": year,
        "total": len(df),
        "committed": committed,
        "uncommitted": uncommitted,
        "commitment_pct": committed / len(df) * 100,
    })

commitment_df = pd.DataFrame(commitment_results)

print(
    commitment_df.to_string(
        index=False,
        formatters={
            "commitment_pct": "{:.2f}".format
        }
    )
)


# =============================================================================
# 9. UNIQUE COMMITTED TEAMS
# =============================================================================

print_section("9. UNIQUE COMMITTED TEAMS")

team_results = []

for year, df in datasets.items():

    committed_teams = (
        df.loc[df["committedTo"].notna(), "committedTo"]
        .nunique()
    )

    team_results.append({
        "season": year,
        "unique_committed_teams": committed_teams,
    })

team_df = pd.DataFrame(team_results)

print(team_df.to_string(index=False))


# =============================================================================
# 10. MISSING CORE RECRUITING METRICS
# =============================================================================

print_section("10. MISSING CORE RECRUITING METRICS")

missing_results = []

for year, df in datasets.items():

    total = len(df)

    missing_results.append({
        "season": year,
        "ranking_missing": df["ranking"].isna().sum(),
        "ranking_pct": df["ranking"].isna().mean() * 100,
        "stars_missing": df["stars"].isna().sum(),
        "stars_pct": df["stars"].isna().mean() * 100,
        "rating_missing": df["rating"].isna().sum(),
        "rating_pct": df["rating"].isna().mean() * 100,
        "committed_missing": df["committedTo"].isna().sum(),
        "position_missing": df["position"].isna().sum(),
    })

missing_df = pd.DataFrame(missing_results)

print(
    missing_df.to_string(
        index=False,
        formatters={
            "ranking_pct": "{:.2f}".format,
            "stars_pct": "{:.2f}".format,
            "rating_pct": "{:.2f}".format,
        }
    )
)


# =============================================================================
# 11. CORE METRIC COMPLETENESS PATTERNS
# =============================================================================

print_section("11. CORE METRIC COMPLETENESS PATTERNS")

pattern_results = []

for year, df in datasets.items():

    metrics = ["ranking", "stars", "rating"]

    all_present = df[metrics].notna().all(axis=1).sum()
    all_missing = df[metrics].isna().all(axis=1).sum()
    partially_missing = (
        df[metrics].notna().any(axis=1)
        & ~df[metrics].notna().all(axis=1)
    ).sum()

    pattern_results.append({
        "season": year,
        "all_present": all_present,
        "partially_missing": partially_missing,
        "all_missing": all_missing,
    })

pattern_df = pd.DataFrame(pattern_results)

print(pattern_df.to_string(index=False))


# =============================================================================
# 12. RATING DISTRIBUTIONS
# =============================================================================

print_section("12. RATING DISTRIBUTIONS")

rating_results = []

for year, df in datasets.items():

    rating = df["rating"].dropna()

    if len(rating) == 0:
        continue

    rating_results.append({
        "season": year,
        "count": len(rating),
        "mean": rating.mean(),
        "std": rating.std(),
        "min": rating.min(),
        "p25": rating.quantile(0.25),
        "median": rating.median(),
        "p75": rating.quantile(0.75),
        "p90": rating.quantile(0.90),
        "p95": rating.quantile(0.95),
        "max": rating.max(),
    })

rating_df = pd.DataFrame(rating_results)

print(
    rating_df.to_string(
        index=False,
        formatters={
            "mean": "{:.4f}".format,
            "std": "{:.4f}".format,
            "min": "{:.4f}".format,
            "p25": "{:.4f}".format,
            "median": "{:.4f}".format,
            "p75": "{:.4f}".format,
            "p90": "{:.4f}".format,
            "p95": "{:.4f}".format,
            "max": "{:.4f}".format,
        }
    )
)


# =============================================================================
# 13. RATING RANGE CHECK
# =============================================================================

print_section("13. RATING RANGE CHECK")

rating_range_pass = True

for year, df in datasets.items():

    rating = df["rating"].dropna()

    if rating.empty:
        continue

    below_zero = (rating < 0).sum()
    above_one = (rating > 1).sum()

    print(
        f"{year}: "
        f"min={rating.min():.4f}, "
        f"max={rating.max():.4f}, "
        f"<0={below_zero:,}, "
        f">1={above_one:,}"
    )

    if below_zero > 0 or above_one > 0:
        rating_range_pass = False

if rating_range_pass:
    print("\nPASS: All non-missing ratings are between 0 and 1.")


# =============================================================================
# 14. STARS DISTRIBUTION
# =============================================================================

print_section("14. STARS DISTRIBUTION")

stars_results = []

for year, df in datasets.items():

    counts = (
        df["stars"]
        .value_counts(dropna=False)
        .sort_index()
    )

    for stars, count in counts.items():

        stars_label = (
            "Missing"
            if pd.isna(stars)
            else str(int(stars))
        )

        stars_results.append({
            "season": year,
            "stars": stars_label,
            "count": count,
            "percent": count / len(df) * 100,
        })

stars_df = pd.DataFrame(stars_results)

print(
    stars_df.to_string(
        index=False,
        formatters={"percent": "{:.2f}".format}
    )
)


# =============================================================================
# 15. STARS RANGE CHECK
# =============================================================================

print_section("15. STARS RANGE CHECK")

stars_range_pass = True

for year, df in datasets.items():

    stars = df["stars"].dropna()

    if stars.empty:
        continue

    invalid = ((stars < 0) | (stars > 5)).sum()

    print(
        f"{year}: "
        f"min={stars.min():.0f}, "
        f"max={stars.max():.0f}, "
        f"invalid={invalid:,}"
    )

    if invalid > 0:
        stars_range_pass = False

if stars_range_pass:
    print("\nPASS: All non-missing star ratings are within 0-5.")


# =============================================================================
# 16. POSITION DISTRIBUTION
# =============================================================================

print_section("16. POSITION DISTRIBUTION")

position_rows = []

for year, df in datasets.items():

    counts = (
        df["position"]
        .value_counts(dropna=False)
    )

    for position, count in counts.items():

        position_rows.append({
            "season": year,
            "position": position,
            "count": count,
        })

position_df = pd.DataFrame(position_rows)

position_pivot = (
    position_df
    .pivot(
        index="position",
        columns="season",
        values="count"
    )
    .fillna(0)
    .astype(int)
)

print(position_pivot.to_string())


# =============================================================================
# 17. MISSING POSITION CHECK
# =============================================================================

print_section("17. MISSING POSITION CHECK")

position_missing_pass = True

for year, df in datasets.items():

    missing = df["position"].isna().sum()

    print(f"{year}: {missing:,}")

    if missing > 0:
        position_missing_pass = False

if position_missing_pass:
    print("\nPASS: No missing positions found.")


# =============================================================================
# 18. YEAR FIELD CONSISTENCY
# =============================================================================

print_section("18. YEAR FIELD CONSISTENCY")

year_field_pass = True

for expected_year, df in datasets.items():

    mismatched = (df["year"] != expected_year).sum()

    print(
        f"{expected_year}: "
        f"{mismatched:,} mismatched records"
    )

    if mismatched > 0:
        year_field_pass = False

if year_field_pass:
    print("\nPASS: Every file's year field matches its filename.")


# =============================================================================
# 19. ID COMPLETENESS
# =============================================================================

print_section("19. ID COMPLETENESS")

id_pass = True

for year, df in datasets.items():

    missing_id = df["id"].isna().sum()

    print(
        f"{year}: "
        f"{missing_id:,} missing IDs"
    )

    if missing_id > 0:
        id_pass = False

if id_pass:
    print("\nPASS: No missing recruiting IDs found.")


# =============================================================================
# 20. CROSS-SEASON SUMMARY
# =============================================================================

print_section("20. CROSS-SEASON SUMMARY")

summary_rows = []

for year, df in datasets.items():

    athlete_ids = df["athleteId"].dropna()
    rating = df["rating"].dropna()

    summary_rows.append({
        "season": year,
        "rows": len(df),
        "columns": len(df.columns),
        "committed_pct": df["committedTo"].notna().mean() * 100,
        "unique_teams": df["committedTo"].nunique(),
        "rating_missing_pct": df["rating"].isna().mean() * 100,
        "stars_missing_pct": df["stars"].isna().mean() * 100,
        "ranking_missing_pct": df["ranking"].isna().mean() * 100,
        "athlete_id_missing_pct": df["athleteId"].isna().mean() * 100,
        "unique_athlete_ids": athlete_ids.nunique(),
        "rating_mean": rating.mean() if len(rating) else None,
        "rating_median": rating.median() if len(rating) else None,
        "duplicate_rows": df.duplicated().sum(),
        "duplicate_athlete_ids": athlete_ids.duplicated().sum(),
    })

summary_df = pd.DataFrame(summary_rows)

print(
    summary_df.to_string(
        index=False,
        formatters={
            "committed_pct": "{:.2f}".format,
            "rating_missing_pct": "{:.2f}".format,
            "stars_missing_pct": "{:.2f}".format,
            "ranking_missing_pct": "{:.2f}".format,
            "athlete_id_missing_pct": "{:.2f}".format,
            "rating_mean": "{:.4f}".format,
            "rating_median": "{:.4f}".format,
        }
    )
)


# =============================================================================
# 21. AUDIT STATUS
# =============================================================================

print_section("21. FINAL AUDIT STATUS")

checks = {
    "All files present": len(missing_files) == 0,
    "Column structure consistent": schema_pass,
    "Expected schema present": expected_schema_pass,
    "No duplicate rows": duplicate_pass,
    "No duplicate non-null athlete IDs": athlete_id_pass,
    "Ratings within 0-1": rating_range_pass,
    "Stars within 0-5": stars_range_pass,
    "No missing positions": position_missing_pass,
    "Year fields consistent": year_field_pass,
    "No missing recruiting IDs": id_pass,
}

all_pass = True

for check_name, passed in checks.items():

    status = "PASS" if passed else "FAIL"

    print(f"{status:<6} {check_name}")

    if not passed:
        all_pass = False


print("\n" + "=" * 80)

if all_pass:
    print("OVERALL AUDIT STATUS: PASS")
    print("=" * 80)
    print(
        "\nThe 2015-2025 raw recruiting datasets are structurally "
        "consistent and ready for transformation."
    )
else:
    print("OVERALL AUDIT STATUS: REVIEW REQUIRED")
    print("=" * 80)
    print(
        "\nOne or more structural checks require review before "
        "building the recruiting transformation."
    )