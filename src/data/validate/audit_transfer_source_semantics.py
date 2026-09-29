"""
Audit CFBD transfer source semantics.

This audit is intentionally read-only. It does NOT modify raw or processed data.

Checks:
    1. Inspect every raw transfer record with stars == 1.
    2. Inspect pre-2022 transfer records appearing in the 2023 source file.
    3. Reconcile unusually large transfer counts for selected teams/seasons
       against the raw source records and processed transfer data.

The purpose of this audit is to distinguish:
    - transformation / feature-generation problems
from:
    - unusual but legitimate source-data semantics.

This audit should be run before integrating transfer features into the
modeling pipeline.

Expected project structure:

    data/
        raw/
            player/
                portal/
                    player_transfer_{year}.csv
        processed/
            player/
                portal/
                    transfer_portal_{year}.csv

Run from the project root:

    python src/data/validate/audit_transfer_source_semantics.py
"""

from pathlib import Path
import sys

import pandas as pd


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_TRANSFER_DIR = PROJECT_ROOT / "data" / "raw" / "player" / "portal"
PROCESSED_TRANSFER_DIR = (
    PROJECT_ROOT / "data" / "processed" / "player" / "portal"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Source years to inspect.
SOURCE_YEARS = range(2015, 2026)

# Teams / seasons selected for high-count reconciliation.
TEAM_CASES = [
    ("Colorado", 2023),
    ("Colorado", 2024),
    ("Washington State", 2025),
    ("Tennessee", 2021),
]

# Columns we want to display when inspecting individual source records.
DISPLAY_COLUMNS = [
    "id",
    "player",
    "origin",
    "destination",
    "transferDate",
    "transfer_year",
    "rating",
    "stars",
    "is_withdrawn",
]


# =============================================================================
# HELPERS
# =============================================================================

def print_header(title: str) -> None:
    """Print a consistent section header."""
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_subheader(title: str) -> None:
    """Print a consistent subsection header."""
    print()
    print("-" * 80)
    print(title)
    print("-" * 80)


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Return the first matching column from candidates.

    Matching is case-insensitive and ignores surrounding whitespace.
    """
    normalized = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in normalized:
            return normalized[key]

    return None


def load_transfer_file(
    directory: Path,
    year: int,
) -> tuple[pd.DataFrame | None, Path | None]:
    """
    Load one yearly transfer file.

    Returns:
        (DataFrame, path)

    If the file does not exist, returns (None, None).
    """
    path = directory / f"player_transfer_{year}.csv"

    if not path.exists():
        return None, None

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"ERROR reading {path}: {exc}")
        return None, path

    return df, path


def load_processed_transfer_file(
    year: int,
) -> tuple[pd.DataFrame | None, Path | None]:
    """Load one processed yearly transfer file."""
    path = PROCESSED_TRANSFER_DIR / f"transfer_portal_{year}.csv"

    if not path.exists():
        return None, None

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"ERROR reading {path}: {exc}")
        return None, path

    return df, path


def normalize_string(series: pd.Series) -> pd.Series:
    """Normalize string values for comparisons."""
    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )


def numeric_series(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Convert a dataframe column to numeric values."""
    return pd.to_numeric(df[column], errors="coerce")


def safe_bool_count(
    df: pd.DataFrame,
    column: str,
    value: bool,
) -> int | None:
    """
    Count boolean-like values safely.

    Supports actual booleans plus common string representations.
    """
    if column not in df.columns:
        return None

    series = df[column]

    if pd.api.types.is_bool_dtype(series):
        return int((series == value).sum())

    normalized = (
        series
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if value:
        return int(normalized.isin(["true", "1", "yes"]).sum())

    return int(normalized.isin(["false", "0", "no"]).sum())


def display_records(
    df: pd.DataFrame,
    columns: list[str],
    sort_columns: list[str] | None = None,
) -> None:
    """Display selected columns that actually exist."""
    available = [
        column
        for column in columns
        if column in df.columns
    ]

    if not available:
        print("No requested display columns are available.")
        print(f"Available columns: {list(df.columns)}")
        return

    output = df[available].copy()

    if sort_columns:
        valid_sort_columns = [
            column
            for column in sort_columns
            if column in output.columns
        ]

        if valid_sort_columns:
            output = output.sort_values(valid_sort_columns)

    # Avoid pandas truncating useful diagnostic information.
    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_columns",
        None,
        "display.width",
        240,
        "display.max_colwidth",
        40,
    ):
        print(output.to_string(index=False))


# =============================================================================
# LOAD ALL RAW TRANSFER DATA
# =============================================================================

def load_all_raw_transfer_data() -> dict[int, pd.DataFrame]:
    """
    Load all available raw transfer files.

    Returns:
        Dictionary keyed by source year.
    """
    print_subheader("Loading raw transfer files")

    raw_data: dict[int, pd.DataFrame] = {}

    for year in SOURCE_YEARS:
        df, path = load_transfer_file(RAW_TRANSFER_DIR, year)

        if df is None:
            print(f"{year}: MISSING")
            continue

        raw_data[year] = df

        print(
            f"{year}: "
            f"{len(df):,} rows | "
            f"{len(df.columns):,} columns | "
            f"{path.name}"
        )

    if not raw_data:
        raise FileNotFoundError(
            f"No raw transfer files found in {RAW_TRANSFER_DIR}"
        )

    return raw_data


# =============================================================================
# 1. ALL STARS == 1 RECORDS
# =============================================================================

def audit_one_star_records(
    raw_data: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """
    Find and inspect every raw record with stars == 1.

    Returns a dataframe containing all matching records plus source_file.
    """
    print_header("1. ALL RAW RECORDS WITH STARS == 1")

    matches: list[pd.DataFrame] = []

    for source_year, df in raw_data.items():
        stars_column = find_column(
            df,
            ["stars", "star", "stars_rating"],
        )

        if stars_column is None:
            print(
                f"{source_year}: "
                "No stars column found."
            )
            continue

        stars = numeric_series(df, stars_column)

        subset = df.loc[stars == 1].copy()

        if subset.empty:
            continue

        subset["source_year"] = source_year
        subset["source_file"] = (
            f"player_transfer_{source_year}.csv"
        )

        matches.append(subset)

    if not matches:
        print("No raw records with stars == 1 were found.")
        return pd.DataFrame()

    result = pd.concat(
        matches,
        ignore_index=True,
    )

    print(f"Total stars == 1 records: {len(result):,}")

    print_subheader("Seven-star record details")

    # Build display columns dynamically because raw API files may differ
    # slightly across seasons.
    available_display_columns = [
        column
        for column in DISPLAY_COLUMNS
        if column in result.columns
    ]

    available_display_columns += [
        "source_year",
        "source_file",
    ]

    display_records(
        result,
        available_display_columns,
        sort_columns=["source_year"],
    )

    # -------------------------------------------------------------------------
    # Data-quality details
    # -------------------------------------------------------------------------

    print_subheader("Stars == 1 field completeness")

    completeness_columns = [
        column
        for column in DISPLAY_COLUMNS
        if column in result.columns
    ]

    completeness_rows = []

    for column in completeness_columns:
        missing = int(result[column].isna().sum())

        if result[column].dtype == "object":
            missing += int(
                (
                    result[column]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    == ""
                ).sum()
            )

        completeness_rows.append(
            {
                "field": column,
                "missing": missing,
                "total": len(result),
                "complete_pct": (
                    100 * (len(result) - missing) / len(result)
                ),
            }
        )

    completeness_df = pd.DataFrame(completeness_rows)

    with pd.option_context(
        "display.width",
        160,
    ):
        print(
            completeness_df.to_string(
                index=False,
                formatters={
                    "complete_pct": "{:.2f}%".format,
                },
            )
        )

    return result


# =============================================================================
# 2. PRE-2022 RECORDS IN 2023 FILE
# =============================================================================

def audit_pre_2022_records_in_2023(
    raw_data: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """
    Inspect records in the 2023 source file whose transfer year/date
    indicates a pre-2022 transfer.

    This is intentionally descriptive. It does not classify the records
    as valid or invalid.
    """
    print_header("2. PRE-2022 RECORDS APPEARING IN THE 2023 SOURCE FILE")

    if 2023 not in raw_data:
        print("2023 raw transfer file is unavailable.")
        return pd.DataFrame()

    df = raw_data[2023].copy()

    transfer_year_column = find_column(
        df,
        ["transfer_year", "transferYear"],
    )

    transfer_date_column = find_column(
        df,
        ["transferDate", "transfer_date"],
    )

    if transfer_year_column is None and transfer_date_column is None:
        print(
            "Could not identify either transfer year or transfer date "
            "in the 2023 raw file."
        )
        print(f"Available columns: {list(df.columns)}")
        return pd.DataFrame()

    # -------------------------------------------------------------------------
    # Derive a comparable transfer year.
    # -------------------------------------------------------------------------

    if transfer_year_column is not None:
        transfer_year = pd.to_numeric(
            df[transfer_year_column],
            errors="coerce",
        )
    else:
        transfer_year = pd.Series(
            pd.NaT,
            index=df.index,
        )

    if transfer_date_column is not None:
        parsed_dates = pd.to_datetime(
            df[transfer_date_column],
            errors="coerce",
            utc=True,
        )

        date_year = parsed_dates.dt.year

        # Use transfer_year when present; otherwise use transferDate year.
        effective_year = transfer_year.copy()

        effective_year = effective_year.fillna(date_year)
    else:
        effective_year = transfer_year

    historical_mask = (
        effective_year.notna()
        & (effective_year < 2022)
    )

    historical = df.loc[historical_mask].copy()

    historical["derived_transfer_year"] = (
        effective_year.loc[historical.index]
    )

    print(
        f"2023 source rows: {len(df):,}"
    )
    print(
        f"Pre-2022 rows identified: {len(historical):,}"
    )

    if historical.empty:
        print("No pre-2022 records were found.")
        return historical

    # -------------------------------------------------------------------------
    # Year distribution.
    # -------------------------------------------------------------------------

    print_subheader("Pre-2022 records by transfer year")

    year_counts = (
        historical["derived_transfer_year"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("transfer_year")
        .reset_index(name="count")
    )

    print(year_counts.to_string(index=False))

    # -------------------------------------------------------------------------
    # Specifically inspect 2020 and 2021.
    # -------------------------------------------------------------------------

    for year in [2020, 2021]:
        subset = historical[
            historical["derived_transfer_year"] == year
        ].copy()

        print_subheader(
            f"2023-file records with transfer year {year}"
        )

        print(f"Record count: {len(subset):,}")

        if subset.empty:
            print("None.")
            continue

        columns = [
            column
            for column in DISPLAY_COLUMNS
            if column in subset.columns
        ]

        columns += [
            "derived_transfer_year",
        ]

        display_records(
            subset,
            columns,
            sort_columns=[
                "destination",
                "origin",
            ],
        )

    # -------------------------------------------------------------------------
    # Inspect all pre-2022 records if there are only a handful.
    # -------------------------------------------------------------------------

    if len(historical) <= 100:
        print_subheader(
            "All pre-2022 records in the 2023 source file"
        )

        columns = [
            column
            for column in DISPLAY_COLUMNS
            if column in historical.columns
        ]

        columns += [
            "derived_transfer_year",
        ]

        display_records(
            historical,
            columns,
            sort_columns=[
                "derived_transfer_year",
                "destination",
                "origin",
            ],
        )
    else:
        print(
            "More than 100 pre-2022 records were found; "
            "the complete record-level listing is omitted."
        )

    return historical


# =============================================================================
# TEAM COUNT HELPERS
# =============================================================================

def identify_team_column(
    df: pd.DataFrame,
) -> str | None:
    """Identify the destination/team column."""
    return find_column(
        df,
        [
            "destination",
            "destinationTeam",
            "destination_team",
            "team",
        ],
    )


def identify_withdrawn_column(
    df: pd.DataFrame,
) -> str | None:
    """Identify the withdrawal-status column."""
    return find_column(
        df,
        [
            "is_withdrawn",
            "isWithdrawn",
            "withdrawn",
        ],
    )


def identify_transfer_year_column(
    df: pd.DataFrame,
) -> str | None:
    """Identify transfer year column."""
    return find_column(
        df,
        [
            "transfer_year",
            "transferYear",
        ],
    )


def identify_destination_flag_column(
    df: pd.DataFrame,
) -> str | None:
    """Identify the processed destination flag."""
    return find_column(
        df,
        [
            "has_destination",
            "hasDestination",
        ],
    )


def identify_countable_destination_column(
    df: pd.DataFrame,
) -> str | None:
    """Identify the processed non-withdrawn destination flag."""
    return find_column(
        df,
        [
            "is_nonwithdrawn_destination",
            "isNonwithdrawnDestination",
        ],
    )


def count_team_rows(
    df: pd.DataFrame,
    team: str,
) -> pd.DataFrame:
    """
    Return all rows associated with a destination team.

    Matching is exact after whitespace normalization.
    """
    destination_column = identify_team_column(df)

    if destination_column is None:
        return pd.DataFrame()

    destination = normalize_string(
        df[destination_column]
    )

    return df.loc[
        destination.str.casefold() == team.casefold()
    ].copy()


# =============================================================================
# 3. HIGH-COUNT TEAM RECONCILIATION
# =============================================================================

def reconcile_team_case(
    team: str,
    year: int,
    raw_data: dict[int, pd.DataFrame],
) -> dict:
    """
    Reconcile one team-season against raw and processed transfer data.

    The counts are deliberately presented separately so that we do not
    silently assume that every raw API row should become a feature count.
    """
    print_subheader(
        f"{team} — {year}"
    )

    raw_df = raw_data.get(year)

    if raw_df is None:
        print("Raw transfer file unavailable.")
        return {
            "team": team,
            "year": year,
            "raw_count": None,
            "processed_count": None,
        }

    raw_team = count_team_rows(
        raw_df,
        team,
    )

    print(
        f"Raw destination records: {len(raw_team):,}"
    )

    if raw_team.empty:
        print(
            f"No raw records with destination == '{team}'."
        )
        return {
            "team": team,
            "year": year,
            "raw_count": 0,
            "processed_count": None,
        }

    # -------------------------------------------------------------------------
    # Raw withdrawal status.
    # -------------------------------------------------------------------------

    withdrawn_column = identify_withdrawn_column(raw_team)

    if withdrawn_column is not None:
        withdrawn_normalized = (
            raw_team[withdrawn_column]
            .fillna(False)
            .astype(str)
            .str.strip()
            .str.lower()
        )

        is_withdrawn = withdrawn_normalized.isin(
            ["true", "1", "yes"]
        )

        print(
            f"Raw withdrawn records: "
            f"{int(is_withdrawn.sum()):,}"
        )
        print(
            f"Raw non-withdrawn records: "
            f"{int((~is_withdrawn).sum()):,}"
        )
    else:
        is_withdrawn = pd.Series(
            False,
            index=raw_team.index,
        )

        print(
            "Withdrawal column not available in raw data."
        )

    # -------------------------------------------------------------------------
    # Display all raw records for this team.
    # -------------------------------------------------------------------------

    print_subheader(
        f"Raw transfer records — {team} {year}"
    )

    raw_columns = [
        column
        for column in DISPLAY_COLUMNS
        if column in raw_team.columns
    ]

    if raw_columns:
        display_records(
            raw_team,
            raw_columns,
            sort_columns=[
                "transferDate",
                "player",
            ],
        )
    else:
        print(
            f"Available columns: {list(raw_team.columns)}"
        )

    # -------------------------------------------------------------------------
    # Processed reconciliation.
    # -------------------------------------------------------------------------

    processed_df, processed_path = (
        load_processed_transfer_file(year)
    )

    processed_count = None

    if processed_df is None:
        print()
        print(
            "Processed transfer file unavailable; "
            "cannot perform processed-count reconciliation."
        )
    else:
        print()
        print(
            f"Processed file: {processed_path.name}"
        )

        processed_team = count_team_rows(
            processed_df,
            team,
        )

        processed_count = len(processed_team)

        print(
            f"Processed destination records: "
            f"{processed_count:,}"
        )

        destination_flag = identify_destination_flag_column(
            processed_team
        )

        countable_flag = identify_countable_destination_column(
            processed_team
        )

        processed_withdrawn_column = (
            identify_withdrawn_column(processed_team)
        )

        if destination_flag is not None:
            destination_values = (
                processed_team[destination_flag]
                .fillna(False)
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(["true", "1", "yes"])
            )

            print(
                f"Processed has_destination=True: "
                f"{int(destination_values.sum()):,}"
            )

        if countable_flag is not None:
            countable_values = (
                processed_team[countable_flag]
                .fillna(False)
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(["true", "1", "yes"])
            )

            print(
                "Processed "
                "is_nonwithdrawn_destination=True: "
                f"{int(countable_values.sum()):,}"
            )

        if processed_withdrawn_column is not None:
            processed_withdrawn = (
                processed_team[processed_withdrawn_column]
                .fillna(False)
                .astype(str)
                .str.strip()
                .str.lower()
                .isin(["true", "1", "yes"])
            )

            print(
                f"Processed withdrawn records: "
                f"{int(processed_withdrawn.sum()):,}"
            )

        # Display processed rows when the case is manageable.
        if len(processed_team) <= 100:
            print_subheader(
                f"Processed transfer records — {team} {year}"
            )

            processed_columns = [
                column
                for column in [
                    "player",
                    "origin",
                    "destination",
                    "transferDate",
                    "transfer_year",
                    "rating",
                    "stars",
                    "is_withdrawn",
                    "has_destination",
                    "is_nonwithdrawn_destination",
                ]
                if column in processed_team.columns
            ]

            if processed_columns:
                display_records(
                    processed_team,
                    processed_columns,
                    sort_columns=[
                        "transferDate",
                        "player",
                    ],
                )

    # -------------------------------------------------------------------------
    # Count by origin.
    # -------------------------------------------------------------------------

    origin_column = find_column(
        raw_team,
        ["origin", "originTeam", "origin_team"],
    )

    if origin_column is not None:
        print_subheader(
            f"Raw records by origin — {team} {year}"
        )

        origin_counts = (
            normalize_string(raw_team[origin_column])
            .replace("", "<MISSING>")
            .value_counts()
            .rename_axis("origin")
            .reset_index(name="count")
        )

        print(
            origin_counts.to_string(index=False)
        )

    # -------------------------------------------------------------------------
    # Count by transfer year if available.
    # -------------------------------------------------------------------------

    transfer_year_column = identify_transfer_year_column(
        raw_team
    )

    if transfer_year_column is not None:
        print_subheader(
            f"Raw records by transfer year — {team} {year}"
        )

        years = pd.to_numeric(
            raw_team[transfer_year_column],
            errors="coerce",
        )

        year_counts = (
            years
            .value_counts(dropna=False)
            .sort_index()
            .rename_axis("transfer_year")
            .reset_index(name="count")
        )

        print(
            year_counts.to_string(index=False)
        )

    # -------------------------------------------------------------------------
    # Determine whether simple raw/non-withdrawn count agrees with processed.
    # -------------------------------------------------------------------------

    nonwithdrawn_raw_count = int(
        (~is_withdrawn).sum()
    )

    if processed_count is not None:
        if processed_count == nonwithdrawn_raw_count:
            reconciliation = "PASS"
        else:
            reconciliation = "REVIEW"

        print()
        print(
            "Raw non-withdrawn vs processed destination count: "
            f"{nonwithdrawn_raw_count:,} vs {processed_count:,} "
            f"-> {reconciliation}"
        )
    else:
        reconciliation = "NOT CHECKED"

    return {
        "team": team,
        "year": year,
        "raw_count": len(raw_team),
        "raw_nonwithdrawn_count": nonwithdrawn_raw_count,
        "processed_count": processed_count,
        "reconciliation": reconciliation,
    }


def audit_high_count_teams(
    raw_data: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Run reconciliation for each selected team-season."""
    print_header("3. HIGH-COUNT TEAM RECONCILIATION")

    results = []

    for team, year in TEAM_CASES:
        results.append(
            reconcile_team_case(
                team,
                year,
                raw_data,
            )
        )

    return pd.DataFrame(results)


# =============================================================================
# 4. SUMMARY
# =============================================================================

def print_summary(
    one_star_df: pd.DataFrame,
    pre_2022_df: pd.DataFrame,
    team_results: pd.DataFrame,
) -> None:
    """
    Print a concise summary.

    This section deliberately avoids automatically declaring unusual source
    records invalid. It reports what the audit established.
    """
    print_header("4. SOURCE-SEMANTICS AUDIT SUMMARY")

    summary_rows = []

    # -------------------------------------------------------------------------
    # Stars == 1
    # -------------------------------------------------------------------------

    if len(one_star_df) == 7:
        one_star_result = "PASS"
        one_star_detail = "Exactly 7 raw stars == 1 records inspected."
    elif len(one_star_df) > 0:
        one_star_result = "REVIEW"
        one_star_detail = (
            f"{len(one_star_df):,} raw stars == 1 records found."
        )
    else:
        one_star_result = "REVIEW"
        one_star_detail = "No stars == 1 records found."

    summary_rows.append(
        {
            "check": "All stars == 1 records",
            "result": one_star_result,
            "detail": one_star_detail,
        }
    )

    # -------------------------------------------------------------------------
    # Pre-2022 records
    # -------------------------------------------------------------------------

    if len(pre_2022_df) == 0:
        pre_2022_result = "PASS"
        pre_2022_detail = (
            "No pre-2022 records found in the 2023 source file."
        )
    else:
        pre_2022_result = "REVIEW"
        pre_2022_detail = (
            f"{len(pre_2022_df):,} pre-2022 records found; "
            "source semantics require interpretation."
        )

    summary_rows.append(
        {
            "check": "Pre-2022 records in 2023 file",
            "result": pre_2022_result,
            "detail": pre_2022_detail,
        }
    )

    # -------------------------------------------------------------------------
    # Team cases
    # -------------------------------------------------------------------------

    if not team_results.empty:
        for _, row in team_results.iterrows():
            result = row.get(
                "reconciliation",
                "NOT CHECKED",
            )

            summary_rows.append(
                {
                    "check": (
                        f"{row.get('team')} "
                        f"{int(row.get('year'))}"
                    ),
                    "result": result,
                    "detail": (
                        "Raw non-withdrawn destination count = "
                        f"{row.get('raw_nonwithdrawn_count')}; "
                        "processed destination count = "
                        f"{row.get('processed_count')}"
                    ),
                }
            )

    summary_df = pd.DataFrame(summary_rows)

    with pd.option_context(
        "display.max_colwidth",
        100,
        "display.width",
        180,
    ):
        print(
            summary_df.to_string(index=False)
        )

    print()
    print(
        "Interpretation:"
    )
    print(
        "  PASS     = the targeted structural/reconciliation check agrees."
    )
    print(
        "  REVIEW   = the audit found an unusual source pattern or mismatch "
        "that requires interpretation."
    )
    print(
        "  NOT CHECKED = required source data was unavailable."
    )
    print()
    print(
        "IMPORTANT: A REVIEW result for historical or unusually large "
        "source records does not by itself indicate bad data."
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Run the complete source-semantics audit."""
    print("=" * 80)
    print("TRANSFER SOURCE-SEMANTICS AUDIT")
    print("=" * 80)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Raw transfer directory: {RAW_TRANSFER_DIR}")
    print(
        f"Processed transfer directory: "
        f"{PROCESSED_TRANSFER_DIR}"
    )

    # -------------------------------------------------------------------------
    # Validate directories.
    # -------------------------------------------------------------------------

    if not RAW_TRANSFER_DIR.exists():
        print()
        print(
            f"ERROR: Raw transfer directory does not exist:\n"
            f"  {RAW_TRANSFER_DIR}"
        )
        sys.exit(1)

    if not PROCESSED_TRANSFER_DIR.exists():
        print()
        print(
            "WARNING: Processed transfer directory does not exist:\n"
            f"  {PROCESSED_TRANSFER_DIR}"
        )

    # -------------------------------------------------------------------------
    # Load raw data.
    # -------------------------------------------------------------------------

    raw_data = load_all_raw_transfer_data()

    # -------------------------------------------------------------------------
    # 1. Stars == 1.
    # -------------------------------------------------------------------------

    one_star_df = audit_one_star_records(
        raw_data
    )

    # -------------------------------------------------------------------------
    # 2. Pre-2022 records in 2023.
    # -------------------------------------------------------------------------

    pre_2022_df = audit_pre_2022_records_in_2023(
        raw_data
    )

    # -------------------------------------------------------------------------
    # 3. High-count teams.
    # -------------------------------------------------------------------------

    team_results = audit_high_count_teams(
        raw_data
    )

    # -------------------------------------------------------------------------
    # 4. Summary.
    # -------------------------------------------------------------------------

    print_summary(
        one_star_df,
        pre_2022_df,
        team_results,
    )

    print()
    print("=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()