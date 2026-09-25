import os
import pandas as pd


INPUT_DIR = "data/processed/player/portal"


def audit_transfer_portal(year):
    filepath = os.path.join(
        INPUT_DIR,
        f"transfer_portal_{year}.csv"
    )

    print("\n" + "=" * 80)
    print(f"TRANSFER PORTAL AUDIT - {year}")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------

    df = pd.read_csv(filepath)

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    # -------------------------------------------------------------------------
    # Expected columns
    # -------------------------------------------------------------------------

    expected_columns = [
        "season",
        "first_name",
        "last_name",
        "position",
        "origin",
        "destination",
        "transfer_date",
        "transfer_year",
        "eligibility",
        "is_withdrawn",
        "has_destination",
        "is_nonwithdrawn_destination",
        "rating",
        "has_rating",
        "stars",
        "has_stars",
    ]

    missing_columns = [
        col
        for col in expected_columns
        if col not in df.columns
    ]

    unexpected_columns = [
        col
        for col in df.columns
        if col not in expected_columns
    ]

    print("\nCOLUMN CHECK")
    print("-" * 80)

    if not missing_columns:
        print("Missing expected columns: 0")
    else:
        print(f"Missing expected columns: {missing_columns}")

    if not unexpected_columns:
        print("Unexpected columns: 0")
    else:
        print(f"Unexpected columns: {unexpected_columns}")

    # -------------------------------------------------------------------------
    # Missing values
    # -------------------------------------------------------------------------

    print("\nMISSING VALUES")
    print("-" * 80)

    missing = df.isna().sum()

    for column in expected_columns:
        count = missing[column]

        if count > 0:
            pct = count / len(df) * 100
            print(
                f"{column:<30} "
                f"{count:>6,} "
                f"({pct:>6.2f}%)"
            )
        else:
            print(
                f"{column:<30} "
                f"{0:>6,} "
                f"(  0.00%)"
            )

    # -------------------------------------------------------------------------
    # Date validation
    # -------------------------------------------------------------------------

    print("\nDATE VALIDATION")
    print("-" * 80)

    df["transfer_date"] = pd.to_datetime(
        df["transfer_date"],
        errors="coerce",
        utc=True
    )

    invalid_dates = df["transfer_date"].isna().sum()

    print(f"Invalid / missing transfer dates: {invalid_dates:,}")

    if invalid_dates == 0:
        print(
            f"Earliest transfer date: "
            f"{df['transfer_date'].min().date()}"
        )
        print(
            f"Latest transfer date:   "
            f"{df['transfer_date'].max().date()}"
        )

    # -------------------------------------------------------------------------
    # Transfer year consistency
    # -------------------------------------------------------------------------

    calculated_transfer_year = df["transfer_date"].dt.year

    transfer_year_mismatch = (
        calculated_transfer_year
        != df["transfer_year"]
    ).sum()

    print(
        f"Transfer year mismatches: "
        f"{transfer_year_mismatch:,}"
    )

    # -------------------------------------------------------------------------
    # Eligibility distribution
    # -------------------------------------------------------------------------

    print("\nELIGIBILITY DISTRIBUTION")
    print("-" * 80)

    eligibility_counts = (
        df["eligibility"]
        .value_counts(dropna=False)
    )

    for value, count in eligibility_counts.items():
        pct = count / len(df) * 100

        print(
            f"{str(value):<25} "
            f"{count:>6,} "
            f"({pct:>6.2f}%)"
        )

    # -------------------------------------------------------------------------
    # Destination availability
    # -------------------------------------------------------------------------

    print("\nDESTINATION AVAILABILITY")
    print("-" * 80)

    destination_counts = (
        df["has_destination"]
        .value_counts()
        .sort_index()
    )

    for value, count in destination_counts.items():
        pct = count / len(df) * 100

        print(
            f"has_destination={str(value):<5} "
            f"{count:>6,} "
            f"({pct:>6.2f}%)"
        )

    # -------------------------------------------------------------------------
    # Transfer status cross-tab
    # -------------------------------------------------------------------------

    print("\nELIGIBILITY × DESTINATION")
    print("-" * 80)

    eligibility_destination = pd.crosstab(
        df["eligibility"],
        df["has_destination"],
        margins=True
    )

    print(eligibility_destination)

    # -------------------------------------------------------------------------
    # Non-withdrawn destination flag
    # -------------------------------------------------------------------------

    print("\nNON-WITHDRAWN DESTINATION")
    print("-" * 80)

    nonwithdrawn_counts = (
        df["is_nonwithdrawn_destination"]
        .value_counts()
        .sort_index()
    )

    for value, count in nonwithdrawn_counts.items():
        pct = count / len(df) * 100

        print(
            f"is_nonwithdrawn_destination={str(value):<5} "
            f"{count:>6,} "
            f"({pct:>6.2f}%)"
        )

    # -------------------------------------------------------------------------
    # Flag consistency checks
    # -------------------------------------------------------------------------

    print("\nFLAG CONSISTENCY CHECKS")
    print("-" * 80)

    expected_has_destination = df["destination"].notna()

    destination_flag_errors = (
        df["has_destination"]
        != expected_has_destination
    ).sum()

    expected_is_withdrawn = (
        df["eligibility"] == "Withdrawn"
    )

    withdrawn_flag_errors = (
        df["is_withdrawn"]
        != expected_is_withdrawn
    ).sum()

    expected_nonwithdrawn_destination = (
        df["has_destination"]
        & ~df["is_withdrawn"]
    )

    nonwithdrawn_destination_errors = (
        df["is_nonwithdrawn_destination"]
        != expected_nonwithdrawn_destination
    ).sum()

    expected_has_rating = df["rating"].notna()

    rating_flag_errors = (
        df["has_rating"]
        != expected_has_rating
    ).sum()

    expected_has_stars = df["stars"].notna()

    stars_flag_errors = (
        df["has_stars"]
        != expected_has_stars
    ).sum()

    print(
        f"has_destination errors: "
        f"{destination_flag_errors:,}"
    )

    print(
        f"is_withdrawn errors: "
        f"{withdrawn_flag_errors:,}"
    )

    print(
        f"is_nonwithdrawn_destination errors: "
        f"{nonwithdrawn_destination_errors:,}"
    )

    print(
        f"has_rating errors: "
        f"{rating_flag_errors:,}"
    )

    print(
        f"has_stars errors: "
        f"{stars_flag_errors:,}"
    )

    # -------------------------------------------------------------------------
    # Semantic checks
    # -------------------------------------------------------------------------

    print("\nSEMANTIC CHECKS")
    print("-" * 80)

    # A non-withdrawn destination must have a destination.
    invalid_nonwithdrawn_destination = (
        df["is_nonwithdrawn_destination"]
        & ~df["has_destination"]
    ).sum()

    # A non-withdrawn destination cannot be withdrawn.
    withdrawn_nonwithdrawn_destination = (
        df["is_nonwithdrawn_destination"]
        & df["is_withdrawn"]
    ).sum()

    # A withdrawn record should not be classified as a
    # non-withdrawn destination.
    withdrawn_with_destination_flagged = (
        df["is_withdrawn"]
        & df["is_nonwithdrawn_destination"]
    ).sum()

    print(
        "Non-withdrawn destination without destination: "
        f"{invalid_nonwithdrawn_destination:,}"
    )

    print(
        "Non-withdrawn destination marked withdrawn: "
        f"{withdrawn_nonwithdrawn_destination:,}"
    )

    print(
        "Withdrawn records classified as "
        "non-withdrawn destination: "
        f"{withdrawn_with_destination_flagged:,}"
    )

    # -------------------------------------------------------------------------
    # Duplicate checks
    # -------------------------------------------------------------------------

    print("\nDUPLICATE CHECKS")
    print("-" * 80)

    exact_duplicates = df.duplicated().sum()

    print(
        f"Exact duplicate records: "
        f"{exact_duplicates:,}"
    )

    # A player name is NOT treated as a unique identifier.
    # This is only a diagnostic to identify repeated names.
    name_duplicates = (
        df.duplicated(
            subset=["first_name", "last_name"],
            keep=False
        )
        .sum()
    )

    print(
        f"Records sharing a first/last name with another record: "
        f"{name_duplicates:,}"
    )

    # -------------------------------------------------------------------------
    # Talent availability
    # -------------------------------------------------------------------------

    print("\nTALENT DATA AVAILABILITY")
    print("-" * 80)

    rating_available = df["has_rating"].sum()
    stars_available = df["has_stars"].sum()

    both_available = (
        df["has_rating"]
        & df["has_stars"]
    ).sum()

    stars_without_rating = (
        ~df["has_rating"]
        & df["has_stars"]
    ).sum()

    rating_without_stars = (
        df["has_rating"]
        & ~df["has_stars"]
    ).sum()

    neither_available = (
        ~df["has_rating"]
        & ~df["has_stars"]
    ).sum()

    print(
        f"Rating available:       "
        f"{rating_available:>6,} "
        f"({rating_available / len(df) * 100:>6.2f}%)"
    )

    print(
        f"Stars available:        "
        f"{stars_available:>6,} "
        f"({stars_available / len(df) * 100:>6.2f}%)"
    )

    print(
        f"Both available:         "
        f"{both_available:>6,} "
        f"({both_available / len(df) * 100:>6.2f}%)"
    )

    print(
        f"Stars without rating:   "
        f"{stars_without_rating:>6,} "
        f"({stars_without_rating / len(df) * 100:>6.2f}%)"
    )

    print(
        f"Rating without stars:   "
        f"{rating_without_stars:>6,} "
        f"({rating_without_stars / len(df) * 100:>6.2f}%)"
    )

    print(
        f"Neither available:      "
        f"{neither_available:>6,} "
        f"({neither_available / len(df) * 100:>6.2f}%)"
    )

    # -------------------------------------------------------------------------
    # Position distribution
    # -------------------------------------------------------------------------

    print("\nPOSITION DISTRIBUTION")
    print("-" * 80)

    position_counts = (
        df["position"]
        .value_counts(dropna=False)
    )

    for position, count in position_counts.items():
        pct = count / len(df) * 100

        print(
            f"{str(position):<10} "
            f"{count:>6,} "
            f"({pct:>6.2f}%)"
        )

    # -------------------------------------------------------------------------
    # Team availability
    # -------------------------------------------------------------------------

    print("\nTEAM AVAILABILITY")
    print("-" * 80)

    origin_count = df["origin"].notna().sum()
    destination_count = df["destination"].notna().sum()

    unique_origins = df["origin"].nunique(
        dropna=True
    )

    unique_destinations = df["destination"].nunique(
        dropna=True
    )

    print(
        f"Records with origin:       "
        f"{origin_count:>6,} "
        f"({origin_count / len(df) * 100:>6.2f}%)"
    )

    print(
        f"Records with destination:  "
        f"{destination_count:>6,} "
        f"({destination_count / len(df) * 100:>6.2f}%)"
    )

    print(
        f"Unique origin teams:       "
        f"{unique_origins:>6,}"
    )

    print(
        f"Unique destination teams:  "
        f"{unique_destinations:>6,}"
    )

    # -------------------------------------------------------------------------
    # Transfer date distribution by calendar year
    # -------------------------------------------------------------------------

    print("\nTRANSFER DATE DISTRIBUTION")
    print("-" * 80)

    transfer_year_counts = (
        df["transfer_year"]
        .value_counts()
        .sort_index()
    )

    for transfer_year, count in transfer_year_counts.items():
        pct = count / len(df) * 100

        print(
            f"{int(transfer_year):<10} "
            f"{count:>6,} "
            f"({pct:>6.2f}%)"
        )

    # -------------------------------------------------------------------------
    # Final audit summary
    # -------------------------------------------------------------------------

    audit_errors = (
        len(missing_columns)
        + destination_flag_errors
        + withdrawn_flag_errors
        + nonwithdrawn_destination_errors
        + rating_flag_errors
        + stars_flag_errors
        + invalid_dates
        + transfer_year_mismatch
        + exact_duplicates
        + invalid_nonwithdrawn_destination
        + withdrawn_nonwithdrawn_destination
        + withdrawn_with_destination_flagged
    )

    print("\n" + "=" * 80)

    if audit_errors == 0:
        print("AUDIT STATUS: PASSED")
    else:
        print(
            f"AUDIT STATUS: REVIEW REQUIRED "
            f"({audit_errors:,} issues)"
        )

    print("=" * 80)


if __name__ == "__main__":

    for year in range(2021, 2026):
        audit_transfer_portal(year)