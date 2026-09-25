import os
import pandas as pd


INPUT_DIR = "data/raw/player/portal"
OUTPUT_DIR = "data/processed/player/portal"


def transform_transfer_portal(year):
    input_path = os.path.join(
        INPUT_DIR,
        f"player_transfer_{year}.csv"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"transfer_portal_{year}.csv"
    )

    df = pd.read_csv(input_path)

    # ------------------------------------------------------------------
    # Standardize column names
    # ------------------------------------------------------------------

    df = df.rename(
        columns={
            "season": "season",
            "firstName": "first_name",
            "lastName": "last_name",
            "position": "position",
            "origin": "origin",
            "destination": "destination",
            "transferDate": "transfer_date",
            "rating": "rating",
            "stars": "stars",
            "eligibility": "eligibility",
        }
    )

    # ------------------------------------------------------------------
    # Standardize transfer date
    # ------------------------------------------------------------------

    df["transfer_date"] = pd.to_datetime(
        df["transfer_date"],
        errors="coerce",
        utc=True
    )

    # ------------------------------------------------------------------
    # Transfer status indicators
    # ------------------------------------------------------------------

    df["has_destination"] = df["destination"].notna()

    df["is_withdrawn"] = (
        df["eligibility"].eq("Withdrawn")
    )

    df["is_completed_transfer"] = (
        df["has_destination"]
        & ~df["is_withdrawn"]
    )

    # ------------------------------------------------------------------
    # Talent-data availability indicators
    # ------------------------------------------------------------------

    df["has_rating"] = df["rating"].notna()

    df["has_stars"] = df["stars"].notna()

    # ------------------------------------------------------------------
    # Calendar date information
    # ------------------------------------------------------------------

    df["transfer_year"] = df["transfer_date"].dt.year

    # ------------------------------------------------------------------
    # Column ordering
    # ------------------------------------------------------------------

    columns = [
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
        "is_completed_transfer",
        "rating",
        "has_rating",
        "stars",
        "has_stars",
    ]

    df = df[columns]

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Transformed {year}: "
        f"{len(df):,} records → {output_path}"
    )

    return df


if __name__ == "__main__":

    for year in range(2021, 2026):
        transform_transfer_portal(year)