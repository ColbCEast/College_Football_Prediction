import os
import pandas as pd


INPUT_DIR = "data/processed/player/portal"
OUTPUT_DIR = "data/processed/player/portal"


def aggregate_transfer_portal(year):
    input_path = os.path.join(
        INPUT_DIR,
        f"transfer_portal_{year}.csv"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"transfer_team_{year}.csv"
    )

    df = pd.read_csv(input_path)

    # -------------------------------------------------------------------------
    # Data types
    # -------------------------------------------------------------------------

    df["transfer_date"] = pd.to_datetime(
        df["transfer_date"],
        errors="coerce",
        utc=True
    )

    # -------------------------------------------------------------------------
    # Prepare incoming transfer records
    # -------------------------------------------------------------------------
    #
    # An incoming transfer must:
    #   1. Have a destination
    #   2. Not be marked Withdrawn
    #
    # We intentionally use the descriptive flag created during the
    # player-level transformation rather than assuming the transfer was
    # definitively completed.
    # -------------------------------------------------------------------------

    incoming = df[
        df["is_nonwithdrawn_destination"]
    ].copy()

    incoming = incoming.rename(
        columns={
            "destination": "team"
        }
    )

    # -------------------------------------------------------------------------
    # Prepare outgoing transfer records
    # -------------------------------------------------------------------------
    #
    # Origin is available for every record in the current CFBD data.
    #
    # We retain all origin records here rather than assuming that every
    # destination-bearing record represents a completed transfer.
    # This preserves the underlying portal activity for later analysis.
    # -------------------------------------------------------------------------

    outgoing = df[
        df["origin"].notna()
    ].copy()

    outgoing = outgoing.rename(
        columns={
            "origin": "team"
        }
    )

    # -------------------------------------------------------------------------
    # Helper function for talent aggregates
    # -------------------------------------------------------------------------

    def aggregate_side(data, prefix):
        if data.empty:
            return pd.DataFrame(
                columns=[
                    "season",
                    "team",
                    f"{prefix}_count",
                    f"{prefix}_rated_count",
                    f"{prefix}_rating_sum",
                    f"{prefix}_rating_mean",
                    f"{prefix}_star_count",
                    f"{prefix}_stars_sum",
                    f"{prefix}_stars_mean",
                    f"{prefix}_2star_count",
                    f"{prefix}_3star_count",
                    f"{prefix}_4star_count",
                    f"{prefix}_5star_count",
                ]
            )

        grouped = (
            data
            .groupby(
                ["season", "team"],
                dropna=False
            )
            .agg(
                **{
                    f"{prefix}_count": (
                        "team",
                        "size"
                    ),
                    f"{prefix}_rated_count": (
                        "has_rating",
                        "sum"
                    ),
                    f"{prefix}_rating_sum": (
                        "rating",
                        "sum"
                    ),
                    f"{prefix}_rating_mean": (
                        "rating",
                        "mean"
                    ),
                    f"{prefix}_star_count": (
                        "has_stars",
                        "sum"
                    ),
                    f"{prefix}_stars_sum": (
                        "stars",
                        "sum"
                    ),
                    f"{prefix}_stars_mean": (
                        "stars",
                        "mean"
                    ),
                    f"{prefix}_2star_count": (
                        "stars",
                        lambda x: (x == 2).sum()
                    ),
                    f"{prefix}_3star_count": (
                        "stars",
                        lambda x: (x == 3).sum()
                    ),
                    f"{prefix}_4star_count": (
                        "stars",
                        lambda x: (x == 4).sum()
                    ),
                    f"{prefix}_5star_count": (
                        "stars",
                        lambda x: (x == 5).sum()
                    ),
                }
            )
            .reset_index()
        )

        return grouped

    # -------------------------------------------------------------------------
    # Aggregate incoming transfers
    # -------------------------------------------------------------------------

    incoming_agg = aggregate_side(
        incoming,
        "incoming"
    )

    # -------------------------------------------------------------------------
    # Aggregate outgoing transfers
    # -------------------------------------------------------------------------

    outgoing_agg = aggregate_side(
        outgoing,
        "outgoing"
    )

    # -------------------------------------------------------------------------
    # Create complete team-season index
    # -------------------------------------------------------------------------
    #
    # Using the union of teams appearing on either side ensures teams with
    # only incoming or only outgoing records are retained.
    # -------------------------------------------------------------------------

    team_seasons = pd.concat(
        [
            incoming_agg[["season", "team"]],
            outgoing_agg[["season", "team"]],
        ],
        ignore_index=True
    ).drop_duplicates()

    # -------------------------------------------------------------------------
    # Merge incoming and outgoing aggregates
    # -------------------------------------------------------------------------

    team_agg = team_seasons.merge(
        incoming_agg,
        on=["season", "team"],
        how="left"
    )

    team_agg = team_agg.merge(
        outgoing_agg,
        on=["season", "team"],
        how="left"
    )

    # -------------------------------------------------------------------------
    # Fill count fields
    # -------------------------------------------------------------------------
    #
    # Missing counts mean that the team had no records on that side.
    # These can safely be represented as zero.
    #
    # Talent sums/means remain missing when no talent information exists.
    # We do NOT convert missing ratings/stars to zero.
    # -------------------------------------------------------------------------

    count_columns = [
        "incoming_count",
        "incoming_rated_count",
        "incoming_star_count",
        "incoming_2star_count",
        "incoming_3star_count",
        "incoming_4star_count",
        "incoming_5star_count",
        "outgoing_count",
        "outgoing_rated_count",
        "outgoing_star_count",
        "outgoing_2star_count",
        "outgoing_3star_count",
        "outgoing_4star_count",
        "outgoing_5star_count",
    ]

    for column in count_columns:
        team_agg[column] = (
            team_agg[column]
            .fillna(0)
            .astype(int)
        )

    # -------------------------------------------------------------------------
    # Final column order
    # -------------------------------------------------------------------------

    columns = [
        "season",
        "team",

        # Incoming transfer activity
        "incoming_count",
        "incoming_rated_count",
        "incoming_rating_sum",
        "incoming_rating_mean",
        "incoming_star_count",
        "incoming_stars_sum",
        "incoming_stars_mean",
        "incoming_2star_count",
        "incoming_3star_count",
        "incoming_4star_count",
        "incoming_5star_count",

        # Outgoing transfer activity
        "outgoing_count",
        "outgoing_rated_count",
        "outgoing_rating_sum",
        "outgoing_rating_mean",
        "outgoing_star_count",
        "outgoing_stars_sum",
        "outgoing_stars_mean",
        "outgoing_2star_count",
        "outgoing_3star_count",
        "outgoing_4star_count",
        "outgoing_5star_count",
    ]

    team_agg = team_agg[columns]

    # -------------------------------------------------------------------------
    # Sort
    # -------------------------------------------------------------------------

    team_agg = team_agg.sort_values(
        ["season", "team"]
    ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    team_agg.to_csv(
        output_path,
        index=False
    )

    print(
        f"Aggregated {year}: "
        f"{len(team_agg):,} team-seasons → {output_path}"
    )

    return team_agg


if __name__ == "__main__":

    for year in range(2021, 2026):
        aggregate_transfer_portal(year)