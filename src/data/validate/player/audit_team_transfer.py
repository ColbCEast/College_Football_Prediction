import os
import pandas as pd


PLAYER_DIR = "data/processed/player/portal"
TEAM_DIR = "data/processed/player/portal"


EXPECTED_TEAM_COLUMNS = [
    "season",
    "team",

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


def audit_transfer_team(year):

    player_path = os.path.join(
        PLAYER_DIR,
        f"transfer_portal_{year}.csv"
    )

    team_path = os.path.join(
        TEAM_DIR,
        f"transfer_team_{year}.csv"
    )

    print("\n" + "=" * 80)
    print(f"TEAM-LEVEL TRANSFER AGGREGATION AUDIT - {year}")
    print("=" * 80)

    player = pd.read_csv(player_path)
    team = pd.read_csv(team_path)

    print("\nSOURCE DATA")
    print("-" * 80)
    print(f"Player-level rows: {len(player):,}")
    print(f"Team-level rows:   {len(team):,}")

    missing_columns = [
        column
        for column in EXPECTED_TEAM_COLUMNS
        if column not in team.columns
    ]

    unexpected_columns = [
        column
        for column in team.columns
        if column not in EXPECTED_TEAM_COLUMNS
    ]

    print("\nCOLUMN CHECK")
    print("-" * 80)
    print(f"Missing expected columns: {len(missing_columns)}")

    if missing_columns:
        print(missing_columns)

    print(f"Unexpected columns: {len(unexpected_columns)}")

    if unexpected_columns:
        print(unexpected_columns)

    print("\nTEAM-SEASON UNIQUENESS")
    print("-" * 80)

    duplicate_team_seasons = team.duplicated(
        subset=["season", "team"]
    ).sum()

    print(
        f"Duplicate season/team rows: "
        f"{duplicate_team_seasons:,}"
    )

    print("\nTEAM IDENTIFIER CHECK")
    print("-" * 80)

    missing_season = team["season"].isna().sum()
    missing_team = team["team"].isna().sum()

    print(f"Missing season: {missing_season:,}")
    print(f"Missing team:   {missing_team:,}")

    print("\nVALUE VALIDATION")
    print("-" * 80)

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

    negative_counts = {}

    for column in count_columns:
        negative = (team[column] < 0).sum()

        if negative > 0:
            negative_counts[column] = negative

    print(
        f"Negative count values: "
        f"{sum(negative_counts.values()):,}"
    )

    if negative_counts:
        print(negative_counts)

    print("\nINCOMING TRANSFER RECONCILIATION")
    print("-" * 80)

    incoming_player = player[
        player["is_nonwithdrawn_destination"]
    ].copy()

    incoming_player = incoming_player.rename(
        columns={
            "destination": "team"
        }
    )

    incoming_expected = (
        incoming_player
        .groupby(
            ["season", "team"],
            dropna=False
        )
        .agg(
            incoming_count=("team", "size"),
            incoming_rated_count=("has_rating", "sum"),
            incoming_rating_sum=("rating", "sum"),
            incoming_rating_mean=("rating", "mean"),
            incoming_star_count=("has_stars", "sum"),
            incoming_stars_sum=("stars", "sum"),
            incoming_stars_mean=("stars", "mean"),
            incoming_2star_count=(
                "stars",
                lambda x: (x == 2).sum()
            ),
            incoming_3star_count=(
                "stars",
                lambda x: (x == 3).sum()
            ),
            incoming_4star_count=(
                "stars",
                lambda x: (x == 4).sum()
            ),
            incoming_5star_count=(
                "stars",
                lambda x: (x == 5).sum()
            ),
        )
        .reset_index()
    )

    print("\nOUTGOING TRANSFER RECONCILIATION")
    print("-" * 80)

    outgoing_player = player[
        player["origin"].notna()
    ].copy()

    outgoing_player = outgoing_player.rename(
        columns={
            "origin": "team"
        }
    )

    outgoing_expected = (
        outgoing_player
        .groupby(
            ["season", "team"],
            dropna=False
        )
        .agg(
            outgoing_count=("team", "size"),
            outgoing_rated_count=("has_rating", "sum"),
            outgoing_rating_sum=("rating", "sum"),
            outgoing_rating_mean=("rating", "mean"),
            outgoing_star_count=("has_stars", "sum"),
            outgoing_stars_sum=("stars", "sum"),
            outgoing_stars_mean=("stars", "mean"),
            outgoing_2star_count=(
                "stars",
                lambda x: (x == 2).sum()
            ),
            outgoing_3star_count=(
                "stars",
                lambda x: (x == 3).sum()
            ),
            outgoing_4star_count=(
                "stars",
                lambda x: (x == 4).sum()
            ),
            outgoing_5star_count=(
                "stars",
                lambda x: (x == 5).sum()
            ),
        )
        .reset_index()
    )

    expected_team_seasons = pd.concat(
        [
            incoming_expected[["season", "team"]],
            outgoing_expected[["season", "team"]],
        ],
        ignore_index=True
    ).drop_duplicates()

    expected = expected_team_seasons.merge(
        incoming_expected,
        on=["season", "team"],
        how="left"
    )

    expected = expected.merge(
        outgoing_expected,
        on=["season", "team"],
        how="left"
    )

    for column in count_columns:
        expected[column] = (
            expected[column]
            .fillna(0)
            .astype(int)
        )

    actual = team.copy()

    actual = actual.sort_values(
        ["season", "team"]
    ).reset_index(drop=True)

    expected = expected.sort_values(
        ["season", "team"]
    ).reset_index(drop=True)

    actual_index = pd.MultiIndex.from_frame(
        actual[["season", "team"]]
    )

    expected_index = pd.MultiIndex.from_frame(
        expected[["season", "team"]]
    )

    missing_team_seasons = expected_index.difference(
        actual_index
    )

    unexpected_team_seasons = actual_index.difference(
        expected_index
    )

    print(
        f"\nExpected team-seasons: "
        f"{len(expected):,}"
    )

    print(
        f"Actual team-seasons:   "
        f"{len(actual):,}"
    )

    print(
        f"Missing team-seasons:  "
        f"{len(missing_team_seasons):,}"
    )

    print(
        f"Unexpected team-seasons: "
        f"{len(unexpected_team_seasons):,}"
    )

    comparison = expected.merge(
        actual,
        on=["season", "team"],
        how="outer",
        suffixes=("_expected", "_actual"),
        indicator=True
    )

    print("\nCOUNT RECONCILIATION")
    print("-" * 80)

    count_mismatches = {}

    for column in count_columns:
        expected_column = f"{column}_expected"
        actual_column = f"{column}_actual"

        mismatch = (
            comparison[expected_column].fillna(0)
            != comparison[actual_column].fillna(0)
        )

        count_mismatches[column] = mismatch.sum()

        print(
            f"{column:<30} "
            f"{mismatch.sum():>6,} mismatches"
        )

    print("\nTALENT AGGREGATE RECONCILIATION")
    print("-" * 80)

    numeric_talent_columns = [
        "incoming_rating_sum",
        "incoming_rating_mean",
        "incoming_stars_sum",
        "incoming_stars_mean",
        "outgoing_rating_sum",
        "outgoing_rating_mean",
        "outgoing_stars_sum",
        "outgoing_stars_mean",
    ]

    talent_mismatches = {}

    for column in numeric_talent_columns:
        expected_column = f"{column}_expected"
        actual_column = f"{column}_actual"

        expected_values = comparison[expected_column]
        actual_values = comparison[actual_column]

        mismatch = ~(
            expected_values.eq(actual_values)
            | (
                expected_values.isna()
                & actual_values.isna()
            )
        )

        both_present = (
            expected_values.notna()
            & actual_values.notna()
        )

        mismatch = (
            mismatch
            & ~(
                both_present
                & (
                    (expected_values - actual_values)
                    .abs()
                    < 1e-10
                )
            )
        )

        talent_mismatches[column] = mismatch.sum()

        print(
            f"{column:<30} "
            f"{mismatch.sum():>6,} mismatches"
        )

    print("\nSTAR COUNT RECONCILIATION")
    print("-" * 80)

    star_count_columns = [
        "incoming_2star_count",
        "incoming_3star_count",
        "incoming_4star_count",
        "incoming_5star_count",
        "outgoing_2star_count",
        "outgoing_3star_count",
        "outgoing_4star_count",
        "outgoing_5star_count",
    ]

    star_mismatches = {}

    for column in star_count_columns:
        expected_column = f"{column}_expected"
        actual_column = f"{column}_actual"

        mismatch = (
            comparison[expected_column].fillna(0)
            != comparison[actual_column].fillna(0)
        )

        star_mismatches[column] = mismatch.sum()

        print(
            f"{column:<30} "
            f"{mismatch.sum():>6,} mismatches"
        )

    print("\nINCOMING STAR CATEGORY CONSISTENCY")
    print("-" * 80)

    incoming_star_categories = (
        team["incoming_2star_count"]
        + team["incoming_3star_count"]
        + team["incoming_4star_count"]
        + team["incoming_5star_count"]
    )

    incoming_star_exceeds_available = (
        incoming_star_categories
        > team["incoming_star_count"]
    ).sum()

    incoming_uncategorized_stars = (
        team["incoming_star_count"]
        - incoming_star_categories
    )

    print(
        "Categorized 2-5 star count > "
        f"incoming_star_count: "
        f"{incoming_star_exceeds_available:,}"
    )

    print(
        "Incoming stars with values outside "
        f"2-5 categories: "
        f"{(incoming_uncategorized_stars > 0).sum():,} "
        "team-seasons"
    )

    print(
        "Total incoming stars outside "
        f"2-5 categories: "
        f"{incoming_uncategorized_stars.sum():,}"
    )

    print("\nOUTGOING STAR CATEGORY CONSISTENCY")
    print("-" * 80)

    outgoing_star_categories = (
        team["outgoing_2star_count"]
        + team["outgoing_3star_count"]
        + team["outgoing_4star_count"]
        + team["outgoing_5star_count"]
    )

    outgoing_star_exceeds_available = (
        outgoing_star_categories
        > team["outgoing_star_count"]
    ).sum()

    outgoing_uncategorized_stars = (
        team["outgoing_star_count"]
        - outgoing_star_categories
    )

    print(
        "Categorized 2-5 star count > "
        f"outgoing_star_count: "
        f"{outgoing_star_exceeds_available:,}"
    )

    print(
        "Outgoing stars with values outside "
        f"2-5 categories: "
        f"{(outgoing_uncategorized_stars > 0).sum():,} "
        "team-seasons"
    )

    print(
        "Total outgoing stars outside "
        f"2-5 categories: "
        f"{outgoing_uncategorized_stars.sum():,}"
    )

    print("\nRATED / STAR COUNT CONSISTENCY")
    print("-" * 80)

    incoming_rated_exceeds_total = (
        team["incoming_rated_count"]
        > team["incoming_count"]
    ).sum()

    outgoing_rated_exceeds_total = (
        team["outgoing_rated_count"]
        > team["outgoing_count"]
    ).sum()

    incoming_stars_exceeds_total = (
        team["incoming_star_count"]
        > team["incoming_count"]
    ).sum()

    outgoing_stars_exceeds_total = (
        team["outgoing_star_count"]
        > team["outgoing_count"]
    ).sum()

    print(
        "Incoming rated count > incoming count: "
        f"{incoming_rated_exceeds_total:,}"
    )

    print(
        "Outgoing rated count > outgoing count: "
        f"{outgoing_rated_exceeds_total:,}"
    )

    print(
        "Incoming star count > incoming count: "
        f"{incoming_stars_exceeds_total:,}"
    )

    print(
        "Outgoing star count > outgoing count: "
        f"{outgoing_stars_exceeds_total:,}"
    )

    print("\nSEASON-LEVEL RECORD RECONCILIATION")
    print("-" * 80)

    incoming_player_total = len(incoming_player)
    outgoing_player_total = len(outgoing_player)

    incoming_team_total = team["incoming_count"].sum()
    outgoing_team_total = team["outgoing_count"].sum()

    print(
        f"Incoming player records: "
        f"{incoming_player_total:,}"
    )

    print(
        f"Incoming team aggregate: "
        f"{incoming_team_total:,}"
    )

    print(
        f"Outgoing player records: "
        f"{outgoing_player_total:,}"
    )

    print(
        f"Outgoing team aggregate: "
        f"{outgoing_team_total:,}"
    )

    incoming_total_mismatch = (
        incoming_player_total
        != incoming_team_total
    )

    outgoing_total_mismatch = (
        outgoing_player_total
        != outgoing_team_total
    )

    print(
        f"Incoming total mismatch: "
        f"{incoming_total_mismatch}"
    )

    print(
        f"Outgoing total mismatch: "
        f"{outgoing_total_mismatch}"
    )

    total_errors = (
        len(missing_columns)
        + duplicate_team_seasons
        + missing_season
        + missing_team
        + sum(negative_counts.values())
        + len(missing_team_seasons)
        + len(unexpected_team_seasons)
        + sum(count_mismatches.values())
        + sum(talent_mismatches.values())
        + sum(star_mismatches.values())
        + incoming_star_exceeds_available
        + outgoing_star_exceeds_available
        + incoming_rated_exceeds_total
        + outgoing_rated_exceeds_total
        + incoming_stars_exceeds_total
        + outgoing_stars_exceeds_total
        + int(incoming_total_mismatch)
        + int(outgoing_total_mismatch)
    )

    print("\n" + "=" * 80)

    if total_errors == 0:
        print("AUDIT STATUS: PASSED")
    else:
        print(
            f"AUDIT STATUS: REVIEW REQUIRED "
            f"({total_errors:,} issues)"
        )

    print("=" * 80)


if __name__ == "__main__":
    for year in range(2021, 2026):
        audit_transfer_team(year)