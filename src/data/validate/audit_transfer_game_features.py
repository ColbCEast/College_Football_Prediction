"""
Cross-Season Transfer Game Features Audit
==========================================

Purpose
-------
Independently audit transfer game-level features for seasons 2021-2025.

This audit checks:

1. Raw stars-value encoding
2. Raw transfer -> game-feature count reconciliation
3. Rating count / sum / mean reconciliation
4. Star-count reconciliation
5. Temporal cutoff behavior
6. First-game-of-season availability
7. Large / unusual transfer counts
8. Game/team uniqueness
9. Season-level comparability
10. Prior-year-dated transfer records

This script is READ-ONLY. It does not modify source or output files.

Expected inputs
---------------
Processed transfers:
    data/processed/player/portal/transfer_portal_{year}.csv

Final game features:
    data/processed/features/final/final_features_{year}.csv

Generated transfer game features:
    data/processed/player/portal/game_features/transfer_game_features_{year}.csv

Run from project root:
    python src/data/validate/audit_transfer_game_features.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

ROOT = Path(__file__).resolve().parents[3]

TRANSFER_DIR = (
    ROOT
    / "data"
    / "processed"
    / "player"
    / "portal"
)

GAME_FEATURE_DIR = (
    ROOT
    / "data"
    / "processed"
    / "features"
    / "final"
)

TRANSFER_GAME_FEATURE_DIR = TRANSFER_DIR / "game_features"

YEARS = range(2021, 2026)

STAR_VALUES = [2, 3, 4, 5]

DIRECTION_CONFIG = {
    "incoming": {
        "team_column": "destination",
    },
    "outgoing": {
        "team_column": "origin",
    },
}


# =============================================================================
# HELPERS
# =============================================================================

def print_section(title):
    """Print a formatted audit section header."""
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_subsection(title):
    """Print a formatted subsection header."""
    print()
    print("-" * 80)
    print(title)
    print("-" * 80)


def require_columns(df, required, dataset_name):
    """Raise a clear error if required columns are missing."""
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}"
        )


def load_season_data(year):
    """Load processed transfers, final game features, and generated transfer features."""

    transfer_path = (
        TRANSFER_DIR
        / f"transfer_portal_{year}.csv"
    )

    game_path = (
        GAME_FEATURE_DIR
        / f"final_features_{year}.csv"
    )

    transfer_feature_path = (
        TRANSFER_GAME_FEATURE_DIR
        / f"transfer_game_features_{year}.csv"
    )

    required_paths = [
        transfer_path,
        game_path,
        transfer_feature_path,
    ]

    missing_paths = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing_paths:
        print(f"\nSkipping {year}: missing files")

        for path in missing_paths:
            print(f"  Missing: {path}")

        return None

    transfers = pd.read_csv(transfer_path)
    games = pd.read_csv(game_path)
    transfer_features = pd.read_csv(transfer_feature_path)

    return transfers, games, transfer_features


def prepare_transfers(transfers, year):
    """Prepare processed transfer data for independent auditing."""

    required = [
        "season",
        "origin",
        "destination",
        "transfer_date",
        "transfer_year",
        "is_withdrawn",
        "rating",
        "stars",
    ]

    require_columns(
        transfers,
        required,
        f"transfer data {year}",
    )

    transfers = transfers.copy()

    transfers["transfer_date"] = pd.to_datetime(
        transfers["transfer_date"],
        errors="coerce",
        utc=True,
    )

    transfers["rating"] = pd.to_numeric(
        transfers["rating"],
        errors="coerce",
    )

    transfers["stars"] = pd.to_numeric(
        transfers["stars"],
        errors="coerce",
    )

    transfers["origin"] = transfers["origin"].replace(
        {"": np.nan}
    )

    transfers["destination"] = transfers["destination"].replace(
        {"": np.nan}
    )

    transfers["is_withdrawn"] = (
        transfers["is_withdrawn"]
        .fillna(False)
        .astype(bool)
    )

    return transfers


def prepare_games(games, year):
    """Prepare final game data."""

    required = [
        "gameId",
        "season",
        "startDate",
        "homeTeam",
        "awayTeam",
    ]

    require_columns(
        games,
        required,
        f"final game data {year}",
    )

    games = games.copy()

    games["startDate"] = pd.to_datetime(
        games["startDate"],
        errors="coerce",
        utc=True,
    )

    if games["startDate"].isna().any():
        count = games["startDate"].isna().sum()

        raise ValueError(
            f"{year}: {count} game rows have missing startDate."
        )

    return games


def prepare_transfer_features(features, year):
    """Prepare generated transfer game features."""

    required = [
        "gameId",
        "season",
        "startDate",
        "team",
        "homeAway",
        "incoming_transfer_count",
        "incoming_rated_player_count",
        "incoming_rating_sum",
        "incoming_rating_mean",
        "incoming_star_rated_player_count",
        "outgoing_transfer_count",
        "outgoing_rated_player_count",
        "outgoing_rating_sum",
        "outgoing_rating_mean",
        "outgoing_star_rated_player_count",
    ]

    for star in STAR_VALUES:
        required.append(f"incoming_{star}_star_count")
        required.append(f"outgoing_{star}_star_count")

    require_columns(
        features,
        required,
        f"transfer game features {year}",
    )

    features = features.copy()

    features["startDate"] = pd.to_datetime(
        features["startDate"],
        errors="coerce",
        utc=True,
    )

    return features


def make_team_game_table(games):
    """Create one row per game/team."""

    home = games[
        [
            "gameId",
            "season",
            "startDate",
            "homeTeam",
        ]
    ].copy()

    home = home.rename(
        columns={"homeTeam": "team"}
    )

    home["homeAway"] = "home"

    away = games[
        [
            "gameId",
            "season",
            "startDate",
            "awayTeam",
        ]
    ].copy()

    away = away.rename(
        columns={"awayTeam": "team"}
    )

    away["homeAway"] = "away"

    team_games = pd.concat(
        [home, away],
        ignore_index=True,
    )

    return team_games


# =============================================================================
# 1. STAR ENCODING AUDIT
# =============================================================================

def audit_star_encoding(transfers, year):
    """
    Identify every distinct non-missing stars value.

    This is especially important because the generated feature script
    currently counts only stars 2, 3, 4, and 5.
    """

    print_subsection(
        f"{year} - Raw Stars Value Encoding"
    )

    stars = transfers["stars"].dropna()

    if stars.empty:
        print("No non-missing stars values.")
        return

    counts = (
        stars
        .value_counts(dropna=False)
        .sort_index()
    )

    print("Distinct non-missing stars values:")

    for value, count in counts.items():
        print(
            f"  {value!r}: {count:,} records"
        )

    configured_values = set(STAR_VALUES)
    observed_values = set(stars.unique())

    unexpected = sorted(
        observed_values - configured_values
    )

    print()

    if unexpected:
        print(
            "WARNING: Non-missing stars values outside "
            f"{STAR_VALUES} were found:"
        )

        for value in unexpected:
            count = int((stars == value).sum())

            print(
                f"  stars={value!r}: {count:,} records"
            )

        print(
            "\nThese values explain why "
            "star_rated_player_count may exceed the sum of "
            "the 2/3/4/5-star feature columns."
        )
    else:
        print(
            "PASS: All non-missing stars values are represented "
            f"by {STAR_VALUES}."
        )


# =============================================================================
# 2. SOURCE-LEVEL RECONCILIATION
# =============================================================================

def calculate_expected_features(
    transfers,
    team,
    game_date,
    direction,
):
    """
    Independently calculate expected transfer features for one
    team/game from player-level transfer records.

    Eligibility:
        transfer_date <= game_date

    Withdrawn transfers are excluded.

    Incoming:
        destination == team

    Outgoing:
        origin == team
    """

    team_column = DIRECTION_CONFIG[direction]["team_column"]

    eligible = transfers[
        (~transfers["is_withdrawn"])
        & (transfers["transfer_date"].notna())
        & (transfers[team_column].notna())
        & (transfers[team_column] == team)
        & (transfers["transfer_date"] <= game_date)
    ].copy()

    transfer_count = len(eligible)

    rated = eligible[
        eligible["rating"].notna()
    ]

    star_rated = eligible[
        eligible["stars"].notna()
    ]

    rating_count = len(rated)
    rating_sum = rated["rating"].sum()

    if rating_count > 0:
        rating_mean = rated["rating"].mean()
    else:
        rating_mean = np.nan

    star_rated_count = len(star_rated)

    star_counts = {}

    for star in STAR_VALUES:
        star_counts[star] = int(
            (eligible["stars"] == star).sum()
        )

    return {
        "transfer_count": transfer_count,
        "rated_player_count": rating_count,
        "rating_sum": rating_sum,
        "rating_mean": rating_mean,
        "star_rated_player_count": star_rated_count,
        "star_counts": star_counts,
    }


def audit_feature_reconciliation(
    transfers,
    transfer_features,
    year,
):
    """
    Compare generated game features to independently calculated
    eligible transfer aggregates.
    """

    print_subsection(
        f"{year} - Independent Source-to-Feature Reconciliation"
    )

    mismatches = []

    for row in transfer_features.itertuples(index=False):

        game_id = row.gameId
        team = row.team
        game_date = row.startDate

        for direction in ["incoming", "outgoing"]:

            expected = calculate_expected_features(
                transfers=transfers,
                team=team,
                game_date=game_date,
                direction=direction,
            )

            generated_count = getattr(
                row,
                f"{direction}_transfer_count",
            )

            generated_rated = getattr(
                row,
                f"{direction}_rated_player_count",
            )

            generated_rating_sum = getattr(
                row,
                f"{direction}_rating_sum",
            )

            generated_rating_mean = getattr(
                row,
                f"{direction}_rating_mean",
            )

            generated_star_rated = getattr(
                row,
                f"{direction}_star_rated_player_count",
            )

            # -----------------------------------------------------------------
            # Transfer count
            # -----------------------------------------------------------------

            if not np.isclose(
                generated_count,
                expected["transfer_count"],
            ):
                mismatches.append(
                    {
                        "gameId": game_id,
                        "team": team,
                        "direction": direction,
                        "feature": "transfer_count",
                        "generated": generated_count,
                        "expected": expected["transfer_count"],
                    }
                )

            # -----------------------------------------------------------------
            # Rated-player count
            # -----------------------------------------------------------------

            if not np.isclose(
                generated_rated,
                expected["rated_player_count"],
            ):
                mismatches.append(
                    {
                        "gameId": game_id,
                        "team": team,
                        "direction": direction,
                        "feature": "rated_player_count",
                        "generated": generated_rated,
                        "expected": expected["rated_player_count"],
                    }
                )

            # -----------------------------------------------------------------
            # Rating sum
            # -----------------------------------------------------------------

            if not np.isclose(
                generated_rating_sum,
                expected["rating_sum"],
                atol=1e-8,
            ):
                mismatches.append(
                    {
                        "gameId": game_id,
                        "team": team,
                        "direction": direction,
                        "feature": "rating_sum",
                        "generated": generated_rating_sum,
                        "expected": expected["rating_sum"],
                    }
                )

            # -----------------------------------------------------------------
            # Rating mean
            # -----------------------------------------------------------------

            if pd.isna(expected["rating_mean"]):

                if not pd.isna(generated_rating_mean):
                    mismatches.append(
                        {
                            "gameId": game_id,
                            "team": team,
                            "direction": direction,
                            "feature": "rating_mean",
                            "generated": generated_rating_mean,
                            "expected": np.nan,
                        }
                    )

            elif not np.isclose(
                generated_rating_mean,
                expected["rating_mean"],
                atol=1e-8,
            ):
                mismatches.append(
                    {
                        "gameId": game_id,
                        "team": team,
                        "direction": direction,
                        "feature": "rating_mean",
                        "generated": generated_rating_mean,
                        "expected": expected["rating_mean"],
                    }
                )

            # -----------------------------------------------------------------
            # Star-rated count
            # -----------------------------------------------------------------

            if not np.isclose(
                generated_star_rated,
                expected["star_rated_player_count"],
            ):
                mismatches.append(
                    {
                        "gameId": game_id,
                        "team": team,
                        "direction": direction,
                        "feature": "star_rated_player_count",
                        "generated": generated_star_rated,
                        "expected": expected["star_rated_player_count"],
                    }
                )

            # -----------------------------------------------------------------
            # Individual star counts
            # -----------------------------------------------------------------

            for star in STAR_VALUES:

                generated_star = getattr(
                    row,
                    f"{direction}_{star}_star_count",
                )

                expected_star = expected[
                    "star_counts"
                ][star]

                if not np.isclose(
                    generated_star,
                    expected_star,
                ):
                    mismatches.append(
                        {
                            "gameId": game_id,
                            "team": team,
                            "direction": direction,
                            "feature": f"{star}_star_count",
                            "generated": generated_star,
                            "expected": expected_star,
                        }
                    )

    if mismatches:

        mismatch_df = pd.DataFrame(
            mismatches
        )

        print(
            f"FAIL: {len(mismatch_df):,} feature mismatches found."
        )

        print(
            "\nMismatch counts by feature:"
        )

        print(
            mismatch_df["feature"]
            .value_counts()
            .to_string()
        )

        print(
            "\nFirst 20 mismatches:"
        )

        print(
            mismatch_df
            .head(20)
            .to_string(index=False)
        )

        return False

    print(
        "PASS: Generated transfer features reconcile exactly "
        "with independently calculated eligible transfer aggregates."
    )

    return True


# =============================================================================
# 3. TEMPORAL AUDIT
# =============================================================================

def audit_temporal_behavior(
    transfers,
    team_games,
    transfer_features,
    year,
):
    """
    Verify that transfers occurring after a game are not included
    in that game's generated features.
    """

    print_subsection(
        f"{year} - Temporal Cutoff Audit"
    )

    feature_lookup = (
        transfer_features
        .set_index(["gameId", "team"])
    )

    for direction in ["incoming", "outgoing"]:

        team_column = DIRECTION_CONFIG[direction]["team_column"]

        future_comparisons = 0
        failures = []

        for row in team_games.itertuples(index=False):

            team = row.team
            game_date = row.startDate
            game_id = row.gameId

            future = transfers[
                (~transfers["is_withdrawn"])
                & (transfers["transfer_date"].notna())
                & (transfers[team_column] == team)
                & (transfers["transfer_date"] > game_date)
            ]

            if future.empty:
                continue

            future_comparisons += len(future)

            generated_count = feature_lookup.loc[
                (game_id, team),
                f"{direction}_transfer_count",
            ]

            eligible_count = len(
                transfers[
                    (~transfers["is_withdrawn"])
                    & (transfers["transfer_date"].notna())
                    & (transfers[team_column] == team)
                    & (transfers["transfer_date"] <= game_date)
                ]
            )

            if generated_count != eligible_count:
                failures.append(
                    {
                        "gameId": game_id,
                        "team": team,
                        "game_date": game_date,
                        "future_transfer_count": len(future),
                        "generated_count": generated_count,
                        "eligible_count": eligible_count,
                    }
                )

        print(
            f"{direction.capitalize()} future-transfer comparisons: "
            f"{future_comparisons:,}"
        )

        if failures:

            print(
                "FAIL: Generated features do not match the eligible "
                "pre-game transfer population."
            )

            print(
                pd.DataFrame(
                    failures
                )
                .head(20)
                .to_string(index=False)
            )

        else:

            print(
                "PASS: Future transfers are excluded from generated "
                f"{direction} features."
            )


# =============================================================================
# 4. FIRST-GAME AUDIT
# =============================================================================

def audit_first_game(
    transfers,
    games,
    transfer_features,
    year,
):
    """
    Examine the first game of the season and compare generated
    features against transfers available by that timestamp.
    """

    print_subsection(
        f"{year} - First-Game Availability Audit"
    )

    first_game_date = games["startDate"].min()

    first_games = games[
        games["startDate"] == first_game_date
    ].copy()

    first_team_games = make_team_game_table(
        first_games
    )

    print(
        f"First game timestamp: {first_game_date}"
    )

    print(
        f"Games at first timestamp: {len(first_games):,}"
    )

    mismatches = []

    for row in first_team_games.itertuples(index=False):

        feature_row = transfer_features[
            (transfer_features["gameId"] == row.gameId)
            & (transfer_features["team"] == row.team)
        ]

        if len(feature_row) != 1:

            mismatches.append(
                {
                    "gameId": row.gameId,
                    "team": row.team,
                    "issue": "Missing or duplicate feature row",
                }
            )

            continue

        feature_row = feature_row.iloc[0]

        for direction in ["incoming", "outgoing"]:

            expected = calculate_expected_features(
                transfers=transfers,
                team=row.team,
                game_date=row.startDate,
                direction=direction,
            )

            generated = feature_row[
                f"{direction}_transfer_count"
            ]

            if generated != expected["transfer_count"]:

                mismatches.append(
                    {
                        "gameId": row.gameId,
                        "team": row.team,
                        "direction": direction,
                        "generated": generated,
                        "expected": expected["transfer_count"],
                    }
                )

    if mismatches:

        print(
            f"FAIL: {len(mismatches):,} first-game mismatches found."
        )

        print(
            pd.DataFrame(
                mismatches
            )
            .head(20)
            .to_string(index=False)
        )

    else:

        print(
            "PASS: First-game transfer features exactly match "
            "transfers available by the first game timestamp."
        )


# =============================================================================
# 5. LARGE COUNT AUDIT
# =============================================================================

def audit_large_counts(
    transfer_features,
    year,
):
    """
    Identify unusually large transfer counts.

    This is diagnostic rather than an automatic failure.
    """

    print_subsection(
        f"{year} - Large Transfer Count Audit"
    )

    for direction in ["incoming", "outgoing"]:

        column = f"{direction}_transfer_count"

        series = transfer_features[column]

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        upper_fence = q3 + 1.5 * iqr

        extreme = transfer_features[
            series > upper_fence
        ][
            [
                "gameId",
                "startDate",
                "team",
                "homeAway",
                column,
            ]
        ].sort_values(
            column,
            ascending=False,
        )

        print(
            f"{direction.capitalize()}:"
        )

        print(
            f"  Mean: {series.mean():.2f}"
        )

        print(
            f"  Median: {series.median():.2f}"
        )

        print(
            f"  Maximum: {series.max():.0f}"
        )

        print(
            f"  IQR upper fence: {upper_fence:.2f}"
        )

        print(
            f"  Rows above upper fence: {len(extreme):,}"
        )

        if not extreme.empty:

            print(
                "  Largest examples:"
            )

            print(
                extreme
                .head(10)
                .to_string(index=False)
            )


# =============================================================================
# 6. GAME / TEAM UNIQUENESS
# =============================================================================

def audit_game_team_uniqueness(
    games,
    transfer_features,
    year,
):
    """Verify game and team keys are structurally correct."""

    print_subsection(
        f"{year} - Game/Team Structural Audit"
    )

    expected_rows = len(games) * 2
    actual_rows = len(transfer_features)

    duplicate_feature_rows = (
        transfer_features
        .duplicated(["gameId", "team"])
        .sum()
    )

    expected_game_ids = set(
        games["gameId"]
    )

    feature_game_ids = set(
        transfer_features["gameId"]
    )

    missing_game_ids = (
        expected_game_ids
        - feature_game_ids
    )

    unexpected_game_ids = (
        feature_game_ids
        - expected_game_ids
    )

    print(
        f"Expected game/team rows: {expected_rows:,}"
    )

    print(
        f"Generated game/team rows: {actual_rows:,}"
    )

    print(
        f"Duplicate game/team rows: {duplicate_feature_rows:,}"
    )

    print(
        f"Missing game IDs: {len(missing_game_ids):,}"
    )

    print(
        f"Unexpected game IDs: {len(unexpected_game_ids):,}"
    )

    if (
        expected_rows == actual_rows
        and duplicate_feature_rows == 0
        and not missing_game_ids
        and not unexpected_game_ids
    ):

        print(
            "PASS: Game/team structure reconciles with final game data."
        )

    else:

        print(
            "FAIL: Game/team structural discrepancies detected."
        )


# =============================================================================
# 7. PRIOR-YEAR TRANSFER AUDIT
# =============================================================================

def audit_prior_year_transfers(
    transfers,
    games,
    year,
):
    """
    Quantify transfers in each season's processed transfer file whose
    transfer date falls before the calendar year of the game season.

    This is diagnostic only. It does not assume these records are wrong.
    """

    print_subsection(
        f"{year} - Prior-Year Transfer Audit"
    )

    season_start = pd.Timestamp(
        f"{year}-01-01",
        tz="UTC",
    )

    valid = transfers[
        transfers["transfer_date"].notna()
    ]

    prior_year = valid[
        valid["transfer_date"] < season_start
    ]

    print(
        f"Total transfer records: {len(transfers):,}"
    )

    print(
        f"Valid transfer dates: {len(valid):,}"
    )

    print(
        f"Transfers dated before {year}-01-01: "
        f"{len(prior_year):,}"
    )

    if prior_year.empty:

        print(
            "No prior-year-dated transfer records."
        )

        return

    print(
        f"Earliest prior-year transfer: "
        f"{prior_year['transfer_date'].min()}"
    )

    print(
        f"Latest prior-year transfer: "
        f"{prior_year['transfer_date'].max()}"
    )

    print(
        "\nPrior-year records by transfer_year:"
    )

    print(
        prior_year["transfer_year"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nPrior-year records by calendar year of transfer_date:"
    )

    print(
        prior_year["transfer_date"]
        .dt.year
        .value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nNOTE: This is not automatically a problem. "
        "The purpose is to determine whether the CFBD season file "
        "contains a historical transfer pool that should remain "
        "part of the season's cumulative features."
    )


# =============================================================================
# 8. SEASON SUMMARY
# =============================================================================

def build_season_summary(
    transfers,
    transfer_features,
    games,
    year,
):
    """Create a compact season-level summary."""

    return {
        "season": year,
        "games": len(games),
        "game_team_rows": len(transfer_features),
        "transfer_records": len(transfers),
        "valid_transfer_dates": transfers[
            "transfer_date"
        ].notna().sum(),
        "withdrawn_transfers": transfers[
            "is_withdrawn"
        ].sum(),
        "missing_origin": transfers[
            "origin"
        ].isna().sum(),
        "missing_destination": transfers[
            "destination"
        ].isna().sum(),
        "missing_rating": transfers[
            "rating"
        ].isna().sum(),
        "missing_stars": transfers[
            "stars"
        ].isna().sum(),
        "incoming_mean": transfer_features[
            "incoming_transfer_count"
        ].mean(),
        "incoming_max": transfer_features[
            "incoming_transfer_count"
        ].max(),
        "outgoing_mean": transfer_features[
            "outgoing_transfer_count"
        ].mean(),
        "outgoing_max": transfer_features[
            "outgoing_transfer_count"
        ].max(),
        "incoming_rated_mean": transfer_features[
            "incoming_rated_player_count"
        ].mean(),
        "outgoing_rated_mean": transfer_features[
            "outgoing_rated_player_count"
        ].mean(),
        "incoming_star_rated_mean": transfer_features[
            "incoming_star_rated_player_count"
        ].mean(),
        "outgoing_star_rated_mean": transfer_features[
            "outgoing_star_rated_player_count"
        ].mean(),
    }


def audit_cross_season_summary(
    season_summaries,
):
    """Print the cross-season summary table."""

    print_section(
        "CROSS-SEASON COMPARABILITY SUMMARY"
    )

    summary = pd.DataFrame(
        season_summaries
    ).set_index("season")

    display_columns = [
        "games",
        "transfer_records",
        "withdrawn_transfers",
        "missing_origin",
        "missing_destination",
        "missing_rating",
        "missing_stars",
        "incoming_mean",
        "incoming_max",
        "outgoing_mean",
        "outgoing_max",
        "incoming_rated_mean",
        "outgoing_rated_mean",
        "incoming_star_rated_mean",
        "outgoing_star_rated_mean",
    ]

    print(
        summary[
            display_columns
        ]
        .round(3)
        .to_string()
    )

    print(
        "\nNOTE: Large changes across seasons are diagnostic signals, "
        "not automatic errors. Transfer volume and source-data "
        "completeness can legitimately change over time."
    )


def audit_date_ranges(
    transfers,
    games,
    year,
):
    """
    Compare transfer and game date ranges.

    This is diagnostic only. It identifies:
    - overall transfer-date range
    - overall game-date range
    - transfers occurring after the final game
    """

    print_subsection(
        f"{year} - Date Range Audit"
    )

    valid_transfers = transfers[
        transfers["transfer_date"].notna()
    ]

    print(
        "Transfer date range:"
    )

    print(
        f"  {valid_transfers['transfer_date'].min()}"
        f" -> "
        f"{valid_transfers['transfer_date'].max()}"
    )

    print(
        "Game date range:"
    )

    print(
        f"  {games['startDate'].min()}"
        f" -> "
        f"{games['startDate'].max()}"
    )

    post_last_game = valid_transfers[
        valid_transfers["transfer_date"]
        > games["startDate"].max()
    ]

    print(
        f"Transfers after final game: "
        f"{len(post_last_game):,}"
    )

    if not post_last_game.empty:
        print(
            "NOTE: These records should be excluded from "
            "all game-level features occurring before their "
            "transfer dates."
        )

# =============================================================================
# 9. MAIN SEASON PROCESSING
# =============================================================================

def process_season(year):
    """Run all audits for one season."""

    print_section(
        f"TRANSFER GAME FEATURES AUDIT - {year}"
    )

    data = load_season_data(year)

    if data is None:
        return None

    transfers, games, transfer_features = data

    transfers = prepare_transfers(
        transfers,
        year,
    )

    games = prepare_games(
        games,
        year,
    )

    transfer_features = prepare_transfer_features(
        transfer_features,
        year,
    )

    print(
        f"Processed transfer rows: {len(transfers):,}"
    )

    print(
        f"Final game rows: {len(games):,}"
    )

    print(
        f"Transfer game feature rows: "
        f"{len(transfer_features):,}"
    )

    audit_star_encoding(
        transfers,
        year,
    )

    audit_date_ranges(
        transfers,
        games,
        year,
    )

    audit_prior_year_transfers(
        transfers,
        games,
        year,
    )

    audit_game_team_uniqueness(
        games,
        transfer_features,
        year,
    )

    audit_feature_reconciliation(
        transfers,
        transfer_features,
        year,
    )

    team_games = make_team_game_table(
        games
    )

    audit_temporal_behavior(
        transfers,
        team_games,
        transfer_features,
        year,
    )

    audit_first_game(
        transfers,
        games,
        transfer_features,
        year,
    )

    audit_large_counts(
        transfer_features,
        year,
    )

    return build_season_summary(
        transfers,
        transfer_features,
        games,
        year,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the complete cross-season audit."""

    print("=" * 80)
    print("TRANSFER GAME FEATURES - CROSS-SEASON AUDIT")
    print("=" * 80)

    print(
        f"Project root: {ROOT}"
    )

    print(
        f"Seasons: {min(YEARS)}-{max(YEARS)}"
    )

    print(
        "\nThis audit is READ-ONLY."
    )

    season_summaries = []

    for year in YEARS:

        summary = process_season(year)

        if summary is not None:
            season_summaries.append(
                summary
            )

    if season_summaries:

        audit_cross_season_summary(
            season_summaries
        )

    print_section(
        "AUDIT COMPLETE"
    )

    print(
        f"Audited seasons: "
        f"{len(season_summaries)}"
    )

    print(
        "\nNo files were modified by this audit."
    )


if __name__ == "__main__":
    main()