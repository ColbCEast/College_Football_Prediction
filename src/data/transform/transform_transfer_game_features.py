"""
Date-Aware Transfer Game Feature Transformation
================================================

Creates game-date-aware team-level transfer features while preventing
temporal leakage.

For each game/team combination, only transfer events occurring on or
before the game's start date are eligible to contribute to the features.

Incoming transfers are attributed to `destination`.
Outgoing transfers are attributed to `origin`.

Input:
    data/processed/player/portal/transfer_portal_{year}.csv
    data/processed/features/final/final_features_{year}.csv

Output:
    data/processed/player/portal/game_features/
        transfer_game_features_{year}.csv

The output contains one row per game/team combination and is intentionally
kept separate from final_features until the resulting features have been
audited.
"""

from pathlib import Path

import pandas as pd


# =============================================================================
# PATHS
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

OUTPUT_DIR = TRANSFER_DIR / "game_features"


# =============================================================================
# CONFIGURATION
# =============================================================================

YEARS = range(2015, 2026)

TRANSFER_FILE_TEMPLATE = "transfer_portal_{year}.csv"
GAME_FILE_TEMPLATE = "final_features_{year}.csv"
OUTPUT_FILE_TEMPLATE = "transfer_game_features_{year}.csv"


# =============================================================================
# REQUIRED COLUMNS
# =============================================================================

TRANSFER_REQUIRED_COLUMNS = {
    "season",
    "origin",
    "destination",
    "transfer_date",
    "transfer_year",
    "is_withdrawn",
    "rating",
    "stars",
}

GAME_REQUIRED_COLUMNS = {
    "gameId",
    "season",
    "startDate",
    "homeTeam",
    "awayTeam",
}


# =============================================================================
# TRANSFER FEATURE DEFINITIONS
# =============================================================================

STAR_VALUES = [2, 3, 4, 5]


# =============================================================================
# LOADING
# =============================================================================

def load_transfer_data(year: int) -> pd.DataFrame:
    """Load processed transfer data for a season."""

    path = TRANSFER_DIR / TRANSFER_FILE_TEMPLATE.format(year=year)

    if not path.exists():
        raise FileNotFoundError(
            f"Transfer file not found: {path}"
        )

    df = pd.read_csv(path)

    missing = TRANSFER_REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: "
            f"{sorted(missing)}"
        )

    return df


def load_game_data(year: int) -> pd.DataFrame:
    """Load game-level feature data for a season."""

    path = GAME_FEATURE_DIR / GAME_FILE_TEMPLATE.format(year=year)

    if not path.exists():
        raise FileNotFoundError(
            f"Game feature file not found: {path}"
        )

    df = pd.read_csv(path)

    missing = GAME_REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: "
            f"{sorted(missing)}"
        )

    return df


# =============================================================================
# PREPARATION
# =============================================================================

def prepare_transfer_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize transfer fields.

    No aggregation occurs here. The individual transfer event date is
    preserved so that temporal eligibility can be evaluated against each
    game's start date.
    """

    df = df.copy()

    df["transfer_date"] = pd.to_datetime(
        df["transfer_date"],
        errors="coerce",
    )

    df["rating"] = pd.to_numeric(
        df["rating"],
        errors="coerce",
    )

    df["stars"] = pd.to_numeric(
        df["stars"],
        errors="coerce",
    )

    # Normalize team names to strings while preserving missing values.
    for column in ["origin", "destination"]:
        df[column] = df[column].where(
            df[column].notna(),
            pd.NA,
        )

    # Explicit boolean conversion.
    df["is_withdrawn"] = (
        df["is_withdrawn"]
        .fillna(False)
        .astype(bool)
    )

    return df


def prepare_game_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize game dates and retain the fields required for aggregation."""

    df = df.copy()

    df["startDate"] = pd.to_datetime(
        df["startDate"],
        errors="coerce",
    )

    if df["startDate"].isna().any():
        missing_count = df["startDate"].isna().sum()

        raise ValueError(
            f"Found {missing_count:,} games with missing startDate."
        )

    return df[
        [
            "gameId",
            "season",
            "startDate",
            "homeTeam",
            "awayTeam",
        ]
    ].copy()


# =============================================================================
# GAME / TEAM TABLE
# =============================================================================

def create_team_game_table(
    games: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert one row per game into two rows per game:

        game + home team
        game + away team

    This provides the game-date/team grain required for transfer aggregation.
    """

    home = games[
        [
            "gameId",
            "season",
            "startDate",
            "homeTeam",
        ]
    ].rename(
        columns={
            "homeTeam": "team",
        }
    )

    home["homeAway"] = "home"

    away = games[
        [
            "gameId",
            "season",
            "startDate",
            "awayTeam",
        ]
    ].rename(
        columns={
            "awayTeam": "team",
        }
    )

    away["homeAway"] = "away"

    team_games = pd.concat(
        [home, away],
        ignore_index=True,
    )

    return team_games


# =============================================================================
# FEATURE CONSTRUCTION
# =============================================================================

def add_feature_components(
    df: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    """
    Add player-level components used by the team-level aggregation.

    prefix:
        incoming
        outgoing
    """

    df = df.copy()

    df[f"{prefix}_transfer_count"] = 1

    df[f"{prefix}_rated_player_count"] = (
        df["rating"].notna().astype(int)
    )

    df[f"{prefix}_rating_value"] = df["rating"]

    df[f"{prefix}_star_rated_player_count"] = (
        df["stars"].notna().astype(int)
    )

    for star in STAR_VALUES:
        df[f"{prefix}_{star}_star_count"] = (
            df["stars"] == star
        ).astype(int)

    return df


def aggregate_direction(
    team_games: pd.DataFrame,
    transfers: pd.DataFrame,
    direction: str,
) -> pd.DataFrame:
    """
    Aggregate either incoming or outgoing transfers using a game-date cutoff.

    direction:
        incoming -> destination
        outgoing -> origin

    A transfer is eligible for a game when:

        transfer_date <= startDate

    The implementation uses a cumulative as-of merge rather than creating
    every possible team-game/transfer combination.
    """

    if direction == "incoming":
        team_column = "destination"
    elif direction == "outgoing":
        team_column = "origin"
    else:
        raise ValueError(
            f"Invalid direction: {direction}"
        )

    # -------------------------------------------------------------------------
    # PREPARE TRANSFERS
    # -------------------------------------------------------------------------

    transfer_data = transfers.copy()

    transfer_data = transfer_data[
        ~transfer_data["is_withdrawn"]
    ].copy()

    transfer_data = transfer_data[
        transfer_data[team_column].notna()
        & transfer_data["transfer_date"].notna()
    ].copy()

    transfer_data = transfer_data.rename(
        columns={
            team_column: "team"
        }
    )

    # Add player-level feature components.
    transfer_data = add_feature_components(
        transfer_data,
        prefix=direction,
    )

    # Only retain fields needed for aggregation.
    transfer_columns = [
        "team",
        "transfer_date",
        f"{direction}_transfer_count",
        f"{direction}_rated_player_count",
        f"{direction}_rating_value",
        f"{direction}_star_rated_player_count",
    ]

    transfer_columns += [
        f"{direction}_{star}_star_count"
        for star in STAR_VALUES
    ]

    transfer_data = transfer_data[
        transfer_columns
    ].copy()

    # -------------------------------------------------------------------------
    # AGGREGATE TRANSFERS BY TEAM + DATE
    # -------------------------------------------------------------------------
    #
    # Multiple transfers can occur on the same date. Aggregate them first so
    # the subsequent cumulative calculation operates on one row per
    # team/date.
    #

    aggregation = {
        f"{direction}_transfer_count": "sum",
        f"{direction}_rated_player_count": "sum",
        f"{direction}_rating_value": ["sum", "count"],
        f"{direction}_star_rated_player_count": "sum",
    }

    for star in STAR_VALUES:
        aggregation[
            f"{direction}_{star}_star_count"
        ] = "sum"

    daily = (
        transfer_data
        .groupby(
            [
                "team",
                "transfer_date",
            ],
            as_index=False,
        )
        .agg(aggregation)
    )

    # Flatten aggregation columns.
    flattened_columns = []

    for column in daily.columns:

        if isinstance(column, tuple):
            parts = [
                str(part)
                for part in column
                if str(part) != ""
            ]

            flattened_columns.append(
                "_".join(parts).rstrip("_")
            )

        else:
            flattened_columns.append(column)

    daily.columns = flattened_columns

    # Rename daily aggregation columns.
    daily = daily.rename(
        columns={
            f"{direction}_transfer_count_sum":
                f"{direction}_transfer_count",

            f"{direction}_rated_player_count_sum":
                f"{direction}_rated_player_count",

            f"{direction}_rating_value_sum":
                f"{direction}_rating_sum",

            f"{direction}_rating_value_count":
                f"{direction}_rated_rating_count",

            f"{direction}_star_rated_player_count_sum":
                f"{direction}_star_rated_player_count",
        }
    )

    for star in STAR_VALUES:
        daily = daily.rename(
            columns={
                f"{direction}_{star}_star_count_sum":
                    f"{direction}_{star}_star_count"
            }
        )

    # -------------------------------------------------------------------------
    # SORT FOR CUMULATIVE CALCULATION
    # -------------------------------------------------------------------------

    daily = daily.sort_values(
        [
            "team",
            "transfer_date",
        ]
    ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # CUMULATIVE FEATURES
    # -------------------------------------------------------------------------

    cumulative_columns = [
        f"{direction}_transfer_count",
        f"{direction}_rated_player_count",
        f"{direction}_rating_sum",
        f"{direction}_rated_rating_count",
        f"{direction}_star_rated_player_count",
    ]

    cumulative_columns += [
        f"{direction}_{star}_star_count"
        for star in STAR_VALUES
    ]

    for column in cumulative_columns:
        daily[column] = (
            daily
            .groupby("team")[column]
            .cumsum()
        )

    # -------------------------------------------------------------------------
    # CALCULATE CUMULATIVE RATING MEAN
    # -------------------------------------------------------------------------

    daily[
        f"{direction}_rating_mean"
    ] = (
        daily[f"{direction}_rating_sum"]
        / daily[f"{direction}_rated_rating_count"]
    )

    # -------------------------------------------------------------------------
    # PREPARE GAME/TEAM DATA
    # -------------------------------------------------------------------------

    game_data = team_games[
        [
            "gameId",
            "season",
            "startDate",
            "team",
            "homeAway",
        ]
    ].copy()

    # merge_asof requires the actual merge keys to be sorted.
    #
    # The `on` key must be globally sorted, with `team` used as the
    # secondary ordering variable for the `by` grouping.
    game_data = game_data.sort_values(
        [
            "startDate",
            "team",
        ]
    ).reset_index(drop=True)

    daily = daily.sort_values(
        [
            "transfer_date",
            "team",
        ]
    ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # DATE-AWARE AS-OF MERGE
    # -------------------------------------------------------------------------
    #
    # transfer_date <= startDate
    #
    # merge_asof therefore selects the latest cumulative transfer state
    # available by the time of each game.
    #

    result = pd.merge_asof(
        game_data,
        daily,
        left_on="startDate",
        right_on="transfer_date",
        by="team",
        direction="backward",
        allow_exact_matches=True,
    )

    # -------------------------------------------------------------------------
    # RESTORE ZERO / MISSING SEMANTICS
    # -------------------------------------------------------------------------

    count_columns = [
        f"{direction}_transfer_count",
        f"{direction}_rated_player_count",
        f"{direction}_star_rated_player_count",
    ]

    count_columns += [
        f"{direction}_{star}_star_count"
        for star in STAR_VALUES
    ]

    for column in count_columns:
        result[column] = (
            result[column]
            .fillna(0)
            .astype(int)
        )

    # No rated players before the game means:
    #
    #     rating_sum = 0
    #     rating_mean = NaN
    #
    # rather than inventing a rating of zero.
    result[
        f"{direction}_rating_sum"
    ] = result[
        f"{direction}_rating_sum"
    ].fillna(0)

    result[
        f"{direction}_rating_mean"
    ] = result[
        f"{direction}_rating_mean"
    ].where(
        result[
            f"{direction}_rated_rating_count"
        ].fillna(0) > 0
    )

    # This helper column is no longer needed in the final output.
    result = result.drop(
        columns=[
            f"{direction}_rated_rating_count"
        ],
        errors="ignore",
    )

    # Transfer date is useful internally for the as-of merge but should not
    # become a modeling feature.
    result = result.drop(
        columns=["transfer_date"],
        errors="ignore",
    )

    return result


def create_empty_direction_features(
    team_games: pd.DataFrame,
    direction: str,
) -> pd.DataFrame:
    """Create an all-missing/all-zero direction table when no transfers exist."""

    result = team_games.copy()

    count_columns = [
        f"{direction}_transfer_count",
        f"{direction}_rated_player_count",
        f"{direction}_star_rated_player_count",
    ]

    count_columns += [
        f"{direction}_{star}_star_count"
        for star in STAR_VALUES
    ]

    for column in count_columns:
        result[column] = 0

    result[f"{direction}_rating_sum"] = 0.0
    result[f"{direction}_rating_mean"] = pd.NA

    return result


def build_transfer_game_features(
    team_games: pd.DataFrame,
    transfers: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the complete incoming + outgoing transfer feature table.
    """

    incoming = aggregate_direction(
        team_games,
        transfers,
        direction="incoming",
    )

    outgoing = aggregate_direction(
        team_games,
        transfers,
        direction="outgoing",
    )

    outgoing_features = [
        column
        for column in outgoing.columns
        if column.startswith("outgoing_")
    ]

    outgoing_key = outgoing[
        [
            "gameId",
            "team",
        ]
        + outgoing_features
    ].copy()

    result = incoming.merge(
        outgoing_key,
        on=["gameId", "team"],
        how="left",
    )

    return result


# =============================================================================
# AUDITS
# =============================================================================

def audit_transfer_dates(
    transfers: pd.DataFrame,
    year: int,
) -> None:
    """Audit transfer-date coverage."""

    print("\nTransfer date audit")
    print("-" * 80)

    total = len(transfers)
    valid_dates = transfers["transfer_date"].notna().sum()

    print(f"Transfer rows: {total:,}")
    print(f"Valid transfer dates: {valid_dates:,}")
    print(
        f"Missing transfer dates: "
        f"{total - valid_dates:,}"
    )

    if valid_dates > 0:
        print(
            f"Minimum transfer date: "
            f"{transfers['transfer_date'].min()}"
        )
        print(
            f"Maximum transfer date: "
            f"{transfers['transfer_date'].max()}"
        )


def audit_game_team_uniqueness(
    result: pd.DataFrame,
) -> None:
    """Verify exactly one row per game/team combination."""

    duplicate_mask = result.duplicated(
        subset=["gameId", "team"],
        keep=False,
    )

    duplicate_count = duplicate_mask.sum()

    print("\nGame/team uniqueness audit")
    print("-" * 80)
    print(f"Rows: {len(result):,}")
    print(
        f"Duplicate game/team rows: "
        f"{duplicate_count:,}"
    )

    if duplicate_count > 0:
        duplicate_examples = result[
            duplicate_mask
        ][
            ["gameId", "team"]
        ].head(10)

        print("\nDuplicate examples:")
        print(duplicate_examples.to_string(index=False))

        raise AssertionError(
            "Duplicate game/team rows detected."
        )

    print("PASS: One row per game/team.")


def audit_expected_team_rows(
    games: pd.DataFrame,
    result: pd.DataFrame,
) -> None:
    """Verify exactly two team rows per game."""

    expected = len(games) * 2
    actual = len(result)

    print("\nExpected team-row audit")
    print("-" * 80)
    print(f"Games: {len(games):,}")
    print(f"Expected team rows: {expected:,}")
    print(f"Actual team rows: {actual:,}")

    if actual != expected:
        raise AssertionError(
            "Expected exactly two team rows per game."
        )

    print("PASS: Exactly two team rows per game.")


def audit_temporal_cutoff(
    team_games: pd.DataFrame,
    transfers: pd.DataFrame,
    result: pd.DataFrame,
) -> None:
    """
    Verify that generated transfer features do not include transfers
    occurring after the corresponding game start date.

    This audit distinguishes between:

        1. Future transfers that exist in the source data.
        2. Future transfers that are incorrectly represented in the
           generated game-level features.

    A transfer is eligible only when:

        transfer_date <= startDate

    The generated cumulative features are therefore expected to include
    only transfers available by the game's start date.
    """

    print("\nTemporal leakage audit")
    print("-" * 80)

    valid_transfers = transfers[
        transfers["transfer_date"].notna()
        & ~transfers["is_withdrawn"]
    ].copy()

    for direction, team_column in [
        ("incoming", "destination"),
        ("outgoing", "origin"),
    ]:

        direction_transfers = valid_transfers[
            valid_transfers[team_column].notna()
        ].copy()

        direction_transfers = direction_transfers.rename(
            columns={
                team_column: "team",
            }
        )

        # ---------------------------------------------------------------------
        # IDENTIFY FUTURE TRANSFERS
        # ---------------------------------------------------------------------

        check = team_games.merge(
            direction_transfers[
                [
                    "team",
                    "transfer_date",
                ]
            ],
            on="team",
            how="inner",
        )

        future = check[
            check["transfer_date"] > check["startDate"]
        ]

        print(
            f"{direction.capitalize()} transfer/team "
            f"comparisons: {len(check):,}"
        )

        print(
            f"{direction.capitalize()} future comparisons: "
            f"{len(future):,}"
        )

        if len(future) == 0:
            print(
                f"PASS: No future {direction} transfers "
                "exist relative to game dates."
            )
            continue

        print(
            f"NOTE: {len(future):,} future {direction} "
            "transfer/game comparisons exist in the source data."
        )

        # ---------------------------------------------------------------------
        # VERIFY FUTURE TRANSFERS ARE NOT REPRESENTED IN FEATURES
        # ---------------------------------------------------------------------
        #
        # Rather than treating the existence of a future transfer as
        # leakage, verify that the generated cumulative feature does not
        # change until the transfer becomes eligible.
        #

        feature_column = f"{direction}_transfer_count"

        violations = []

        for _, future_row in future.iterrows():

            team = future_row["team"]
            game_date = future_row["startDate"]
            future_transfer_date = future_row["transfer_date"]

            # Find the generated feature for this game/team.
            matching_result = result[
                (result["team"] == team)
                & (result["startDate"] == game_date)
            ]

            if matching_result.empty:
                continue

            # Count all valid transfers for this team that were eligible
            # by the game date.
            eligible_count = len(
                direction_transfers[
                    (direction_transfers["team"] == team)
                    & (
                        direction_transfers["transfer_date"]
                        <= game_date
                    )
                ]
            )

            generated_count = matching_result.iloc[0][
                feature_column
            ]

            if generated_count != eligible_count:
                violations.append(
                    {
                        "team": team,
                        "startDate": game_date,
                        "future_transfer_date": future_transfer_date,
                        "expected_count": eligible_count,
                        "generated_count": generated_count,
                    }
                )

        # ---------------------------------------------------------------------
        # REPORT
        # ---------------------------------------------------------------------

        if violations:

            print(
                f"\n{direction.capitalize()} temporal violations:"
            )

            violations_df = pd.DataFrame(
                violations
            )

            print(
                violations_df.head(10).to_string(
                    index=False
                )
            )

            raise AssertionError(
                f"TEMPORAL LEAKAGE DETECTED in {direction} "
                "transfer aggregation: generated feature counts "
                "do not match the number of transfers eligible by "
                "game date."
            )

        print(
            f"PASS: Future {direction} transfers are not included "
            "in generated game-level features."
        )


def audit_feature_values(
    result: pd.DataFrame,
) -> None:
    """Check basic validity of generated transfer features."""

    print("\nFeature-value audit")
    print("-" * 80)

    count_columns = [
        "incoming_transfer_count",
        "incoming_rated_player_count",
        "incoming_star_rated_player_count",
        "outgoing_transfer_count",
        "outgoing_rated_player_count",
        "outgoing_star_rated_player_count",
    ]

    for star in STAR_VALUES:
        count_columns.append(
            f"incoming_{star}_star_count"
        )
        count_columns.append(
            f"outgoing_{star}_star_count"
        )

    for column in count_columns:

        if (result[column] < 0).any():
            raise AssertionError(
                f"Negative count found in {column}."
            )

        if (result[column] % 1 != 0).any():
            raise AssertionError(
                f"Non-integer count found in {column}."
            )

    # Rating sums may be zero but should never be negative.
    for column in [
        "incoming_rating_sum",
        "outgoing_rating_sum",
    ]:

        if (result[column].dropna() < 0).any():
            raise AssertionError(
                f"Negative rating value found in {column}."
            )

    print("PASS: Feature values satisfy basic validity checks.")


def audit_direction_consistency(
    result: pd.DataFrame,
) -> None:
    """
    Check logical relationships among count-based features.

    The rating and star fields are treated as separate sources of
    player-level information because either field may be missing
    independently.

    Required relationships:

        rated-player count <= total transfer count
        star-rated-player count <= total transfer count

    The individual star-count columns are expected to partition
    star-rated players only if all non-missing star values are represented
    by STAR_VALUES.
    """

    print("\nDirection consistency audit")
    print("-" * 80)

    for direction in ["incoming", "outgoing"]:

        transfer_count = result[
            f"{direction}_transfer_count"
        ]

        rated_count = result[
            f"{direction}_rated_player_count"
        ]

        star_rated_count = result[
            f"{direction}_star_rated_player_count"
        ]

        # ---------------------------------------------------------------------
        # RATED PLAYERS CANNOT EXCEED TOTAL TRANSFERS
        # ---------------------------------------------------------------------

        if (rated_count > transfer_count).any():
            raise AssertionError(
                f"{direction}: rated-player count exceeds "
                "transfer count."
            )

        # ---------------------------------------------------------------------
        # STAR-RATED PLAYERS CANNOT EXCEED TOTAL TRANSFERS
        # ---------------------------------------------------------------------

        if (star_rated_count > transfer_count).any():
            raise AssertionError(
                f"{direction}: star-rated-player count exceeds "
                "transfer count."
            )

        # ---------------------------------------------------------------------
        # INDIVIDUAL STAR COUNTS
        # ---------------------------------------------------------------------
        #
        # Diagnose rather than immediately fail if the individual star
        # categories do not sum to star-rated-player count.
        #
        # This can occur if the source contains a non-missing star value
        # outside STAR_VALUES.
        #

        star_count_columns = [
            f"{direction}_{star}_star_count"
            for star in STAR_VALUES
        ]

        star_count_sum = result[
            star_count_columns
        ].sum(axis=1)

        mismatch = (
            star_count_sum != star_rated_count
        )

        if mismatch.any():

            print(
                f"\nWARNING: {direction} star-count totals do not "
                "equal star-rated-player counts."
            )

            print(
                f"Mismatched game/team rows: "
                f"{mismatch.sum():,}"
            )

            print("\nMismatch examples:")

            examples = result.loc[
                mismatch,
                [
                    "gameId",
                    "team",
                    f"{direction}_transfer_count",
                    f"{direction}_rated_player_count",
                    f"{direction}_star_rated_player_count",
                ]
                + star_count_columns
            ].copy()

            examples["star_count_sum"] = (
                examples[star_count_columns].sum(axis=1)
            )

            examples["unaccounted_star_count"] = (
                examples[
                    f"{direction}_star_rated_player_count"
                ]
                - examples["star_count_sum"]
            )

            print(
                examples.head(10).to_string(index=False)
            )

            print(
                "\nNOTE:"
                f" {direction}_star_rated_player_count represents "
                "all non-missing star values, while the individual "
                "star columns currently represent only the configured "
                f"values: {STAR_VALUES}."
            )

    print(
        "PASS: Incoming/outgoing count relationships are valid."
    )


# =============================================================================
# SEASON PROCESSING
# =============================================================================

def process_season(year: int) -> None:
    """Process one season."""

    print("\n" + "=" * 80)
    print(
        f"PROCESSING DATE-AWARE TRANSFER FEATURES: {year}"
    )
    print("=" * 80)

    transfers = load_transfer_data(year)
    games = load_game_data(year)

    print(
        f"Processed transfer rows: {len(transfers):,}"
    )
    print(
        f"Game rows: {len(games):,}"
    )

    transfers = prepare_transfer_data(transfers)
    games = prepare_game_data(games)

    # -------------------------------------------------------------------------
    # BASIC DATE INFORMATION
    # -------------------------------------------------------------------------

    audit_transfer_dates(
        transfers,
        year,
    )

    print("\nGame date coverage")
    print("-" * 80)
    print(
        f"Minimum game start date: "
        f"{games['startDate'].min()}"
    )
    print(
        f"Maximum game start date: "
        f"{games['startDate'].max()}"
    )

    # -------------------------------------------------------------------------
    # CREATE GAME/TEAM GRAIN
    # -------------------------------------------------------------------------

    team_games = create_team_game_table(
        games
    )

    # -------------------------------------------------------------------------
    # BUILD FEATURES
    # -------------------------------------------------------------------------

    result = build_transfer_game_features(
        team_games,
        transfers,
    )

    # -------------------------------------------------------------------------
    # TEMPORAL AUDIT AFTER AGGREGATION
    # -------------------------------------------------------------------------

    audit_temporal_cutoff(
        team_games,
        transfers,
        result,
    )

    # -------------------------------------------------------------------------
    # AUDITS
    # -------------------------------------------------------------------------

    audit_game_team_uniqueness(
        result
    )

    audit_expected_team_rows(
        games,
        result,
    )

    audit_feature_values(
        result
    )

    audit_direction_consistency(
        result
    )

    # -------------------------------------------------------------------------
    # COLUMN ORDER
    # -------------------------------------------------------------------------

    feature_columns = [
        "incoming_transfer_count",
        "incoming_rated_player_count",
        "incoming_rating_sum",
        "incoming_rating_mean",
        "incoming_star_rated_player_count",
    ]

    feature_columns += [
        f"incoming_{star}_star_count"
        for star in STAR_VALUES
    ]

    feature_columns += [
        "outgoing_transfer_count",
        "outgoing_rated_player_count",
        "outgoing_rating_sum",
        "outgoing_rating_mean",
        "outgoing_star_rated_player_count",
    ]

    feature_columns += [
        f"outgoing_{star}_star_count"
        for star in STAR_VALUES
    ]

    result = result[
        [
            "gameId",
            "season",
            "startDate",
            "team",
            "homeAway",
        ]
        + feature_columns
    ]

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / OUTPUT_FILE_TEMPLATE.format(year=year)
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print("\nOutput")
    print("-" * 80)
    print(f"Saved: {output_path}")
    print(f"Rows: {len(result):,}")
    print(f"Columns: {len(result.columns):,}")

    print("\nFeature columns")
    print("-" * 80)

    for column in feature_columns:
        print(f"  {column}")

    print("\nFeature summary")
    print("-" * 80)

    print(
        result[feature_columns]
        .describe()
        .transpose()
        .to_string()
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 80)
    print("DATE-AWARE TRANSFER GAME FEATURE TRANSFORMATION")
    print("=" * 80)

    print(f"Root: {ROOT}")
    print(f"Transfer directory: {TRANSFER_DIR}")
    print(f"Game feature directory: {GAME_FEATURE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(
        f"Years: {min(YEARS)}-{max(YEARS)}"
    )

    for year in YEARS:

        try:
            process_season(year)

        except FileNotFoundError as exc:
            print(
                f"\nSKIPPING {year}: {exc}"
            )

        except Exception as exc:
            print(
                f"\nFAILED {year}: "
                f"{type(exc).__name__}: {exc}"
            )
            raise

    print("\n" + "=" * 80)
    print(
        "DATE-AWARE TRANSFER FEATURE TRANSFORMATION COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()