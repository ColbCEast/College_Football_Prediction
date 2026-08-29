"""
Create compact matchup features for logistic regression.

Purpose
-------
Transform the validated logistic enhanced feature datasets into a small,
interpretable set of home-vs-away matchup features.

The resulting datasets contain four nested feature sets:

    Model 1: Compact baseline
    Model 2: Compact baseline + recent form
    Model 3: Compact baseline + recent form + SOS
    Model 4: Model 3 + additional matchup dimensions

All matchup features are oriented so that:

    Higher value = better for the home team

The source enhanced datasets are NOT modified.

Input
-----
data/processed/logistic_enhanced_features/
    logistic_features_{year}.csv

Output
------
data/processed/logistic_matchup_features/
    logistic_matchup_features_{year}.csv

Each output file contains:
    - identifiers / target needed for modeling
    - Model 1 features
    - Model 2 features
    - Model 3 features
    - Model 4 features

Model 4 adds seven features intended to capture dimensions not fully
represented by Models 1–3:

    - turnovers
    - third-down efficiency
    - sacks
    - completion percentage
    - total offensive production
    - possession time
    - penalty yards

The purpose of Model 4 is NOT to maximize feature count. It is a controlled
extension designed to test whether these additional football dimensions
provide meaningful out-of-sample predictive information beyond Model 3.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "logistic_enhanced_features"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "logistic_matchup_features"
)

YEARS = list(range(2015, 2026))

TARGET_COLUMN = "win_home"

GAME_ID_CANDIDATES = [
    "id",
    "gameId",
    "game_id",
]

# Minimum acceptable number of rows for an output dataset.
MIN_ROWS = 100


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

# -------------------------------------------------------------------------
# MODEL 1 — COMPACT BASELINE
# -------------------------------------------------------------------------
#
# These features describe the relative strength of the two teams entering
# the game.
#
# Every feature is oriented:
#
#     positive -> favors home team
#     negative -> favors away team
#
# -------------------------------------------------------------------------

MODEL_1_FEATURES = {
    "winPctDiff": (
        "winPctBefore_home",
        "winPctBefore_away",
        1,
    ),

    "pointDifferentialAvgDiff": (
        "pointDifferentialAvgBefore_home",
        "pointDifferentialAvgBefore_away",
        1,
    ),

    "pointsForAvgDiff": (
        "pointsForAvgBefore_home",
        "pointsForAvgBefore_away",
        1,
    ),

    # Higher points allowed is worse.
    #
    # Therefore:
    #
    #     away points allowed - home points allowed
    #
    # Positive -> home has the better defense.
    "pointsAgainstAvgDiff": (
        "pointsAgainstAvgBefore_home",
        "pointsAgainstAvgBefore_away",
        -1,
    ),

    "yardsPerPassAttemptDiff": (
        "yardsPerPassAttemptBefore_home",
        "yardsPerPassAttemptBefore_away",
        1,
    ),

    "yardsPerRushAttemptDiff": (
        "yardsPerRushAttemptBefore_home",
        "yardsPerRushAttemptBefore_away",
        1,
    ),
}


# -------------------------------------------------------------------------
# MODEL 2 — ADD RECENT FORM
# -------------------------------------------------------------------------
#
# These measure whether the home team is currently trending better or worse
# than the away team.
# -------------------------------------------------------------------------

MODEL_2_FEATURES = {
    "pointsForTrendDiff": (
        "pointsForTrend_home",
        "pointsForTrend_away",
        1,
    ),

    # Higher points-against trend is worse.
    #
    # Positive -> away team is allowing more points recently than home team.
    "pointsAgainstTrendDiff": (
        "pointsAgainstTrend_home",
        "pointsAgainstTrend_away",
        -1,
    ),

    "pointDifferentialTrendDiff": (
        "pointDifferentialTrend_home",
        "pointDifferentialTrend_away",
        1,
    ),

    "totalYardsTrendDiff": (
        "totalYardsTrend_home",
        "totalYardsTrend_away",
        1,
    ),

    "netPassingYardsTrendDiff": (
        "netPassingYardsTrend_home",
        "netPassingYardsTrend_away",
        1,
    ),

    "winPctTrendDiff": (
        "winPctTrend_home",
        "winPctTrend_away",
        1,
    ),
}


# -------------------------------------------------------------------------
# MODEL 3 — ADD PRIOR STRENGTH OF SCHEDULE
# -------------------------------------------------------------------------
#
# These features adjust team strength for the quality of opponents faced
# before the current game.
# -------------------------------------------------------------------------

MODEL_3_FEATURES = {
    "priorSOSWinPctDiff": (
        "priorSOSWinPct_home",
        "priorSOSWinPct_away",
        1,
    ),

    "priorSOSPointDiffDiff": (
        "priorSOSPointDiff_home",
        "priorSOSPointDiff_away",
        1,
    ),
}


# -------------------------------------------------------------------------
# MODEL 4 — ADDITIONAL KEY FOOTBALL DIMENSIONS
# -------------------------------------------------------------------------
#
# Model 4 intentionally adds only a small number of carefully selected
# features rather than opening the model to the full feature set.
#
# These features are intended to capture information that Model 3 does not
# fully represent:
#
#   turnovers:
#       ball security / turnover tendency
#
#   third-down percentage:
#       ability to sustain drives and convert key downs
#
#   sacks:
#       pass-rush / pass-protection performance
#
#   completion percentage:
#       passing efficiency / offensive style
#
#   total yards:
#       broader offensive production beyond yards per play
#
#   possession seconds:
#       time-of-possession / drive-control style
#
#   penalty yards:
#       discipline / hidden yardage
#
# All are oriented so:
#
#     positive -> better for home team
# -------------------------------------------------------------------------

MODEL_4_FEATURES = {
    # Higher turnover averages are worse.
    #
    # Therefore:
    #
    #     away turnovers - home turnovers
    #
    # Positive -> home has the better turnover profile.
    "turnoversAvgDiff": (
        "turnoversAvgBefore_home",
        "turnoversAvgBefore_away",
        -1,
    ),

    # Higher third-down conversion percentage is better.
    "thirdDownPctDiff": (
        "thirdDownPctBefore_home",
        "thirdDownPctBefore_away",
        1,
    ),

    # Higher sack averages are treated as better defensive/pass-rush
    # production.
    "sacksAvgDiff": (
        "sacksAvgBefore_home",
        "sacksAvgBefore_away",
        1,
    ),

    # Higher completion percentage is better.
    "completionPctDiff": (
        "completionPctBefore_home",
        "completionPctBefore_away",
        1,
    ),

    # Higher total offensive yards is better.
    "totalYardsAvgDiff": (
        "totalYardsAvgBefore_home",
        "totalYardsAvgBefore_away",
        1,
    ),

    # Higher possession time is treated as a distinct offensive/drive-control
    # characteristic.
    "possessionSecondsAvgDiff": (
        "possessionSecondsAvgBefore_home",
        "possessionSecondsAvgBefore_away",
        1,
    ),

    # Higher penalty yards are worse.
    #
    # Therefore:
    #
    #     away penalty yards - home penalty yards
    #
    # Positive -> home is more disciplined.
    "penaltyYardsAvgDiff": (
        "penaltyYardsAvgBefore_home",
        "penaltyYardsAvgBefore_away",
        -1,
    ),
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_section(title):
    """Print a standardized section header."""

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def fail(message):
    """Raise a validation error with a clear message."""

    raise ValueError(
        f"\nVALIDATION FAILED:\n{message}"
    )


def find_game_id_column(df):
    """
    Identify the game identifier column.

    The enhanced feature files should contain gameId, but we allow a small
    number of known alternatives.
    """

    for column in GAME_ID_CANDIDATES:
        if column in df.columns:
            return column

    fail(
        "Could not identify a game identifier column. "
        f"Expected one of: {GAME_ID_CANDIDATES}"
    )


def validate_input_structure(df, year):
    """Validate the source enhanced feature dataset."""

    print_section(
        f"VALIDATING INPUT STRUCTURE — {year}"
    )

    if df.empty:
        fail(
            f"{year}: Input dataset is empty."
        )

    if TARGET_COLUMN not in df.columns:
        fail(
            f"{year}: Target column '{TARGET_COLUMN}' "
            "is missing from enhanced features."
        )

    game_id = find_game_id_column(df)

    if df[game_id].isna().any():
        fail(
            f"{year}: Game ID contains missing values."
        )

    if df[game_id].duplicated().any():

        duplicate_count = int(
            df[game_id].duplicated().sum()
        )

        fail(
            f"{year}: Found {duplicate_count} duplicate game IDs."
        )

    target_values = set(
        df[TARGET_COLUMN]
        .dropna()
        .unique()
    )

    if not target_values.issubset({0, 1}):

        fail(
            f"{year}: Target contains unexpected values: "
            f"{sorted(target_values)}"
        )

    print(
        f"Input rows:       {len(df):,}"
    )

    print(
        f"Input columns:    {len(df.columns):,}"
    )

    print(
        f"Game ID column:   {game_id}"
    )

    print(
        f"Target:           {TARGET_COLUMN}"
    )

    print(
        "Input structure:  VALID"
    )

    return game_id


def validate_required_columns(
    df,
    feature_definitions,
    model_name,
    year,
):
    """
    Ensure all source columns required by a feature set are present.
    """

    missing = []

    for (
        output_name,
        (
            home_column,
            away_column,
            orientation,
        ),
    ) in feature_definitions.items():

        if home_column not in df.columns:
            missing.append(home_column)

        if away_column not in df.columns:
            missing.append(away_column)

    if missing:

        missing = sorted(
            set(missing)
        )

        fail(
            f"{year}: {model_name} is missing required source columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    print(
        f"{model_name} source columns: VALID"
    )


def create_difference_feature(
    df,
    home_column,
    away_column,
    orientation,
):
    """
    Create a home-vs-away difference.

    orientation = +1:
        home - away

    orientation = -1:
        away - home

    The latter is used for metrics where lower values are better.
    """

    home = pd.to_numeric(
        df[home_column],
        errors="coerce",
    )

    away = pd.to_numeric(
        df[away_column],
        errors="coerce",
    )

    return orientation * (
        home - away
    )


def create_feature_set(
    df,
    feature_definitions,
    model_name,
    year,
):
    """Create one compact matchup feature set."""

    print_section(
        f"CREATING {model_name.upper()} FEATURES — {year}"
    )

    validate_required_columns(
        df=df,
        feature_definitions=feature_definitions,
        model_name=model_name,
        year=year,
    )

    result = pd.DataFrame(
        index=df.index
    )

    for (
        output_name,
        (
            home_column,
            away_column,
            orientation,
        ),
    ) in feature_definitions.items():

        result[output_name] = (
            create_difference_feature(
                df=df,
                home_column=home_column,
                away_column=away_column,
                orientation=orientation,
            )
        )

    print(
        f"Created {len(result.columns)} matchup features:"
    )

    for column in result.columns:
        print(
            f"  {column}"
        )

    return result


def validate_feature_values(
    feature_df,
    feature_definitions,
    model_name,
    year,
):
    """Validate generated matchup features."""

    print_section(
        f"VALIDATING {model_name.upper()} FEATURES — {year}"
    )

    expected_columns = list(
        feature_definitions.keys()
    )

    actual_columns = list(
        feature_df.columns
    )

    if actual_columns != expected_columns:

        fail(
            f"{year}: {model_name} feature columns do not match "
            "expected order."
        )

    for column in expected_columns:

        values = feature_df[column]

        if not pd.api.types.is_numeric_dtype(
            values
        ):

            fail(
                f"{year}: {column} is not numeric."
            )

        finite_values = values[
            values.notna()
        ]

        if not np.isfinite(
            finite_values.to_numpy()
        ).all():

            fail(
                f"{year}: {column} contains infinite values."
            )

    print(
        f"Feature count:     {len(expected_columns)}"
    )

    print(
        "Numeric features:  VALID"
    )

    print(
        "Finite values:     VALID"
    )


def validate_orientation(
    source_df,
    matchup_df,
    feature_definitions,
    year,
):
    """
    Verify that every difference feature has the intended orientation.

    For every row where both source values exist, the generated feature must
    equal the specified home-minus-away or away-minus-home calculation.
    """

    print_section(
        f"VALIDATING FEATURE ORIENTATION — {year}"
    )

    for (
        output_name,
        (
            home_column,
            away_column,
            orientation,
        ),
    ) in feature_definitions.items():

        expected = create_difference_feature(
            df=source_df,
            home_column=home_column,
            away_column=away_column,
            orientation=orientation,
        )

        actual = matchup_df[
            output_name
        ]

        valid = (
            expected.notna()
            & actual.notna()
        )

        if valid.sum() == 0:
            continue

        matches = np.isclose(
            expected.loc[valid].to_numpy(),
            actual.loc[valid].to_numpy(),
            rtol=1e-10,
            atol=1e-10,
        )

        if not matches.all():

            fail(
                f"{year}: Orientation validation failed for "
                f"{output_name}."
            )

    print(
        "Home/away orientation: VALID"
    )


def validate_no_target_leakage(
    feature_columns,
    year,
):
    """
    Ensure target and obvious post-game variables are not accidentally
    included as predictors.
    """

    forbidden_exact = {
        TARGET_COLUMN,
        "points",
        "homePoints",
        "awayPoints",
        "pointsFor",
        "pointsAgainst",
        "pointDifferential",
        "win",
    }

    leaked = (
        set(feature_columns)
        & forbidden_exact
    )

    if leaked:

        fail(
            f"{year}: Potential target/game-outcome leakage detected: "
            f"{sorted(leaked)}"
        )

    print(
        "Target leakage check: VALID"
    )


def validate_missingness(
    source_df,
    matchup_df,
    feature_definitions,
    year,
):
    """
    Validate that matchup missingness is exactly what would be expected
    from the underlying home/away source columns.

    A matchup feature is missing only when either source value is missing.
    """

    print_section(
        f"VALIDATING FEATURE MISSINGNESS — {year}"
    )

    for (
        output_name,
        (
            home_column,
            away_column,
            orientation,
        ),
    ) in feature_definitions.items():

        expected_missing = (
            source_df[home_column].isna()
            | source_df[away_column].isna()
        )

        actual_missing = (
            matchup_df[output_name].isna()
        )

        if not expected_missing.equals(
            actual_missing
        ):

            fail(
                f"{year}: Missingness mismatch for "
                f"{output_name}."
            )

    print(
        "Missingness construction: VALID"
    )


def create_model_columns():
    """
    Return the nested model feature lists.

    Model 4 contains all Model 3 features plus the seven additional
    Model 4 features.
    """

    model_1 = list(
        MODEL_1_FEATURES.keys()
    )

    model_2 = (
        model_1
        + list(
            MODEL_2_FEATURES.keys()
        )
    )

    model_3 = (
        model_2
        + list(
            MODEL_3_FEATURES.keys()
        )
    )

    model_4 = (
        model_3
        + list(
            MODEL_4_FEATURES.keys()
        )
    )

    return {
        "model_1": model_1,
        "model_2": model_2,
        "model_3": model_3,
        "model_4": model_4,
    }


def validate_model_nesting(year):
    """Verify that each model is a strict extension of the previous model."""

    print_section(
        f"VALIDATING MODEL FEATURE NESTING — {year}"
    )

    model_columns = create_model_columns()

    model_1 = model_columns["model_1"]
    model_2 = model_columns["model_2"]
    model_3 = model_columns["model_3"]
    model_4 = model_columns["model_4"]

    if not set(model_1).issubset(
        model_2
    ):

        fail(
            f"{year}: Model 1 features are not contained in Model 2."
        )

    if not set(model_2).issubset(
        model_3
    ):

        fail(
            f"{year}: Model 2 features are not contained in Model 3."
        )

    if not set(model_3).issubset(
        model_4
    ):

        fail(
            f"{year}: Model 3 features are not contained in Model 4."
        )

    print(
        f"Model 1 features: {len(model_1)}"
    )

    print(
        f"Model 2 features: {len(model_2)}"
    )

    print(
        f"Model 3 features: {len(model_3)}"
    )

    print(
        f"Model 4 features: {len(model_4)}"
    )

    print(
        "Feature nesting:   VALID"
    )

    return model_columns


def build_output_dataset(
    source_df,
    game_id,
    matchup_features,
    model_columns,
    year,
):
    """Build the final compact modeling dataset."""

    print_section(
        f"BUILDING OUTPUT DATASET — {year}"
    )

    output = pd.DataFrame()

    output[game_id] = source_df[
        game_id
    ]

    output["season"] = source_df[
        "season"
    ]

    # Keep identifiers useful for inspection / evaluation if available.
    optional_columns = [
        "week",
        "startDate",
        "homeTeam",
        "awayTeam",
    ]

    for column in optional_columns:

        if column in source_df.columns:
            output[column] = source_df[
                column
            ]

    output[TARGET_COLUMN] = source_df[
        TARGET_COLUMN
    ]

    # Add all Model 4 features once.
    #
    # Because Models 1–4 are nested, the Model 4 dataset contains every
    # feature required by all four models.
    for column in model_columns["model_4"]:

        output[column] = matchup_features[
            column
        ]

    if len(output) != len(source_df):

        fail(
            f"{year}: Output row count changed."
        )

    if output[game_id].duplicated().any():

        fail(
            f"{year}: Duplicate game IDs introduced."
        )

    if not output[game_id].equals(
        source_df[game_id]
    ):

        fail(
            f"{year}: Game ordering changed during output construction."
        )

    print(
        f"Output rows:       {len(output):,}"
    )

    print(
        f"Output columns:    {len(output.columns):,}"
    )

    print(
        "Output structure:  VALID"
    )

    return output


def validate_output_dataset(
    output,
    game_id,
    model_columns,
    year,
):
    """Final validation of the output dataset."""

    print_section(
        f"FINAL OUTPUT VALIDATION — {year}"
    )

    if len(output) < MIN_ROWS:

        fail(
            f"{year}: Output contains only "
            f"{len(output)} rows."
        )

    if output[game_id].isna().any():

        fail(
            f"{year}: Output contains missing game IDs."
        )

    if output[game_id].duplicated().any():

        fail(
            f"{year}: Output contains duplicate game IDs."
        )

    target_values = set(
        output[TARGET_COLUMN]
        .dropna()
        .unique()
    )

    if not target_values.issubset(
        {0, 1}
    ):

        fail(
            f"{year}: Invalid target values: "
            f"{sorted(target_values)}"
        )

    expected_features = (
        model_columns["model_4"]
    )

    missing_features = [
        column
        for column in expected_features
        if column not in output.columns
    ]

    if missing_features:

        fail(
            f"{year}: Missing output features:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing_features
            )
        )

    print(
        "Row count:         VALID"
    )

    print(
        "Game uniqueness:   VALID"
    )

    print(
        "Target:            VALID"
    )

    print(
        f"Model 1 features:  {len(model_columns['model_1'])}"
    )

    print(
        f"Model 2 features:  {len(model_columns['model_2'])}"
    )

    print(
        f"Model 3 features:  {len(model_columns['model_3'])}"
    )

    print(
        f"Model 4 features:  {len(model_columns['model_4'])}"
    )

    print(
        "Model features:    VALID"
    )


# =============================================================================
# SEASON PROCESSING
# =============================================================================

def process_year(year):
    """Process one season."""

    print_section(
        f"PROCESSING SEASON {year}"
    )

    input_path = (
        INPUT_DIR
        / f"logistic_features_{year}.csv"
    )

    output_path = (
        OUTPUT_DIR
        / f"logistic_matchup_features_{year}.csv"
    )

    if not input_path.exists():

        fail(
            f"{year}: Input file does not exist:\n"
            f"{input_path}"
        )

    print(
        f"Input:  {input_path}"
    )

    df = pd.read_csv(
        input_path
    )

    print(
        f"Loaded enhanced features: "
        f"{len(df):,} rows × {len(df.columns):,} columns"
    )

    game_id = validate_input_structure(
        df=df,
        year=year,
    )

    # -------------------------------------------------------------------------
    # Validate all feature source columns before constructing anything.
    # -------------------------------------------------------------------------

    validate_required_columns(
        df=df,
        feature_definitions=MODEL_1_FEATURES,
        model_name="MODEL 1",
        year=year,
    )

    validate_required_columns(
        df=df,
        feature_definitions=MODEL_2_FEATURES,
        model_name="MODEL 2",
        year=year,
    )

    validate_required_columns(
        df=df,
        feature_definitions=MODEL_3_FEATURES,
        model_name="MODEL 3",
        year=year,
    )

    validate_required_columns(
        df=df,
        feature_definitions=MODEL_4_FEATURES,
        model_name="MODEL 4",
        year=year,
    )

    # -------------------------------------------------------------------------
    # Create matchup features.
    # -------------------------------------------------------------------------

    model_1 = create_feature_set(
        df=df,
        feature_definitions=MODEL_1_FEATURES,
        model_name="MODEL 1",
        year=year,
    )

    model_2 = create_feature_set(
        df=df,
        feature_definitions=MODEL_2_FEATURES,
        model_name="MODEL 2",
        year=year,
    )

    model_3 = create_feature_set(
        df=df,
        feature_definitions=MODEL_3_FEATURES,
        model_name="MODEL 3",
        year=year,
    )

    model_4 = create_feature_set(
        df=df,
        feature_definitions=MODEL_4_FEATURES,
        model_name="MODEL 4",
        year=year,
    )

    matchup_features = pd.concat(
        [
            model_1,
            model_2,
            model_3,
            model_4,
        ],
        axis=1,
    )

    # -------------------------------------------------------------------------
    # Validate the generated feature sets.
    # -------------------------------------------------------------------------

    validate_feature_values(
        feature_df=model_1,
        feature_definitions=MODEL_1_FEATURES,
        model_name="MODEL 1",
        year=year,
    )

    validate_feature_values(
        feature_df=model_2,
        feature_definitions=MODEL_2_FEATURES,
        model_name="MODEL 2",
        year=year,
    )

    validate_feature_values(
        feature_df=model_3,
        feature_definitions=MODEL_3_FEATURES,
        model_name="MODEL 3",
        year=year,
    )

    validate_feature_values(
        feature_df=model_4,
        feature_definitions=MODEL_4_FEATURES,
        model_name="MODEL 4",
        year=year,
    )

    # -------------------------------------------------------------------------
    # Validate all feature orientations.
    # -------------------------------------------------------------------------

    all_feature_definitions = {
        **MODEL_1_FEATURES,
        **MODEL_2_FEATURES,
        **MODEL_3_FEATURES,
        **MODEL_4_FEATURES,
    }

    validate_orientation(
        source_df=df,
        matchup_df=matchup_features,
        feature_definitions=all_feature_definitions,
        year=year,
    )

    # -------------------------------------------------------------------------
    # Validate all feature missingness.
    # -------------------------------------------------------------------------

    validate_missingness(
        source_df=df,
        matchup_df=matchup_features,
        feature_definitions=all_feature_definitions,
        year=year,
    )

    # -------------------------------------------------------------------------
    # Validate model nesting.
    # -------------------------------------------------------------------------

    model_columns = validate_model_nesting(
        year=year
    )

    # -------------------------------------------------------------------------
    # Validate leakage across the complete Model 4 feature set.
    # -------------------------------------------------------------------------

    validate_no_target_leakage(
        feature_columns=model_columns["model_4"],
        year=year,
    )

    # -------------------------------------------------------------------------
    # Build final output.
    # -------------------------------------------------------------------------

    output = build_output_dataset(
        source_df=df,
        game_id=game_id,
        matchup_features=matchup_features,
        model_columns=model_columns,
        year=year,
    )

    # -------------------------------------------------------------------------
    # Validate final output.
    # -------------------------------------------------------------------------

    validate_output_dataset(
        output=output,
        game_id=game_id,
        model_columns=model_columns,
        year=year,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.to_csv(
        output_path,
        index=False,
    )

    print()
    print(
        f"Saved: {output_path}"
    )

    return {
        "year": year,
        "rows": len(output),
        "columns": len(output.columns),
        "model_1_features": len(
            model_columns["model_1"]
        ),
        "model_2_features": len(
            model_columns["model_2"]
        ),
        "model_3_features": len(
            model_columns["model_3"]
        ),
        "model_4_features": len(
            model_columns["model_4"]
        ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the complete matchup feature-generation pipeline."""

    print_section(
        "LOGISTIC REGRESSION COMPACT MATCHUP FEATURE ENGINEERING"
    )

    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Input:        {INPUT_DIR}"
    )

    print(
        f"Output:       {OUTPUT_DIR}"
    )

    print(
        f"Years:        {YEARS[0]}–{YEARS[-1]}"
    )

    print()
    print(
        "MODEL DESIGN"
    )

    print(
        "-------------"
    )

    print(
        f"Model 1: {len(MODEL_1_FEATURES)} compact baseline features"
    )

    print(
        f"Model 2: + {len(MODEL_2_FEATURES)} recent-form features"
    )

    print(
        f"Model 3: + {len(MODEL_3_FEATURES)} SOS features"
    )

    print(
        f"Model 4: + {len(MODEL_4_FEATURES)} additional football features"
    )

    print()
    print(
        "TOTAL FEATURE COUNTS"
    )

    print(
        "--------------------"
    )

    model_columns = create_model_columns()

    print(
        f"Model 1 total: {len(model_columns['model_1'])}"
    )

    print(
        f"Model 2 total: {len(model_columns['model_2'])}"
    )

    print(
        f"Model 3 total: {len(model_columns['model_3'])}"
    )

    print(
        f"Model 4 total: {len(model_columns['model_4'])}"
    )

    print()
    print(
        "MODEL 4 ADDITIONS"
    )

    print(
        "-----------------"
    )

    for feature in MODEL_4_FEATURES:
        print(
            f"  {feature}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = []

    for year in YEARS:

        try:

            summary = process_year(
                year
            )

            summaries.append(
                summary
            )

        except Exception:

            print_section(
                f"FAILED: {year}"
            )

            raise

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------

    print_section(
        "FINAL SUMMARY"
    )

    summary_df = pd.DataFrame(
        summaries
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Cross-season validation.
    # -------------------------------------------------------------------------

    print_section(
        "CROSS-SEASON VALIDATION"
    )

    expected_model_1 = len(
        MODEL_1_FEATURES
    )

    expected_model_2 = (
        expected_model_1
        + len(MODEL_2_FEATURES)
    )

    expected_model_3 = (
        expected_model_2
        + len(MODEL_3_FEATURES)
    )

    expected_model_4 = (
        expected_model_3
        + len(MODEL_4_FEATURES)
    )

    if not (
        summary_df["model_1_features"]
        == expected_model_1
    ).all():

        fail(
            "Cross-season Model 1 feature count mismatch."
        )

    if not (
        summary_df["model_2_features"]
        == expected_model_2
    ).all():

        fail(
            "Cross-season Model 2 feature count mismatch."
        )

    if not (
        summary_df["model_3_features"]
        == expected_model_3
    ).all():

        fail(
            "Cross-season Model 3 feature count mismatch."
        )

    if not (
        summary_df["model_4_features"]
        == expected_model_4
    ).all():

        fail(
            "Cross-season Model 4 feature count mismatch."
        )

    print(
        f"Model 1 feature count: {expected_model_1}"
    )

    print(
        f"Model 2 feature count: {expected_model_2}"
    )

    print(
        f"Model 3 feature count: {expected_model_3}"
    )

    print(
        f"Model 4 feature count: {expected_model_4}"
    )

    print(
        "Cross-season feature consistency: VALID"
    )

    print_section(
        "COMPACT MATCHUP FEATURE ENGINEERING COMPLETED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()