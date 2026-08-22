"""
Create Final Feature Classification
====================================

Creates feature classification metadata for the FINAL game-level feature
datasets.

Final feature files:
    data/processed/final_features/final_features_{year}.csv

Expected final schema:
    448 columns for every season, 2015-2025.

This is the mature feature-classification layer.

IMPORTANT:
    This script intentionally does NOT depend on the old
    data/metadata/feature_classification.csv file.

    The previous classification belonged to the older 279-column
    team-level dataset and is no longer part of the production pipeline.

Classification categories:
    identifier
    metadata
    pregame
    rolling_pregame
    static_pregame
    current_game
    target_candidate

Predictive-safe definition for true pregame prediction:

    TRUE:
        pregame
        rolling_pregame
        static_pregame

    FALSE:
        identifier
        metadata
        current_game
        target_candidate

The script:

    1. Loads the final feature schema from all seasons.
    2. Confirms every season has the same 448-column schema.
    3. Classifies every final feature using explicit final-schema rules.
    4. Fails if any feature is unresolved.
    5. Validates category/safety consistency.
    6. Validates exact feature coverage.
    7. Saves:
           data/metadata/final_feature_classification.csv

This file should be run after:
    create_final_features.py

and before:
    validate_final_feature_classification.py
    validate_predictive_safety.py
"""

from pathlib import Path
import sys

import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

FINAL_CLASSIFICATION_PATH = (
    METADATA_DIR / "final_feature_classification.csv"
)

FINAL_FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_features"
)

YEARS = list(range(2015, 2026))

EXPECTED_COLUMN_COUNT = 448


# ============================================================================
# VALID CATEGORIES
# ============================================================================

VALID_CATEGORIES = {
    "identifier",
    "metadata",
    "pregame",
    "rolling_pregame",
    "static_pregame",
    "current_game",
    "target_candidate",
}


PREDICTIVE_SAFE_CATEGORIES = {
    "pregame",
    "rolling_pregame",
    "static_pregame",
}


# ============================================================================
# DISPLAY
# ============================================================================

LINE = "=" * 70


def print_header(title):
    print()
    print(LINE)
    print(title)
    print(LINE)


# ============================================================================
# IDENTIFIERS
# ============================================================================

IDENTIFIER_COLUMNS = {
    "id",
    "gameId",
    "game_id",
    "homeId",
    "awayId",
    "teamId",
    "team_id",
    "venueId",
    "season",

    # Final game-level team identifiers
    "homeTeam",
    "awayTeam",
}


# ============================================================================
# METADATA
# ============================================================================

METADATA_COLUMNS = {
    "attendance",
    "awayClassification",
    "awayConference",
    "awayLineScores",
    "conference",
    "conferenceGame",
    "excitementIndex",
    "highlights",
    "homeClassification",
    "homeConference",
    "homeLineScores",
    "neutralSite",
    "notes",
    "playoff",
    "startTimeTBD",
    "venue",
    "venueId",

    # Final game metadata
    "week",
    "completed",
}


# ============================================================================
# STATIC PREGAME FEATURES
# ============================================================================

STATIC_PREGAME_COLUMNS = {
    # ------------------------------------------------------------------------
    # Pregame Elo
    # ------------------------------------------------------------------------

    "homePregameElo",
    "awayPregameElo",

    # ------------------------------------------------------------------------
    # Pregame advanced offense metrics
    # ------------------------------------------------------------------------

    "home_pregame_offense_explosiveness",
    "away_pregame_offense_explosiveness",

    "home_pregame_offense_ppa",
    "away_pregame_offense_ppa",

    "home_pregame_offense_successRate",
    "away_pregame_offense_successRate",

    # ------------------------------------------------------------------------
    # Pregame advanced defense metrics
    # ------------------------------------------------------------------------

    "home_pregame_defense_explosiveness",
    "away_pregame_defense_explosiveness",

    "home_pregame_defense_ppa",
    "away_pregame_defense_ppa",

    "home_pregame_defense_successRate",
    "away_pregame_defense_successRate",

    # ------------------------------------------------------------------------
    # Pregame game metadata
    #
    # These are known before kickoff and therefore are safe for true
    # pregame prediction.
    # ------------------------------------------------------------------------

    "seasonType",
    "startDate",
}


# ============================================================================
# TARGET CANDIDATES
# ============================================================================

TARGET_COLUMNS = {
    "win",
    "win_home",
    "win_away",

    "points",
    "points_home",
    "points_away",

    "homePoints",
    "awayPoints",

    "pointsFor",
    "pointsFor_home",
    "pointsFor_away",

    "pointsAgainst",
    "pointsAgainst_home",
    "pointsAgainst_away",

    "pointDifferential",
    "pointDifferential_home",
    "pointDifferential_away",

    # Final game-level scoring columns
    "homePoints_home",
    "awayPoints_away",
}


# ============================================================================
# CURRENT-GAME STATISTICS
# ============================================================================

CURRENT_GAME_FEATURES = {
    "completionAttempts",
    "completionPct",
    "completions",

    "defensiveTDs",

    "firstDowns",

    "fourthDownAttempts",
    "fourthDownConversions",
    "fourthDownEff",
    "fourthDownPct",

    "fumblesLost",
    "fumblesRecovered",

    "interceptionTDs",
    "interceptionYards",
    "interceptions",

    "kickReturnTDs",
    "kickReturnYards",
    "kickReturns",

    "kickingPoints",

    "netPassingYards",

    "passAttempts",
    "passesDeflected",
    "passesIntercepted",
    "passingTDs",

    "penalties",
    "penaltyYards",

    "points",
    "pointsAgainst",
    "pointsFor",

    "possessionSeconds",
    "possessionTime",

    "puntReturnTDs",
    "puntReturnYards",
    "puntReturns",

    "qbHurries",

    "rushingAttempts",
    "rushingTDs",
    "rushingYards",

    "sacks",

    "tackles",
    "tacklesForLoss",

    "thirdDownAttempts",
    "thirdDownConversions",
    "thirdDownEff",
    "thirdDownPct",

    "totalFumbles",
    "totalPenaltiesYards",
    "totalYards",

    "turnovers",

    "yardsPerPass",
    "yardsPerPassAttempt",
    "yardsPerRushAttempt",
}


# ============================================================================
# ROLLING FEATURE MARKERS
# ============================================================================

ROLLING_MARKERS = (
    "AvgLast3",
    "AvgLast5",
    "Last3",
    "Last5",
)


# ============================================================================
# PRE-GAME FEATURE MARKERS
# ============================================================================

PREGAME_MARKERS = (
    "Before",
    "before",
)


# ============================================================================
# POST-GAME MARKERS
# ============================================================================

POSTGAME_MARKERS = (
    "Postgame",
    "postgame",
)


# ============================================================================
# METADATA PATTERNS
# ============================================================================

METADATA_PATTERNS = (
    "Classification",
    "Conference",
    "LineScores",
)


# ============================================================================
# LOAD FINAL FEATURE SCHEMA
# ============================================================================

def load_final_feature_columns():
    """
    Load the final feature schema from every season.

    All seasons must contain exactly the same 448 columns.
    """

    print_header("LOADING FINAL FEATURE DATASETS")

    if not FINAL_FEATURE_DIR.exists():

        raise FileNotFoundError(
            "Final feature directory does not exist:\n"
            f"  {FINAL_FEATURE_DIR}"
        )

    season_columns = {}

    for year in YEARS:

        path = (
            FINAL_FEATURE_DIR
            / f"final_features_{year}.csv"
        )

        if not path.exists():

            raise FileNotFoundError(
                f"Could not find final feature file for {year}:\n"
                f"  {path}"
            )

        df = pd.read_csv(
            path,
            nrows=0,
        )

        columns = list(df.columns)

        season_columns[year] = columns

        print(
            f"{year}: "
            f"{len(columns):,} columns"
        )

    # ------------------------------------------------------------------------
    # Reference schema
    # ------------------------------------------------------------------------

    reference_year = YEARS[0]

    reference_columns = season_columns[
        reference_year
    ]

    print()
    print(
        f"Reference season: {reference_year}"
    )

    print(
        f"Reference columns: "
        f"{len(reference_columns):,}"
    )

    # ------------------------------------------------------------------------
    # Expected column count
    # ------------------------------------------------------------------------

    if len(reference_columns) != EXPECTED_COLUMN_COUNT:

        raise ValueError(
            f"Expected {EXPECTED_COLUMN_COUNT} final feature columns "
            f"for {reference_year}, "
            f"but found {len(reference_columns)}."
        )

    # ------------------------------------------------------------------------
    # Duplicate columns
    # ------------------------------------------------------------------------

    duplicate_columns = pd.Series(
        reference_columns
    )

    duplicates = duplicate_columns[
        duplicate_columns.duplicated(
            keep=False
        )
    ]

    if len(duplicates) > 0:

        raise ValueError(
            "Duplicate columns found in final feature schema:\n"
            + "\n".join(
                f"  {column}"
                for column in sorted(
                    duplicates.unique()
                )
            )
        )

    print(
        "PASS: Reference schema contains "
        "no duplicate columns."
    )

    # ------------------------------------------------------------------------
    # Compare all seasons
    # ------------------------------------------------------------------------

    schema_errors = []

    for year in YEARS:

        columns = season_columns[year]

        missing = sorted(
            set(reference_columns)
            - set(columns)
        )

        extra = sorted(
            set(columns)
            - set(reference_columns)
        )

        if missing or extra:

            schema_errors.append(year)

            print()
            print(
                f"SCHEMA MISMATCH: {year}"
            )

            print(
                f"  Columns: {len(columns):,}"
            )

            if missing:

                print("  Missing:")

                for column in missing:
                    print(
                        f"    {column}"
                    )

            if extra:

                print("  Extra:")

                for column in extra:
                    print(
                        f"    {column}"
                    )

    if schema_errors:

        raise ValueError(
            "Final feature datasets do not have "
            "identical schemas."
        )

    print()
    print(
        "PASS: All final feature datasets contain "
        f"the same {EXPECTED_COLUMN_COUNT}-column schema."
    )

    return reference_columns


# ============================================================================
# REMOVE HOME/AWAY SUFFIX
# ============================================================================

def remove_game_side_suffix(column):
    """
    Remove the final _home or _away suffix.

    Examples:
        pointsFor_home
            -> pointsFor

        totalYards_away
            -> totalYards

        homeWinsBefore_home
            -> homeWinsBefore
    """

    if column.endswith("_home"):

        return column[:-5]

    if column.endswith("_away"):

        return column[:-5]

    return column


# ============================================================================
# CLASSIFICATION HELPERS
# ============================================================================

def is_identifier(column):
    """
    Determine whether a column is an identifier.
    """

    if column in IDENTIFIER_COLUMNS:

        return True

    if column.endswith("Id"):

        return True

    if column.endswith("_id"):

        return True

    return False


def is_metadata(column):
    """
    Determine whether a column is metadata.
    """

    if column in METADATA_COLUMNS:

        return True

    for pattern in METADATA_PATTERNS:

        if pattern in column:

            return True

    return False


def is_target_candidate(column):
    """
    Determine whether a column represents a potential model target.
    """

    return column in TARGET_COLUMNS


def is_static_pregame(column):
    """
    Determine whether a column is an explicitly defined static
    pregame feature.
    """

    if column in STATIC_PREGAME_COLUMNS:

        return True

    # Advanced pregame metrics
    if "_pregame_" in column:

        return True

    return False


def is_rolling_pregame(column):
    """
    Determine whether a column is a historical rolling feature.
    """

    for marker in ROLLING_MARKERS:

        if marker in column:

            return True

    return False


def is_pregame(column):
    """
    Determine whether a column is a historical pregame feature.

    Features containing "Before" represent information accumulated
    prior to the current game.
    """

    for marker in PREGAME_MARKERS:

        if marker in column:

            return True

    return False


def is_current_game(column):
    """
    Determine whether a column represents a current-game statistic.
    """

    base = remove_game_side_suffix(
        column
    )

    return base in CURRENT_GAME_FEATURES


# ============================================================================
# CLASSIFY SINGLE COLUMN
# ============================================================================

def classify_final_column(column):
    """
    Classify a single final feature.

    Rules are deliberately explicit and ordered by safety.

    The order matters because some current-game statistics such as
    points, pointsFor, and pointsAgainst are also present in TARGET_COLUMNS.
    Targets must therefore be identified before current-game statistics.
    """

    # ------------------------------------------------------------------------
    # 1. Identifier
    # ------------------------------------------------------------------------

    if is_identifier(column):

        return (
            "identifier",
            False,
            "identifier column",
        )

    # ------------------------------------------------------------------------
    # 2. Target candidate
    # ------------------------------------------------------------------------

    if is_target_candidate(column):

        return (
            "target_candidate",
            False,
            "game outcome or target candidate",
        )

    # ------------------------------------------------------------------------
    # 3. Metadata
    # ------------------------------------------------------------------------

    if is_metadata(column):

        return (
            "metadata",
            False,
            "game metadata",
        )

    # ------------------------------------------------------------------------
    # 4. Static pregame
    # ------------------------------------------------------------------------

    if is_static_pregame(column):

        return (
            "static_pregame",
            True,
            "known before kickoff",
        )

    # ------------------------------------------------------------------------
    # 5. Rolling pregame
    # ------------------------------------------------------------------------

    if is_rolling_pregame(column):

        return (
            "rolling_pregame",
            True,
            "historical rolling statistic",
        )

    # ------------------------------------------------------------------------
    # 6. Historical pregame
    # ------------------------------------------------------------------------

    if is_pregame(column):

        return (
            "pregame",
            True,
            "historical statistic accumulated before current game",
        )

    # ------------------------------------------------------------------------
    # 7. Explicit postgame detection
    # ------------------------------------------------------------------------

    for marker in POSTGAME_MARKERS:

        if marker in column:

            return (
                "current_game",
                False,
                "postgame/current-game information",
            )

    # ------------------------------------------------------------------------
    # 8. Current-game statistics
    # ------------------------------------------------------------------------

    if is_current_game(column):

        return (
            "current_game",
            False,
            "current-game statistic",
        )

    # ------------------------------------------------------------------------
    # 9. Unresolved
    # ------------------------------------------------------------------------

    return (
        None,
        None,
        "UNRESOLVED",
    )


# ============================================================================
# CLASSIFY ALL FEATURES
# ============================================================================

def create_final_classification(
    final_columns,
):
    """
    Create classification metadata for all final features.
    """

    print_header(
        "CLASSIFYING FINAL FEATURE SCHEMA"
    )

    rows = []

    unresolved = []

    for column in final_columns:

        (
            category,
            predictive_safe,
            reason,
        ) = classify_final_column(
            column
        )

        if category is None:

            unresolved.append(
                column
            )

        rows.append(
            {
                "column": column,
                "category": category,
                "predictive_safe": predictive_safe,
                "classification_source": (
                    "final_schema_rule"
                    if category is not None
                    else "unresolved"
                ),
                "classification_reason": reason,
            }
        )

    classification = pd.DataFrame(
        rows
    )

    print()
    print(
        f"Final feature columns: "
        f"{len(final_columns):,}"
    )

    print(
        f"Classified columns: "
        f"{classification['category'].notna().sum():,}"
    )

    print(
        f"Unclassified columns: "
        f"{len(unresolved):,}"
    )

    if unresolved:

        print()
        print(
            "UNRESOLVED COLUMNS:"
        )

        for column in unresolved:

            print(
                f"  {column}"
            )

    return (
        classification,
        unresolved,
    )


# ============================================================================
# VALIDATE GENERATED CLASSIFICATION
# ============================================================================

def validate_generated_classification(
    classification,
    final_columns,
):
    """
    Validate classification before saving.
    """

    print_header(
        "VALIDATING GENERATED CLASSIFICATION"
    )

    errors = []

    # ------------------------------------------------------------------------
    # Required columns
    # ------------------------------------------------------------------------

    required_columns = {
        "column",
        "category",
        "predictive_safe",
        "classification_source",
        "classification_reason",
    }

    missing_required = (
        required_columns
        - set(classification.columns)
    )

    if missing_required:

        errors.append(
            "Missing required classification columns."
        )

        print(
            "FAIL: Missing required columns:"
        )

        for column in sorted(
            missing_required
        ):

            print(
                f"  {column}"
            )

    else:

        print(
            "PASS: Required classification columns "
            "are present."
        )

    # ------------------------------------------------------------------------
    # Row count
    # ------------------------------------------------------------------------

    print()

    print(
        f"Classification rows: "
        f"{len(classification):,}"
    )

    if len(classification) != len(final_columns):

        errors.append(
            "Classification row count does not "
            "match final feature count."
        )

        print(
            "FAIL: Classification row count "
            "does not match final schema."
        )

    else:

        print(
            "PASS: Classification row count "
            "matches final schema."
        )

    # ------------------------------------------------------------------------
    # Duplicate feature names
    # ------------------------------------------------------------------------

    duplicate_columns = classification[
        classification["column"].duplicated(
            keep=False
        )
    ]

    duplicate_count = (
        duplicate_columns["column"].nunique()
    )

    print()

    print(
        f"Duplicate classified feature names: "
        f"{duplicate_count:,}"
    )

    if duplicate_count:

        errors.append(
            "Duplicate classified feature names."
        )

        for column in sorted(
            duplicate_columns["column"].unique()
        ):

            print(
                f"  {column}"
            )

    else:

        print(
            "PASS: No duplicate classified "
            "feature names."
        )

    # ------------------------------------------------------------------------
    # Missing categories
    # ------------------------------------------------------------------------

    missing_categories = classification[
        classification["category"].isna()
    ]

    print()

    print(
        f"Missing feature categories: "
        f"{len(missing_categories):,}"
    )

    if len(missing_categories):

        errors.append(
            "One or more features have no category."
        )

        for column in missing_categories[
            "column"
        ]:

            print(
                f"  {column}"
            )

    else:

        print(
            "PASS: Every feature has a category."
        )

    # ------------------------------------------------------------------------
    # Missing predictive-safe flags
    # ------------------------------------------------------------------------

    missing_safe = classification[
        classification["predictive_safe"].isna()
    ]

    print()

    print(
        f"Missing predictive-safe flags: "
        f"{len(missing_safe):,}"
    )

    if len(missing_safe):

        errors.append(
            "One or more features have no "
            "predictive-safe flag."
        )

        for column in missing_safe[
            "column"
        ]:

            print(
                f"  {column}"
            )

    else:

        print(
            "PASS: Every feature has a "
            "predictive-safe flag."
        )

    # ------------------------------------------------------------------------
    # Category validity
    # ------------------------------------------------------------------------

    invalid_categories = (
        set(
            classification["category"]
            .dropna()
        )
        - VALID_CATEGORIES
    )

    print()

    print(
        f"Invalid categories: "
        f"{len(invalid_categories):,}"
    )

    if invalid_categories:

        errors.append(
            "Invalid feature categories found."
        )

        for category in sorted(
            invalid_categories
        ):

            print(
                f"  {category}"
            )

    else:

        print(
            "PASS: All feature categories are valid."
        )

    # ------------------------------------------------------------------------
    # Exact final feature coverage
    # ------------------------------------------------------------------------

    final_set = set(
        final_columns
    )

    classified_set = set(
        classification["column"]
    )

    missing_from_classification = (
        final_set
        - classified_set
    )

    classification_not_in_final = (
        classified_set
        - final_set
    )

    print_header(
        "FINAL CLASSIFICATION COVERAGE"
    )

    print(
        f"Final feature columns:       "
        f"{len(final_set):,}"
    )

    print(
        f"Classified feature columns:  "
        f"{len(classified_set):,}"
    )

    print(
        f"Unclassified final columns:  "
        f"{len(missing_from_classification):,}"
    )

    print(
        f"Classified but not final:    "
        f"{len(classification_not_in_final):,}"
    )

    if missing_from_classification:

        errors.append(
            "Final feature columns are missing "
            "from classification."
        )

        print()
        print(
            "MISSING FROM CLASSIFICATION:"
        )

        for column in sorted(
            missing_from_classification
        ):

            print(
                f"  {column}"
            )

    else:

        print(
            "PASS: Every final feature is classified."
        )

    if classification_not_in_final:

        errors.append(
            "Classification contains columns not "
            "present in final feature data."
        )

        print()
        print(
            "CLASSIFIED BUT NOT IN FINAL DATA:"
        )

        for column in sorted(
            classification_not_in_final
        ):

            print(
                f"  {column}"
            )

    else:

        print(
            "PASS: Every classified feature exists "
            "in final feature data."
        )

    # ------------------------------------------------------------------------
    # Category counts
    # ------------------------------------------------------------------------

    print_header(
        "FINAL CATEGORY COUNTS"
    )

    category_counts = (
        classification["category"]
        .value_counts()
        .sort_index()
    )

    for category in sorted(
        VALID_CATEGORIES
    ):

        count = category_counts.get(
            category,
            0,
        )

        print(
            f"  {category:<20}"
            f"{count:,}"
        )

    # ------------------------------------------------------------------------
    # Predictive-safe counts
    # ------------------------------------------------------------------------

    print()

    print(
        "Predictive-safe counts:"
    )

    safe_counts = (
        classification["predictive_safe"]
        .value_counts()
        .sort_index()
    )

    for value, count in safe_counts.items():

        print(
            f"  {str(value):<6}"
            f"{count:,}"
        )

    # ------------------------------------------------------------------------
    # Category / safety consistency
    # ------------------------------------------------------------------------

    inconsistent = []

    for _, row in classification.iterrows():

        expected_safe = (
            row["category"]
            in PREDICTIVE_SAFE_CATEGORIES
        )

        actual_safe = bool(
            row["predictive_safe"]
        )

        if expected_safe != actual_safe:

            inconsistent.append(
                {
                    "column": row["column"],
                    "category": row["category"],
                    "actual": actual_safe,
                    "expected": expected_safe,
                }
            )

    print()

    print(
        "Predictive-safe/category inconsistencies: "
        f"{len(inconsistent):,}"
    )

    if inconsistent:

        errors.append(
            "Predictive-safe flags are inconsistent "
            "with feature categories."
        )

        for row in inconsistent:

            print(
                f"  {row['column']}: "
                f"category={row['category']}, "
                f"safe={row['actual']}, "
                f"expected={row['expected']}"
            )

    else:

        print(
            "PASS: Predictive-safe flags are "
            "consistent with categories."
        )

    # ------------------------------------------------------------------------
    # Safety rules
    # ------------------------------------------------------------------------

    safety_violations = []

    for _, row in classification.iterrows():

        category = row["category"]

        safe = bool(
            row["predictive_safe"]
        )

        # These categories must never be safe.
        unsafe_categories = {
            "identifier",
            "metadata",
            "current_game",
            "target_candidate",
        }

        if (
            category in unsafe_categories
            and safe
        ):

            safety_violations.append(
                (
                    row["column"],
                    category,
                    safe,
                )
            )

        # These categories must be safe.
        if (
            category
            in PREDICTIVE_SAFE_CATEGORIES
            and not safe
        ):

            safety_violations.append(
                (
                    row["column"],
                    category,
                    safe,
                )
            )

    print()

    print(
        f"Category safety violations: "
        f"{len(safety_violations):,}"
    )

    if safety_violations:

        errors.append(
            "Category safety violations detected."
        )

        for (
            column,
            category,
            safe,
        ) in safety_violations:

            print(
                f"  {column}: "
                f"category={category}, "
                f"safe={safe}"
            )

    else:

        print(
            "PASS: Category safety rules are satisfied."
        )

    return errors


# ============================================================================
# SAVE CLASSIFICATION
# ============================================================================

def save_classification(
    classification,
):
    """
    Save final feature classification.
    """

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    classification = (
        classification
        .sort_values("column")
        .reset_index(drop=True)
    )

    classification.to_csv(
        FINAL_CLASSIFICATION_PATH,
        index=False,
    )

    print_header(
        "SAVED FINAL FEATURE CLASSIFICATION"
    )

    print(
        "Path:"
    )

    print(
        f"  {FINAL_CLASSIFICATION_PATH}"
    )

    print()

    print(
        f"Rows: "
        f"{len(classification):,}"
    )

    print(
        f"Columns: "
        f"{len(classification.columns):,}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print_header(
        "CREATING FINAL FEATURE CLASSIFICATION"
    )

    print(
        "This script creates classification metadata "
        "directly from the mature 448-column final "
        "feature schema."
    )

    print()

    print(
        "No legacy feature classification is used."
    )

    print()

    print(
        "Final feature directory:"
    )

    print(
        f"  {FINAL_FEATURE_DIR}"
    )

    # ------------------------------------------------------------------------
    # Load final schema
    # ------------------------------------------------------------------------

    final_columns = (
        load_final_feature_columns()
    )

    # ------------------------------------------------------------------------
    # Create classification
    # ------------------------------------------------------------------------

    (
        classification,
        unresolved,
    ) = create_final_classification(
        final_columns
    )

    # ------------------------------------------------------------------------
    # Never save unresolved classification
    # ------------------------------------------------------------------------

    if unresolved:

        print_header(
            "CLASSIFICATION FAILED"
        )

        print(
            f"FAIL: {len(unresolved):,} "
            "feature(s) remain unclassified."
        )

        print()

        print(
            "The classification file was NOT saved."
        )

        print()

        print(
            "Add an explicit classification rule "
            "for every unresolved feature before "
            "continuing."
        )

        sys.exit(1)

    # ------------------------------------------------------------------------
    # Validate generated classification
    # ------------------------------------------------------------------------

    errors = validate_generated_classification(
        classification,
        final_columns,
    )

    if errors:

        print_header(
            "FINAL CLASSIFICATION FAILED"
        )

        for error in errors:

            print(
                f"FAIL: {error}"
            )

        print()

        print(
            "The classification file was NOT saved."
        )

        sys.exit(1)

    # ------------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------------

    save_classification(
        classification
    )

    # ------------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------------

    print_header(
        "FINAL RESULT"
    )

    print(
        "PASS: Final feature classification "
        "created successfully."
    )

    print()

    print(
        "The mature classification is now "
        "independent of the legacy 279-column "
        "feature classification."
    )

    print()

    print(
        "Next validation steps:"
    )

    print(
        "  python src\\data\\validate\\validate_final_feature_classification.py"
    )

    print(
        "  python src\\data\\validate\\validate_predictive_safety.py"
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()