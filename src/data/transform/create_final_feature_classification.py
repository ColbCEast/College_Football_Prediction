"""
Create Final Feature Classification
====================================

Creates feature classification metadata for the FINAL game-level feature
datasets.

Final feature files:
    data/processed/final_features/final_features_{year}.csv

Expected final schema:
    448 columns for every season, 2015-2025.

The previous feature classification was created against an older 279-column
team-level dataset. This script creates a new classification against the
actual final 448-column game-level schema.

Classification categories:
    identifier
    metadata
    pregame
    rolling_pregame
    static_pregame
    current_game
    target_candidate

Predictive-safe definition for pregame modeling:
    TRUE:
        pregame
        rolling_pregame
        static_pregame

    FALSE:
        identifier
        metadata
        current_game
        target_candidate
"""

from pathlib import Path
import sys

import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

OLD_CLASSIFICATION_PATH = (
    METADATA_DIR / "feature_classification.csv"
)

FINAL_CLASSIFICATION_PATH = (
    METADATA_DIR / "final_feature_classification.csv"
)

FINAL_FEATURE_DIR = (
    PROJECT_ROOT / "data" / "processed" / "final_features"
)


YEARS = list(range(2015, 2026))


# ============================================================================
# EXPECTED CATEGORIES
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
# LOAD EXISTING CLASSIFICATION
# ============================================================================

def load_old_classification():
    """
    Load the previous 279-column classification.

    This is used as a reference when translating old team-level features into
    the final home/away game-level features.
    """

    if not OLD_CLASSIFICATION_PATH.exists():
        raise FileNotFoundError(
            f"Could not find existing classification:\n"
            f"  {OLD_CLASSIFICATION_PATH}"
        )

    classification = pd.read_csv(
        OLD_CLASSIFICATION_PATH
    )

    print("Loaded existing classification:")
    print(f"  Rows:    {len(classification):,}")
    print(f"  Columns: {len(classification.columns):,}")

    required = {
        "column",
        "category",
        "predictive_safe",
    }

    missing = required - set(classification.columns)

    if missing:
        raise ValueError(
            "Existing classification is missing required columns:\n"
            + "\n".join(
                f"  {column}"
                for column in sorted(missing)
            )
        )

    return classification


# ============================================================================
# LOAD FINAL FEATURE COLUMNS
# ============================================================================

def load_final_feature_columns():
    """
    Load the final feature schema from every season.

    All seasons must contain exactly the same 448 columns.
    """

    print_header("LOADING FINAL FEATURE DATASETS")

    if not FINAL_FEATURE_DIR.exists():
        raise FileNotFoundError(
            f"Final feature directory does not exist:\n"
            f"  {FINAL_FEATURE_DIR}"
        )

    season_columns = {}

    for year in YEARS:

        path = FINAL_FEATURE_DIR / (
            f"final_features_{year}.csv"
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

        season_columns[year] = list(df.columns)

        print(
            f"{year}: "
            f"{len(df.columns):,} columns"
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
    # Require expected 448-column schema
    # ------------------------------------------------------------------------

    if len(reference_columns) != 448:
        raise ValueError(
            f"Expected 448 final feature columns for {reference_year}, "
            f"but found {len(reference_columns)}."
        )

    # ------------------------------------------------------------------------
    # Compare every season
    # ------------------------------------------------------------------------

    schema_errors = []

    for year in YEARS:

        columns = season_columns[year]

        missing = sorted(
            set(reference_columns) - set(columns)
        )

        extra = sorted(
            set(columns) - set(reference_columns)
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
                    print(f"    {column}")

            if extra:

                print("  Extra:")

                for column in extra:
                    print(f"    {column}")

    if schema_errors:
        raise ValueError(
            "Final feature datasets do not have identical schemas."
        )

    print()
    print(
        "PASS: All final feature datasets contain "
        "the same 448-column schema."
    )

    return reference_columns


# ============================================================================
# OLD CLASSIFICATION LOOKUP
# ============================================================================

def create_old_lookup(old_classification):
    """
    Create a dictionary from the old classification.
    """

    lookup = {}

    for _, row in old_classification.iterrows():

        lookup[row["column"]] = {
            "category": row["category"],
            "predictive_safe": bool(
                row["predictive_safe"]
            ),
        }

    return lookup


# ============================================================================
# NAME NORMALIZATION
# ============================================================================

def remove_game_side_suffix(column):
    """
    Remove the final _home or _away suffix.

    Examples:
        pointsFor_home
            -> pointsFor

        pointsForBefore_home
            -> pointsForBefore

        homePointsForBefore_home
            -> homePointsForBefore
    """

    if column.endswith("_home"):
        return column[:-5]

    if column.endswith("_away"):
        return column[:-5]

    return column


def generate_possible_old_names(column):
    """
    Generate possible old classification names for a final feature.

    This handles the transformations introduced when team-level data was
    converted into game-level home/away data.
    """

    candidates = []

    # Exact
    candidates.append(column)

    # Remove final _home/_away
    base = remove_game_side_suffix(column)
    candidates.append(base)

    # Remove home/away prefix from the base
    if base.startswith("home"):
        candidates.append(base[4:])

    if base.startswith("away"):
        candidates.append(base[4:])

    # Lowercase first letter after home/away prefix.
    if base.startswith("home") and len(base) > 4:
        remainder = base[4:]
        candidates.append(
            remainder[0].lower() + remainder[1:]
        )

    if base.startswith("away") and len(base) > 4:
        remainder = base[4:]
        candidates.append(
            remainder[0].lower() + remainder[1:]
        )

    # Remove duplicate candidates while preserving order.
    result = []

    for candidate in candidates:

        if candidate not in result:
            result.append(candidate)

    return result


def lookup_old_classification(
    column,
    old_lookup,
):
    """
    Try to find the old classification corresponding to a final column.
    """

    candidates = generate_possible_old_names(
        column
    )

    for candidate in candidates:

        if candidate in old_lookup:
            return old_lookup[candidate]

    return None


# ============================================================================
# EXPLICIT IDENTIFIERS
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
}


# ============================================================================
# EXPLICIT METADATA
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
}


# ============================================================================
# TARGET COLUMNS
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
}


# ============================================================================
# CURRENT-GAME FEATURES
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
# ROLLING FEATURES
# ============================================================================

ROLLING_MARKERS = (
    "AvgLast3",
    "AvgLast5",
    "Last3",
    "Last5",
)


# ============================================================================
# EXPLICIT STATIC PREGAME FEATURES
# ============================================================================

STATIC_PREGAME_COLUMNS = {
    "homePregameElo",
    "awayPregameElo",

    "home_pregame_offense_explosiveness",
    "away_pregame_offense_explosiveness",

    "home_pregame_offense_ppa",
    "away_pregame_offense_ppa",

    "home_pregame_offense_successRate",
    "away_pregame_offense_successRate",

    "home_pregame_defense_explosiveness",
    "away_pregame_defense_explosiveness",

    "home_pregame_defense_ppa",
    "away_pregame_defense_ppa",

    "home_pregame_defense_successRate",
    "away_pregame_defense_successRate",
}


# ============================================================================
# CLASSIFY BY FINAL COLUMN NAME
# ============================================================================

def classify_final_column(
    column,
    old_lookup,
):
    """
    Classify one final 448-column feature.

    Explicit final-schema rules take precedence over the old classification.
    """

    # ------------------------------------------------------------------------
    # 1. Identifier
    # ------------------------------------------------------------------------

    if column in IDENTIFIER_COLUMNS:
        return (
            "identifier",
            False,
            "explicit identifier",
        )

    # ------------------------------------------------------------------------
    # 2. Metadata
    # ------------------------------------------------------------------------

    if column in METADATA_COLUMNS:
        return (
            "metadata",
            False,
            "explicit metadata",
        )

    # ------------------------------------------------------------------------
    # 3. Target
    # ------------------------------------------------------------------------

    if column in TARGET_COLUMNS:
        return (
            "target_candidate",
            False,
            "game outcome/target",
        )

    # ------------------------------------------------------------------------
    # 4. Postgame features
    # ------------------------------------------------------------------------

    if (
        "Postgame" in column
        or "postgame" in column
    ):
        return (
            "current_game",
            False,
            "postgame information",
        )

    # ------------------------------------------------------------------------
    # 5. Explicit static pregame features
    # ------------------------------------------------------------------------

    if column in STATIC_PREGAME_COLUMNS:
        return (
            "static_pregame",
            True,
            "explicit pregame feature",
        )

    # ------------------------------------------------------------------------
    # 6. Explicit pregame advanced metrics
    # ------------------------------------------------------------------------

    if "_pregame_" in column:
        return (
            "static_pregame",
            True,
            "explicit pregame advanced metric",
        )

    # ------------------------------------------------------------------------
    # 7. Historical rolling features
    # ------------------------------------------------------------------------

    if any(
        marker in column
        for marker in ROLLING_MARKERS
    ):
        return (
            "rolling_pregame",
            True,
            "historical rolling feature",
        )

    # ------------------------------------------------------------------------
    # 8. Historical "Before" features
    # ------------------------------------------------------------------------

    if (
        "Before" in column
        or "before" in column
    ):
        return (
            "pregame",
            True,
            "historical pregame feature",
        )

    # ------------------------------------------------------------------------
    # 9. Current-game statistics
    # ------------------------------------------------------------------------

    base = remove_game_side_suffix(column)

    if base in CURRENT_GAME_FEATURES:
        return (
            "current_game",
            False,
            "current-game statistic",
        )

    # ------------------------------------------------------------------------
    # 10. Metadata patterns
    # ------------------------------------------------------------------------

    metadata_patterns = (
        "Classification",
        "Conference",
        "LineScores",
    )

    for pattern in metadata_patterns:

        if pattern in column:
            return (
                "metadata",
                False,
                f"metadata pattern: {pattern}",
            )

    # ------------------------------------------------------------------------
    # 11. ID patterns
    # ------------------------------------------------------------------------

    if (
        column.endswith("Id")
        or column.endswith("_id")
    ):
        return (
            "identifier",
            False,
            "ID-like column",
        )

    # ------------------------------------------------------------------------
    # 12. Fall back to old classification
    # ------------------------------------------------------------------------

    old_match = lookup_old_classification(
        column,
        old_lookup,
    )

    if old_match is not None:

        return (
            old_match["category"],
            old_match["predictive_safe"],
            "translated from previous classification",
        )

    # ------------------------------------------------------------------------
    # 13. Unresolved
    # ------------------------------------------------------------------------

    return (
        None,
        None,
        "UNRESOLVED",
    )


# ============================================================================
# CREATE CLASSIFICATION
# ============================================================================

def create_final_classification(
    final_columns,
    old_classification,
):
    """
    Create classification for all 448 final features.
    """

    print_header(
        "CLASSIFYING FINAL 448 FEATURE COLUMNS"
    )

    old_lookup = create_old_lookup(
        old_classification
    )

    rows = []
    unresolved = []

    for column in final_columns:

        category, predictive_safe, reason = (
            classify_final_column(
                column,
                old_lookup,
            )
        )

        if category is None:

            unresolved.append(column)

        rows.append({
            "column": column,
            "category": category,
            "predictive_safe": predictive_safe,
            "classification_source": (
                "final_schema_rule"
                if reason != "translated from previous classification"
                else "previous_classification"
            ),
            "classification_reason": reason,
        })

    classification = pd.DataFrame(rows)

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
        print("UNRESOLVED COLUMNS:")

        for column in unresolved:
            print(f"  {column}")

    return classification, unresolved


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

    required = {
        "column",
        "category",
        "predictive_safe",
    }

    missing_required = (
        required - set(classification.columns)
    )

    if missing_required:

        errors.append(
            "Missing required columns: "
            + ", ".join(sorted(missing_required))
        )

        print(
            "FAIL: Missing required columns:"
        )

        for column in sorted(missing_required):
            print(f"  {column}")

    else:

        print(
            "PASS: Required classification columns are present."
        )

    # ------------------------------------------------------------------------
    # Duplicate columns
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
            print(f"  {column}")

    else:

        print(
            "PASS: No duplicate classified feature names."
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

        for column in missing_categories["column"]:
            print(f"  {column}")

    else:

        print(
            "PASS: Every classified feature has a category."
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
            "One or more features have no predictive-safe flag."
        )

        for column in missing_safe["column"]:
            print(f"  {column}")

    else:

        print(
            "PASS: Every classified feature has a predictive-safe flag."
        )

    # ------------------------------------------------------------------------
    # Category validity
    # ------------------------------------------------------------------------

    invalid_categories = set(
        classification["category"].dropna()
    ) - VALID_CATEGORIES

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
            print(f"  {category}")

    else:

        print(
            "PASS: All categories are valid."
        )

    # ------------------------------------------------------------------------
    # Final coverage
    # ------------------------------------------------------------------------

    final_set = set(final_columns)

    classified_set = set(
        classification["column"]
    )

    missing_from_classification = (
        final_set - classified_set
    )

    classification_not_in_final = (
        classified_set - final_set
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
            "Final feature columns are missing from classification."
        )

        print()
        print("MISSING FROM CLASSIFICATION:")

        for column in sorted(
            missing_from_classification
        ):
            print(f"  {column}")

    else:

        print(
            "PASS: Every final feature is classified."
        )

    if classification_not_in_final:

        errors.append(
            "Classification contains columns not present "
            "in final feature data."
        )

        print()
        print("CLASSIFIED BUT NOT IN FINAL DATA:")

        for column in sorted(
            classification_not_in_final
        ):
            print(f"  {column}")

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

    for category, count in category_counts.items():
        print(
            f"  {category}: {count:,}"
        )

    # ------------------------------------------------------------------------
    # Predictive-safe counts
    # ------------------------------------------------------------------------

    print()
    print("Predictive-safe counts:")

    safe_counts = (
        classification["predictive_safe"]
        .value_counts()
        .sort_index()
    )

    for value, count in safe_counts.items():
        print(
            f"  {value}: {count:,}"
        )

    # ------------------------------------------------------------------------
    # Predictive-safe consistency
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

            inconsistent.append({
                "column": row["column"],
                "category": row["category"],
                "actual": actual_safe,
                "expected": expected_safe,
            })

    print()
    print(
        f"Predictive-safe/category inconsistencies: "
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
            "PASS: Predictive-safe flags are consistent "
            "with feature categories."
        )

    return errors


# ============================================================================
# SAVE
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

    classification = classification.sort_values(
        "column"
    ).reset_index(drop=True)

    classification.to_csv(
        FINAL_CLASSIFICATION_PATH,
        index=False,
    )

    print_header(
        "SAVED FINAL FEATURE CLASSIFICATION"
    )

    print(
        f"Path:"
    )

    print(
        f"  {FINAL_CLASSIFICATION_PATH}"
    )

    print()
    print(
        f"Rows: {len(classification):,}"
    )

    print(
        f"Columns: {len(classification.columns):,}"
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print_header(
        "CREATING FINAL FEATURE CLASSIFICATION"
    )

    print(
        "Final feature directory:"
    )

    print(
        f"  {FINAL_FEATURE_DIR}"
    )

    # ------------------------------------------------------------------------
    # Load old classification
    # ------------------------------------------------------------------------

    print_header(
        "LOADING EXISTING CLASSIFICATION"
    )

    old_classification = (
        load_old_classification()
    )

    # ------------------------------------------------------------------------
    # Load actual final 448-column schema
    # ------------------------------------------------------------------------

    final_columns = (
        load_final_feature_columns()
    )

    # ------------------------------------------------------------------------
    # Create classification
    # ------------------------------------------------------------------------

    classification, unresolved = (
        create_final_classification(
            final_columns,
            old_classification,
        )
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
            f"feature(s) remain unclassified."
        )

        print()
        print(
            "The classification file was NOT saved."
        )

        print(
            "Resolve the remaining columns before proceeding."
        )

        sys.exit(1)

    # ------------------------------------------------------------------------
    # Validate
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
        "PASS: Final feature classification created successfully."
    )

    print()
    print(
        "Next step:"
    )

    print(
        "  python src\\data\\validate\\validate_final_feature_classification.py"
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()