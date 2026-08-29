import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

CLASSIFICATION_PATH = (
    "data/metadata/final_feature_classification.csv"
)

FINAL_FEATURE_PATH = (
    "data/processed/features/final/"
)


# ============================================================
# EXPECTED SEASONS
# ============================================================

SEASONS = range(2015, 2026)


# ============================================================
# REQUIRED CLASSIFICATION COLUMNS
# ============================================================

REQUIRED_CLASSIFICATION_COLUMNS = [
    "column",
    "category",
    "predictive_safe",
]


# ============================================================
# LOAD CLASSIFICATION
# ============================================================

def load_classification():

    if not os.path.exists(CLASSIFICATION_PATH):

        raise FileNotFoundError(
            f"Feature classification file does not exist:\n"
            f"{CLASSIFICATION_PATH}"
        )

    classification = pd.read_csv(
        CLASSIFICATION_PATH
    )

    print("\nLoaded feature classification:")
    print(
        f"  Rows:    {len(classification)}"
    )
    print(
        f"  Columns: {len(classification.columns)}"
    )

    return classification


# ============================================================
# LOAD FINAL FEATURES
# ============================================================

def load_final_features(year):

    path = (
        f"{FINAL_FEATURE_PATH}"
        f"final_features_{year}.csv"
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Final feature file does not exist:\n"
            f"{path}"
        )

    return pd.read_csv(path)


# ============================================================
# VALIDATE CLASSIFICATION STRUCTURE
# ============================================================

def validate_classification_structure(
    classification
):

    print("\n" + "=" * 70)
    print("VALIDATING CLASSIFICATION STRUCTURE")
    print("=" * 70)

    passed = True

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_CLASSIFICATION_COLUMNS
        if column not in classification.columns
    ]

    if missing_columns:

        print(
            "\nFAIL: Classification file is missing "
            "required columns:"
        )

        for column in missing_columns:
            print(f"  {column}")

        return False

    print(
        "\nPASS: Required classification columns "
        "are present."
    )

    # --------------------------------------------------------
    # Duplicate classified columns
    # --------------------------------------------------------

    duplicate_columns = (
        classification["column"]
        .value_counts()
        .loc[lambda x: x > 1]
    )

    print(
        "\nDuplicate classified feature names:"
    )
    print(
        f"  {len(duplicate_columns)}"
    )

    if len(duplicate_columns) > 0:

        print(
            "  FAIL: Features appear more than "
            "once in classification."
        )

        print(duplicate_columns)

        passed = False

    else:

        print(
            "  PASS: No duplicate classified "
            "feature names."
        )

    # --------------------------------------------------------
    # Missing category
    # --------------------------------------------------------

    missing_categories = (
        classification["category"]
        .isna()
        .sum()
    )

    print(
        "\nMissing feature categories:"
    )
    print(
        f"  {missing_categories}"
    )

    if missing_categories > 0:

        print(
            "  FAIL: Some features do not have "
            "a category."
        )

        passed = False

    else:

        print(
            "  PASS: Every classified feature "
            "has a category."
        )

    # --------------------------------------------------------
    # Missing predictive-safe flag
    # --------------------------------------------------------

    missing_safety = (
        classification["predictive_safe"]
        .isna()
        .sum()
    )

    print(
        "\nMissing predictive-safe flags:"
    )
    print(
        f"  {missing_safety}"
    )

    if missing_safety > 0:

        print(
            "  FAIL: Some features do not have "
            "a predictive-safe classification."
        )

        passed = False

    else:

        print(
            "  PASS: Every classified feature "
            "has a predictive-safe flag."
        )

    return passed


# ============================================================
# LOAD FINAL FEATURE COLUMNS
# ============================================================

def get_final_feature_columns():

    final_columns = {}

    for year in SEASONS:

        df = load_final_features(year)

        final_columns[year] = set(
            df.columns
        )

    return final_columns


# ============================================================
# VALIDATE FINAL COLUMN CONSISTENCY
# ============================================================

def validate_final_column_consistency(
    final_columns
):

    print("\n" + "=" * 70)
    print("VALIDATING FINAL COLUMN CONSISTENCY")
    print("=" * 70)

    reference_year = 2015

    reference_columns = final_columns[
        reference_year
    ]

    passed = True

    print(
        f"\nReference season: {reference_year}"
    )
    print(
        f"Reference columns: "
        f"{len(reference_columns)}"
    )

    for year in SEASONS:

        current_columns = final_columns[year]

        missing = (
            reference_columns
            - current_columns
        )

        extra = (
            current_columns
            - reference_columns
        )

        print(
            f"\n{year}:"
        )
        print(
            f"  Columns: {len(current_columns)}"
        )
        print(
            f"  Missing vs {reference_year}: "
            f"{len(missing)}"
        )
        print(
            f"  Extra vs {reference_year}: "
            f"{len(extra)}"
        )

        if missing:

            print(
                "  Missing columns:"
            )

            for column in sorted(missing):
                print(
                    f"    {column}"
                )

            passed = False

        if extra:

            print(
                "  Extra columns:"
            )

            for column in sorted(extra):
                print(
                    f"    {column}"
                )

            passed = False

    if passed:

        print(
            "\nPASS: All final feature datasets "
            "have identical column structures."
        )

    else:

        print(
            "\nFAIL: Final feature datasets "
            "do not have identical column structures."
        )

    return passed


# ============================================================
# VALIDATE CLASSIFICATION AGAINST FINAL FEATURES
# ============================================================

def validate_classification_coverage(
    classification,
    final_columns
):

    print("\n" + "=" * 70)
    print("VALIDATING CLASSIFICATION COVERAGE")
    print("=" * 70)

    classified_columns = set(
        classification["column"]
    )

    # Union of all final feature columns
    all_final_columns = set().union(
        *final_columns.values()
    )

    # --------------------------------------------------------
    # Final columns not classified
    # --------------------------------------------------------

    unclassified = (
        all_final_columns
        - classified_columns
    )

    # --------------------------------------------------------
    # Classified columns not present in final data
    # --------------------------------------------------------

    obsolete = (
        classified_columns
        - all_final_columns
    )

    print(
        "\nFinal feature columns:"
    )
    print(
        f"  {len(all_final_columns)}"
    )

    print(
        "\nClassified feature columns:"
    )
    print(
        f"  {len(classified_columns)}"
    )

    print(
        "\nUnclassified final feature columns:"
    )
    print(
        f"  {len(unclassified)}"
    )

    if unclassified:

        print(
            "\n  UNCLASSIFIED COLUMNS:"
        )

        for column in sorted(unclassified):
            print(
                f"    {column}"
            )

    else:

        print(
            "  PASS: Every final feature "
            "column is classified."
        )

    print(
        "\nClassified columns not present "
        "in final features:"
    )
    print(
        f"  {len(obsolete)}"
    )

    if obsolete:

        print(
            "\n  CLASSIFIED BUT NOT IN FINAL DATA:"
        )

        for column in sorted(obsolete):
            print(
                f"    {column}"
            )

    else:

        print(
            "  PASS: Every classified feature "
            "exists in the final dataset."
        )

    return (
        len(unclassified) == 0
        and len(obsolete) == 0
    )


# ============================================================
# PRINT CLASSIFICATION COUNTS
# ============================================================

def print_classification_counts(
    classification
):

    print("\n" + "=" * 70)
    print("CURRENT FEATURE CLASSIFICATION")
    print("=" * 70)

    category_counts = (
        classification["category"]
        .value_counts()
        .sort_index()
    )

    print("\nCategory counts:")

    for category, count in category_counts.items():

        print(
            f"  {category}: {count}"
        )

    # --------------------------------------------------------
    # Predictive safety
    # --------------------------------------------------------

    print(
        "\nPredictive-safe counts:"
    )

    safety_counts = (
        classification["predictive_safe"]
        .value_counts()
    )

    for value, count in safety_counts.items():

        print(
            f"  {value}: {count}"
        )


# ============================================================
# COMPARE EACH SEASON TO CLASSIFICATION
# ============================================================

def validate_each_season(
    classification,
    final_columns
):

    print("\n" + "=" * 70)
    print("VALIDATING EACH SEASON")
    print("=" * 70)

    classified_columns = set(
        classification["column"]
    )

    all_passed = True

    for year in SEASONS:

        columns = final_columns[year]

        missing_classification = (
            columns
            - classified_columns
        )

        print(
            f"\n{year}:"
        )

        print(
            f"  Final columns: "
            f"{len(columns)}"
        )

        print(
            f"  Classified columns present: "
            f"{len(columns & classified_columns)}"
        )

        print(
            f"  Unclassified columns: "
            f"{len(missing_classification)}"
        )

        if missing_classification:

            print(
                "  FAIL: Unclassified columns:"
            )

            for column in sorted(
                missing_classification
            ):
                print(
                    f"    {column}"
                )

            all_passed = False

        else:

            print(
                "  PASS: All season columns "
                "are classified."
            )

    return all_passed


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print(
        "VALIDATING FINAL FEATURE CLASSIFICATION"
    )
    print("=" * 70)

    all_passed = True

    # --------------------------------------------------------
    # 1. Load classification
    # --------------------------------------------------------

    classification = load_classification()

    # --------------------------------------------------------
    # 2. Validate classification structure
    # --------------------------------------------------------

    classification_structure_valid = (
        validate_classification_structure(
            classification
        )
    )

    if not classification_structure_valid:
        all_passed = False

    # --------------------------------------------------------
    # 3. Print current classification
    # --------------------------------------------------------

    print_classification_counts(
        classification
    )

    # --------------------------------------------------------
    # 4. Load final feature columns
    # --------------------------------------------------------

    final_columns = get_final_feature_columns()

    # --------------------------------------------------------
    # 5. Validate column consistency
    # --------------------------------------------------------

    column_consistency_valid = (
        validate_final_column_consistency(
            final_columns
        )
    )

    if not column_consistency_valid:
        all_passed = False

    # --------------------------------------------------------
    # 6. Validate classification coverage
    # --------------------------------------------------------

    classification_coverage_valid = (
        validate_classification_coverage(
            classification,
            final_columns
        )
    )

    if not classification_coverage_valid:
        all_passed = False

    # --------------------------------------------------------
    # 7. Validate each season
    # --------------------------------------------------------

    season_validation_valid = (
        validate_each_season(
            classification,
            final_columns
        )
    )

    if not season_validation_valid:
        all_passed = False

    # --------------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL VALIDATION RESULT")
    print("=" * 70)

    if all_passed:

        print(
            "PASS: Final feature classification "
            "is complete and consistent."
        )

        print(
            "\nAll final feature columns are "
            "represented in the classification file."
        )

    else:

        print(
            "FAIL: Final feature classification "
            "requires investigation."
        )

        print(
            "\nDo NOT proceed to modeling until "
            "the unclassified or inconsistent "
            "features have been resolved."
        )