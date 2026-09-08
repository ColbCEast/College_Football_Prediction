"""
Correct Feature Classification Metadata

Purpose:
    Correct the predictive-safe classification of metadata fields that should
    not be used as model predictors.

Corrections:
    - startDate  -> predictive_safe = False
    - seasonType -> predictive_safe = False

Expected result:
    310 predictive-safe features

Input:
    data/metadata/final_feature_classification.csv

Output:
    The same CSV, corrected in place.

This is a one-time metadata correction script.
"""


# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path

import pandas as pd


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CLASSIFICATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "final_feature_classification.csv"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

FEATURES_TO_RECLASSIFY = [
    "startDate",
    "seasonType",
]

EXPECTED_PREDICTIVE_SAFE_COUNT = 310


# =============================================================================
# LOAD DATA
# =============================================================================

def load_classification():
    """Load the feature classification metadata."""

    print("=" * 80)
    print("LOADING FEATURE CLASSIFICATION")
    print("=" * 80)

    print()
    print("Project root:")
    print(f"  {PROJECT_ROOT}")

    print()
    print("Classification file:")
    print(f"  {CLASSIFICATION_PATH}")

    if not CLASSIFICATION_PATH.exists():
        raise FileNotFoundError(
            "Could not find final_feature_classification.csv at:\n"
            f"{CLASSIFICATION_PATH}"
        )

    df = pd.read_csv(CLASSIFICATION_PATH)

    print()
    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    return df


# =============================================================================
# VALIDATE STRUCTURE
# =============================================================================

def validate_structure(df):
    """Validate that the metadata contains the required columns."""

    print()
    print("=" * 80)
    print("VALIDATING METADATA STRUCTURE")
    print("=" * 80)

    required_columns = {
        "column",
        "predictive_safe",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "The classification file is missing required columns:\n"
            f"{sorted(missing_columns)}"
        )

    print("Required columns: PASSED")


# =============================================================================
# VALIDATE TARGET FEATURES
# =============================================================================

def validate_target_features(df):
    """Confirm each feature appears exactly once."""

    print()
    print("=" * 80)
    print("VALIDATING FEATURES TO RECLASSIFY")
    print("=" * 80)

    for feature in FEATURES_TO_RECLASSIFY:

        matches = df["column"].eq(feature)
        count = int(matches.sum())

        print()
        print(f"{feature}:")
        print(f"  Occurrences: {count}")

        if count != 1:
            raise ValueError(
                f"Expected exactly one row for {feature!r}, "
                f"but found {count}."
            )

    print()
    print("Feature existence validation: PASSED")


# =============================================================================
# DISPLAY CURRENT CLASSIFICATION
# =============================================================================

def display_current_classification(df):
    """Display the metadata rows before modification."""

    print()
    print("=" * 80)
    print("CURRENT CLASSIFICATION")
    print("=" * 80)

    rows = df[
        df["column"].isin(FEATURES_TO_RECLASSIFY)
    ]

    print()
    print(rows.to_string(index=False))


# =============================================================================
# CORRECT CLASSIFICATION
# =============================================================================

def correct_classification(df):
    """
    Reclassify startDate and seasonType as not predictive-safe.

    Only the predictive_safe column is modified.
    """

    print()
    print("=" * 80)
    print("CORRECTING CLASSIFICATION")
    print("=" * 80)

    mask = df["column"].isin(FEATURES_TO_RECLASSIFY)

    original_values = (
        df.loc[mask, ["column", "predictive_safe"]]
        .copy()
    )

    # Make the correction.
    df.loc[mask, "predictive_safe"] = False

    print()
    print("Changes made:")

    for _, row in original_values.iterrows():

        feature = row["column"]
        old_value = row["predictive_safe"]

        new_value = df.loc[
            df["column"].eq(feature),
            "predictive_safe"
        ].iloc[0]

        print(
            f"  {feature}: "
            f"{old_value} -> {new_value}"
        )

    return df


# =============================================================================
# VALIDATE CORRECTION
# =============================================================================

def validate_correction(df):
    """Verify that the corrected metadata is internally consistent."""

    print()
    print("=" * 80)
    print("VALIDATING CORRECTION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Confirm both fields are no longer predictive-safe
    # -------------------------------------------------------------------------

    for feature in FEATURES_TO_RECLASSIFY:

        value = df.loc[
            df["column"].eq(feature),
            "predictive_safe"
        ].iloc[0]

        if bool(value):
            raise ValueError(
                f"{feature!r} was not successfully reclassified."
            )

    print()
    print("Metadata fields excluded from predictors: PASSED")

    # -------------------------------------------------------------------------
    # Count predictive-safe features
    # -------------------------------------------------------------------------

    predictive_safe_count = int(
        df["predictive_safe"].eq(True).sum()
    )

    print()
    print(
        f"Predictive-safe features after correction: "
        f"{predictive_safe_count}"
    )

    if predictive_safe_count != EXPECTED_PREDICTIVE_SAFE_COUNT:
        raise ValueError(
            "Unexpected predictive-safe feature count.\n"
            f"Expected: {EXPECTED_PREDICTIVE_SAFE_COUNT}\n"
            f"Found:    {predictive_safe_count}"
        )

    print()
    print("Predictive-safe feature count: PASSED")

    # -------------------------------------------------------------------------
    # Display corrected rows
    # -------------------------------------------------------------------------

    print()
    print("Corrected classification:")

    rows = df[
        df["column"].isin(FEATURES_TO_RECLASSIFY)
    ]

    print()
    print(rows.to_string(index=False))

    return predictive_safe_count


# =============================================================================
# SAVE
# =============================================================================

def save_classification(df):
    """Save the corrected metadata back to the original CSV."""

    print()
    print("=" * 80)
    print("SAVING CORRECTED CLASSIFICATION")
    print("=" * 80)

    df.to_csv(
        CLASSIFICATION_PATH,
        index=False,
    )

    print()
    print("Saved successfully:")
    print(f"  {CLASSIFICATION_PATH}")


# =============================================================================
# MAIN
# =============================================================================

def main():

    print()
    print("=" * 80)
    print("FEATURE CLASSIFICATION CORRECTION")
    print("=" * 80)

    # =========================================================================
    # 1. Load
    # =========================================================================

    df = load_classification()

    # =========================================================================
    # 2. Validate structure
    # =========================================================================

    validate_structure(df)

    # =========================================================================
    # 3. Validate target features
    # =========================================================================

    validate_target_features(df)

    # =========================================================================
    # 4. Show current classification
    # =========================================================================

    display_current_classification(df)

    # =========================================================================
    # 5. Correct classification
    # =========================================================================

    df = correct_classification(df)

    # =========================================================================
    # 6. Validate correction
    # =========================================================================

    predictive_safe_count = validate_correction(df)

    # =========================================================================
    # 7. Save
    # =========================================================================

    save_classification(df)

    # =========================================================================
    # 8. Final summary
    # =========================================================================

    print()
    print("=" * 80)
    print("FEATURE CLASSIFICATION CORRECTION COMPLETE")
    print("=" * 80)

    print()
    print("Reclassified:")
    print("  - startDate  -> predictive_safe = False")
    print("  - seasonType -> predictive_safe = False")

    print()
    print(
        f"Final predictive-safe feature count: "
        f"{predictive_safe_count}"
    )

    print()
    print("The metadata is now ready for Gradient Boosting Model 5.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()