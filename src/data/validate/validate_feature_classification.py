import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# Paths
# ============================================================

MASTER_PATH = "data/master/master_game_data.csv"
CLASSIFICATION_PATH = "data/metadata/feature_classification.csv"


# ============================================================
# Load data
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

print("\nLoading master game-level data...")
df = pd.read_csv(MASTER_PATH)

print(f"Master shape: {df.shape}")
print(f"Master columns: {len(df.columns)}")

print("\nLoading feature classification...")
feature_df = pd.read_csv(CLASSIFICATION_PATH)

print(f"Classification shape: {feature_df.shape}")
print(f"Classification columns: {len(feature_df.columns)}")


# ============================================================
# Helper function
# ============================================================

def print_result(check_name, passed, details=""):
    """
    Print a standardized validation result.
    """

    if passed:
        print(f"✓ {check_name}")
    else:
        print(f"✗ {check_name}")

    if details:
        print(f"  {details}")


# ============================================================
# 1. Basic classification structure
# ============================================================

print("\n" + "=" * 70)
print("1. CLASSIFICATION STRUCTURE")
print("=" * 70)


# ------------------------------------------------------------
# Check required classification columns
# ------------------------------------------------------------

required_classification_columns = {
    "column",
    "category",
    "predictive_safe",
    "target_candidate",
    "leakage_risk",
    "reason",
    "notes",
}

missing_classification_columns = (
    required_classification_columns
    - set(feature_df.columns)
)

print_result(
    "Required classification columns exist",
    len(missing_classification_columns) == 0,
    (
        f"Missing: {sorted(missing_classification_columns)}"
        if missing_classification_columns
        else ""
    )
)


# ------------------------------------------------------------
# Check duplicate classification rows
# ------------------------------------------------------------

duplicate_columns = feature_df[
    feature_df["column"].duplicated(keep=False)
]["column"].unique()

print_result(
    "No duplicate feature classifications",
    len(duplicate_columns) == 0,
    (
        f"Duplicates: {list(duplicate_columns)}"
        if len(duplicate_columns) > 0
        else ""
    )
)


# ------------------------------------------------------------
# Check for dataset columns missing from classification
# ------------------------------------------------------------

dataset_columns = set(df.columns)
classified_columns = set(feature_df["column"])

missing_from_classification = (
    dataset_columns - classified_columns
)

print_result(
    "Every dataset column has a classification",
    len(missing_from_classification) == 0,
    (
        f"Missing: {sorted(missing_from_classification)}"
        if missing_from_classification
        else ""
    )
)


# ------------------------------------------------------------
# Check for classification columns not in dataset
# ------------------------------------------------------------

extra_classification_columns = (
    classified_columns - dataset_columns
)

print_result(
    "No classifications exist for nonexistent columns",
    len(extra_classification_columns) == 0,
    (
        f"Extra: {sorted(extra_classification_columns)}"
        if extra_classification_columns
        else ""
    )
)


# ------------------------------------------------------------
# Check total count
# ------------------------------------------------------------

print_result(
    "Classification count matches dataset column count",
    len(feature_df) == len(df.columns),
    (
        f"Dataset: {len(df.columns)}, "
        f"Classification: {len(feature_df)}"
    )
)


# ============================================================
# 2. Validate category definitions
# ============================================================

print("\n" + "=" * 70)
print("2. CATEGORY VALIDATION")
print("=" * 70)


allowed_categories = {
    "pregame",
    "rolling_pregame",
    "static_pregame",
    "current_game",
    "target_candidate",
    "metadata",
    "identifier",
}

invalid_categories = set(
    feature_df["category"].dropna()
) - allowed_categories

print_result(
    "All categories are recognized",
    len(invalid_categories) == 0,
    (
        f"Invalid categories: {sorted(invalid_categories)}"
        if invalid_categories
        else ""
    )
)


# ============================================================
# 3. Validate predictive-safe logic
# ============================================================

print("\n" + "=" * 70)
print("3. PREDICTIVE-SAFE LOGIC")
print("=" * 70)


pregame_categories = {
    "pregame",
    "rolling_pregame",
    "static_pregame",
}

unsafe_categories = {
    "current_game",
    "target_candidate",
    "metadata",
    "identifier",
}


# ------------------------------------------------------------
# Pregame categories should be predictive safe
# ------------------------------------------------------------

pregame_rows = feature_df[
    feature_df["category"].isin(pregame_categories)
]

pregame_not_safe = pregame_rows[
    pregame_rows["predictive_safe"] != True
]

print_result(
    "All pregame categories are marked predictive-safe",
    len(pregame_not_safe) == 0,
    (
        f"Problem columns: "
        f"{pregame_not_safe['column'].tolist()}"
        if len(pregame_not_safe) > 0
        else ""
    )
)


# ------------------------------------------------------------
# Unsafe categories should not be predictive safe
# ------------------------------------------------------------

unsafe_rows = feature_df[
    feature_df["category"].isin(unsafe_categories)
]

unsafe_marked_safe = unsafe_rows[
    unsafe_rows["predictive_safe"] == True
]

print_result(
    "No unsafe categories are marked predictive-safe",
    len(unsafe_marked_safe) == 0,
    (
        f"Problem columns: "
        f"{unsafe_marked_safe['column'].tolist()}"
        if len(unsafe_marked_safe) > 0
        else ""
    )
)


# ============================================================
# 4. Target candidate validation
# ============================================================

print("\n" + "=" * 70)
print("4. TARGET VALIDATION")
print("=" * 70)


target_rows = feature_df[
    feature_df["target_candidate"] == True
]


# Target candidates should not be predictive safe

target_marked_safe = target_rows[
    target_rows["predictive_safe"] == True
]

print_result(
    "Target candidates are not marked predictive-safe",
    len(target_marked_safe) == 0,
    (
        f"Problem columns: "
        f"{target_marked_safe['column'].tolist()}"
        if len(target_marked_safe) > 0
        else ""
    )
)


print("\nTarget candidates:")

for column in target_rows["column"]:
    print(f"  {column}")


# ============================================================
# 5. Current-game validation
# ============================================================

print("\n" + "=" * 70)
print("5. CURRENT-GAME VALIDATION")
print("=" * 70)


current_game_rows = feature_df[
    feature_df["category"] == "current_game"
]

current_game_safe = current_game_rows[
    current_game_rows["predictive_safe"] == True
]

print_result(
    "Current-game features are not predictive-safe",
    len(current_game_safe) == 0,
    (
        f"Problem columns: "
        f"{current_game_safe['column'].tolist()}"
        if len(current_game_safe) > 0
        else ""
    )
)


# ============================================================
# 6. Identifier validation
# ============================================================

print("\n" + "=" * 70)
print("6. IDENTIFIER VALIDATION")
print("=" * 70)


identifier_rows = feature_df[
    feature_df["category"] == "identifier"
]

identifier_safe = identifier_rows[
    identifier_rows["predictive_safe"] == True
]

print_result(
    "Identifiers are not predictive-safe",
    len(identifier_safe) == 0,
    (
        f"Problem columns: "
        f"{identifier_safe['column'].tolist()}"
        if len(identifier_safe) > 0
        else ""
    )
)


# ============================================================
# 7. Metadata validation
# ============================================================

print("\n" + "=" * 70)
print("7. METADATA VALIDATION")
print("=" * 70)


metadata_rows = feature_df[
    feature_df["category"] == "metadata"
]

metadata_safe = metadata_rows[
    metadata_rows["predictive_safe"] == True
]

print_result(
    "Metadata is not automatically predictive-safe",
    len(metadata_safe) == 0,
    (
        f"Columns marked safe: "
        f"{metadata_safe['column'].tolist()}"
        if len(metadata_safe) > 0
        else ""
    )
)


# ============================================================
# 8. Validate Before columns
# ============================================================

print("\n" + "=" * 70)
print("8. PREGAME COLUMN NAME VALIDATION")
print("=" * 70)


before_columns = [
    column
    for column in df.columns
    if "Before" in column
]


before_classifications = feature_df[
    feature_df["column"].isin(before_columns)
]


before_not_pregame = before_classifications[
    ~before_classifications["category"].isin(
        {"pregame"}
    )
]


print_result(
    "All 'Before' columns are classified as pregame",
    len(before_not_pregame) == 0,
    (
        f"Problem columns: "
        f"{before_not_pregame['column'].tolist()}"
        if len(before_not_pregame) > 0
        else ""
    )
)


# ============================================================
# 9. Validate rolling columns
# ============================================================

print("\n" + "=" * 70)
print("9. ROLLING FEATURE VALIDATION")
print("=" * 70)


rolling_columns = [
    column
    for column in df.columns
    if column.endswith("Last3")
    or column.endswith("Last5")
]


rolling_classifications = feature_df[
    feature_df["column"].isin(rolling_columns)
]


rolling_not_classified_correctly = rolling_classifications[
    rolling_classifications["category"] != "rolling_pregame"
]


print_result(
    "All Last3/Last5 columns are classified as rolling pregame",
    len(rolling_not_classified_correctly) == 0,
    (
        f"Problem columns: "
        f"{rolling_not_classified_correctly['column'].tolist()}"
        if len(rolling_not_classified_correctly) > 0
        else ""
    )
)


# ============================================================
# 10. Check first-game winPctBefore behavior
# ============================================================

print("\n" + "=" * 70)
print("10. winPctBefore VALIDATION")
print("=" * 70)


if {
    "gamesBefore",
    "winsBefore",
    "winPctBefore",
}.issubset(df.columns):

    # Expected win percentage based on previous games.
    #
    # For a team's first game, there are no previous games,
    # so NaN is considered correct.
    expected_win_pct = np.where(
        df["gamesBefore"] == 0,
        np.nan,
        df["winsBefore"] / df["gamesBefore"]
    )

    actual_win_pct = df["winPctBefore"].to_numpy()

    valid_comparison = (
        ~np.isnan(expected_win_pct)
        & ~pd.isna(actual_win_pct)
    )

    mismatches = (
        valid_comparison
        & ~np.isclose(
            expected_win_pct,
            actual_win_pct,
            rtol=1e-9,
            atol=1e-9
        )
    )

    # First-game NaN check
    first_game_rows = df["gamesBefore"] == 0

    first_game_not_nan = (
        first_game_rows
        & df["winPctBefore"].notna()
    )

    mismatch_count = mismatches.sum()
    first_game_problem_count = first_game_not_nan.sum()

    print_result(
        "winPctBefore matches winsBefore / gamesBefore",
        mismatch_count == 0,
        f"Mismatches: {mismatch_count}"
    )

    print_result(
        "First-game winPctBefore values are NaN",
        first_game_problem_count == 0,
        f"Problems: {first_game_problem_count}"
    )

else:

    print("Required columns for winPctBefore validation are missing.")


# ============================================================
# 11. Check predictive-safe columns for suspicious names
# ============================================================

print("\n" + "=" * 70)
print("11. PREDICTIVE-SAFE SUSPICIOUS COLUMN CHECK")
print("=" * 70)


suspicious_terms = {
    "points",
    "yards",
    "attempts",
    "completions",
    "touchdowns",
    "turnovers",
    "sacks",
    "tackles",
    "penalties",
    "possession",
    "interceptions",
    "fumbles",
    "returns",
    "firstDowns",
    "thirdDown",
    "fourthDown",
}


safe_features = feature_df[
    feature_df["predictive_safe"] == True
]["column"]


suspicious_safe_features = []

for column in safe_features:

    # These terms are okay if the feature explicitly indicates
    # that it is based on previous games.
    has_pregame_indicator = (
        "Before" in column
        or "Pregame" in column
        or column.endswith("Last3")
        or column.endswith("Last5")
    )

    if not has_pregame_indicator:

        if any(term.lower() in column.lower()
               for term in suspicious_terms):

            suspicious_safe_features.append(column)


print_result(
    "No unexplained current-statistic names are marked safe",
    len(suspicious_safe_features) == 0,
    (
        f"Review: {suspicious_safe_features}"
        if suspicious_safe_features
        else ""
    )
)


# ============================================================
# 12. Summary
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)


print(f"\nDataset columns:          {len(df.columns)}")
print(f"Classification rows:     {len(feature_df)}")

print(
    f"\nPredictive-safe features: "
    f"{feature_df['predictive_safe'].sum()}"
)

print(
    f"Target candidates:        "
    f"{feature_df['target_candidate'].sum()}"
)

print(
    f"Current-game features:    "
    f"{(feature_df['category'] == 'current_game').sum()}"
)

print(
    f"Metadata columns:         "
    f"{(feature_df['category'] == 'metadata').sum()}"
)

print(
    f"Identifier columns:       "
    f"{(feature_df['category'] == 'identifier').sum()}"
)


# ============================================================
# Final status
# ============================================================

all_structural_checks = (
    len(missing_classification_columns) == 0
    and len(duplicate_columns) == 0
    and len(missing_from_classification) == 0
    and len(extra_classification_columns) == 0
    and len(feature_df) == len(df.columns)
    and len(invalid_categories) == 0
    and len(pregame_not_safe) == 0
    and len(unsafe_marked_safe) == 0
    and len(target_marked_safe) == 0
    and len(current_game_safe) == 0
    and len(identifier_safe) == 0
    and len(metadata_safe) == 0
    and len(before_not_pregame) == 0
    and len(rolling_not_classified_correctly) == 0
    and (
        "gamesBefore" not in df.columns
        or "winsBefore" not in df.columns
        or "winPctBefore" not in df.columns
        or (
            mismatch_count == 0
            and first_game_problem_count == 0
        )
    )
    and len(suspicious_safe_features) == 0
)


print("\n" + "=" * 70)

if all_structural_checks:
    print("✓ FEATURE CLASSIFICATION VALIDATION PASSED")
    print("=" * 70)
    print("\nThe classification metadata is internally consistent.")
    print("No unexplained predictive-safe features were detected.")
else:
    print("⚠ FEATURE CLASSIFICATION VALIDATION REQUIRES REVIEW")
    print("=" * 70)
    print("\nOne or more checks failed. Review the sections above.")

print()