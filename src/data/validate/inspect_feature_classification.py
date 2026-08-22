import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

CLASSIFICATION_PATH = (
    "data/metadata/feature_classification.csv"
)

FINAL_FEATURES_PATH = (
    "data/processed/final_features/final_features_2015.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("INSPECTING UNKNOWN FEATURE CLASSIFICATION")
print("=" * 70)

classification = pd.read_csv(
    CLASSIFICATION_PATH
)

final_features = pd.read_csv(
    FINAL_FEATURES_PATH,
    nrows=5
)

print("\nLoaded data:")
print(f"  Classification rows: {len(classification):,}")
print(f"  Final feature columns: {len(final_features.columns):,}")


# ============================================================
# IDENTIFY UNKNOWN FEATURES
# ============================================================

unknown = classification[
    classification["category"] == "unknown"
].copy()

print("\n" + "=" * 70)
print("UNKNOWN FEATURE SUMMARY")
print("=" * 70)

print(f"\nUnknown features: {len(unknown):,}")


if len(unknown) == 0:

    print("\nPASS: No unknown features require inspection.")

    raise SystemExit


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def infer_feature_context(column):
    """
    Identify useful structural information from a feature name.
    This does NOT classify the feature.
    It only provides information for manual inspection.
    """

    context = []

    # Home / away
    if column.startswith("home"):
        context.append("HOME")

    elif column.startswith("away"):
        context.append("AWAY")

    # Pregame indicators
    if "Before" in column:
        context.append("CONTAINS_BEFORE")

    if "Pregame" in column:
        context.append("CONTAINS_PREGAME")

    # Rolling indicators
    if column.endswith("Last3"):
        context.append("ROLLING_LAST3")

    if column.endswith("Last5"):
        context.append("ROLLING_LAST5")

    # Common historical indicators
    if "Previous" in column:
        context.append("CONTAINS_PREVIOUS")

    if "Prior" in column:
        context.append("CONTAINS_PRIOR")

    if "Rolling" in column:
        context.append("CONTAINS_ROLLING")

    if "Avg" in column:
        context.append("AVERAGE")

    if "Average" in column:
        context.append("AVERAGE")

    if "Mean" in column:
        context.append("MEAN")

    if "Total" in column:
        context.append("TOTAL")

    # Common game-stat terminology
    game_stat_terms = [
        "points",
        "yards",
        "attempts",
        "completions",
        "touchdowns",
        "TDs",
        "turnovers",
        "interceptions",
        "fumbles",
        "sacks",
        "tackles",
        "penalties",
        "possession",
        "downs",
        "returns",
    ]

    matched_terms = []

    lower_column = column.lower()

    for term in game_stat_terms:

        if term.lower() in lower_column:
            matched_terms.append(term)

    if matched_terms:
        context.append(
            "STAT=" + ", ".join(matched_terms)
        )

    return context


def suggest_category(column):
    """
    Provide a NON-AUTHORITATIVE suggestion based on
    obvious naming patterns.

    These suggestions are intentionally conservative.
    """

    suggestions = []

    # Explicit pregame indicators
    if "Before" in column or "Pregame" in column:
        suggestions.append("pregame")

    # Rolling indicators
    if (
        column.endswith("Last3")
        or column.endswith("Last5")
        or "Rolling" in column
    ):
        suggestions.append("rolling_pregame")

    # Current-game terminology
    current_game_terms = [
        "points",
        "yards",
        "attempts",
        "completions",
        "touchdowns",
        "TDs",
        "turnovers",
        "interceptions",
        "fumbles",
        "sacks",
        "tackles",
        "penalties",
        "possession",
        "downs",
        "returns",
    ]

    lower_column = column.lower()

    if any(
        term.lower() in lower_column
        for term in current_game_terms
    ):
        suggestions.append("current_game")

    # Remove duplicate suggestions while preserving order
    suggestions = list(dict.fromkeys(suggestions))

    if len(suggestions) == 0:
        return "NO_AUTOMATIC_SUGGESTION"

    if len(suggestions) == 1:
        return suggestions[0]

    return " / ".join(suggestions)


# ============================================================
# BUILD INSPECTION TABLE
# ============================================================

inspection_records = []

for _, row in unknown.iterrows():

    column = row["column"]

    context = infer_feature_context(column)

    suggestion = suggest_category(column)

    inspection_records.append({
        "column": column,
        "current_category": row["category"],
        "predictive_safe_current": row["predictive_safe"],
        "target_candidate_current": row["target_candidate"],
        "leakage_risk_current": row["leakage_risk"],
        "automatic_suggestion": suggestion,
        "feature_context": " | ".join(context)
            if context
            else "NO_OBVIOUS_CONTEXT",
    })


inspection_df = pd.DataFrame(
    inspection_records
)


# ============================================================
# PRINT INSPECTION REPORT
# ============================================================

print("\n" + "=" * 70)
print("UNKNOWN FEATURES FOR MANUAL REVIEW")
print("=" * 70)

for index, row in inspection_df.iterrows():

    print("\n" + "-" * 70)

    print(f"#{index + 1}")
    print(f"Column: {row['column']}")

    print(
        f"Current category: "
        f"{row['current_category']}"
    )

    print(
        f"Current predictive-safe: "
        f"{row['predictive_safe_current']}"
    )

    print(
        f"Current target candidate: "
        f"{row['target_candidate_current']}"
    )

    print(
        f"Current leakage risk: "
        f"{row['leakage_risk_current']}"
    )

    print(
        f"Automatic suggestion: "
        f"{row['automatic_suggestion']}"
    )

    print(
        f"Feature context: "
        f"{row['feature_context']}"
    )


# ============================================================
# SAVE INSPECTION FILE
# ============================================================

output_path = Path(
    "data/metadata/unknown_feature_inspection.csv"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

inspection_df.to_csv(
    output_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)

print(
    f"\nUnknown features inspected: "
    f"{len(inspection_df):,}"
)

print(
    f"Saved inspection file:\n"
    f"  {output_path}"
)


print("\n" + "-" * 70)
print("AUTOMATIC SUGGESTION COUNTS")
print("-" * 70)

print(
    inspection_df[
        "automatic_suggestion"
    ].value_counts()
)


print("\n" + "-" * 70)
print("FEATURE CONTEXT COUNTS")
print("-" * 70)

print(
    inspection_df[
        "feature_context"
    ].value_counts()
)


print("\n" + "=" * 70)
print("END")
print("=" * 70)