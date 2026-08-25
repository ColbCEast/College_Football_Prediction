"""
This script will be used to narrow down the final
features dataset to the target variable (win_home)
for logistic regression.

This script will:
    1. Load the final feature classification metadata
    2. Load the final feature files for seasons 2015-2025
    3. Identify the win_home target and all predictive-safe features
    4. Validate the feature schema across the 11 seasons
    5. Create a single game-level modeling dataset
    6. Preserve the season and game identifiers for temporal splitting/evaluation
    7. Save the prepared dataset for later modeling

The following will be saved for later scripts:
    1. Train/Validation/Test splits
    2. Missing value handling
    3. Feature scaling
    4. Model training
    5. Model evaluation
"""

from pathlib import Path
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CLASSIFICATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "final_feature_classification.csv"
)

FEATURES_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_features"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "modeling"
)

OUTPUT_PATH = OUTPUT_DIR / "logistic_regression_data.csv"

START_YEAR = 2015
END_YEAR = 2025

TARGET_COLUMN = "win_home"

# Columns that should be retained for identification and temporal splitting,
# but should NOT be used as model predictors.
REQUIRED_ID_COLUMNS = [
    "season",
    "gameId",
]

# Expected predictive-safe category names from the final classification.
PREDICTIVE_SAFE_CATEGORIES = {
    "pregame",
    "rolling_pregame",
    "static_pregame",
}


# ============================================================================
# HELPERS
# ============================================================================

def load_feature_classification(path: Path) -> pd.DataFrame:
    """Load and validate the final feature classification metadata."""

    print("\n" + "=" * 70)
    print("LOADING FEATURE CLASSIFICATION")
    print("=" * 70)

    if not path.exists():
        raise FileNotFoundError(
            f"Feature classification file not found:\n{path}"
        )

    classification = pd.read_csv(path)

    print(f"Classification shape: {classification.shape}")
    print(f"Classification columns: {list(classification.columns)}")

    required_columns = {
        "column",
        "category",
        "predictive_safe",
    }

    missing_columns = required_columns - set(classification.columns)

    if missing_columns:
        raise ValueError(
            "Feature classification is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if classification["column"].duplicated().any():
        duplicates = classification.loc[
            classification["column"].duplicated(),
            "column",
        ].tolist()

        raise ValueError(
            "Duplicate features found in classification:\n"
            f"{duplicates}"
        )

    return classification


def identify_model_features(
    classification: pd.DataFrame,
) -> list[str]:
    """Identify features allowed for the logistic regression model."""

    print("\n" + "=" * 70)
    print("IDENTIFYING MODEL FEATURES")
    print("=" * 70)

    # Use both the category and predictive_safe flag.
    model_features = classification.loc[
        (
            classification["predictive_safe"].astype(bool)
            & classification["category"].isin(PREDICTIVE_SAFE_CATEGORIES)
        ),
        "column",
    ].tolist()

    if not model_features:
        raise ValueError(
            "No predictive-safe model features were identified."
        )

    print(f"Predictive-safe model features: {len(model_features)}")

    selected_classification = classification[
        classification["column"].isna(model_features)
    ]

    category_counts = (
        selected_classification["category"]
        .value_counts()
        .sort_index()
    )

    print("\nFeatures by category:")
    for category, count in category_counts.items():
        print(f"  {category:<25} {count:>5}")

    return model_features


def load_season_features(
    year: int,
    expected_features: list[str],
) -> pd.DataFrame:
    """Load and validate one season of final features."""

    path = FEATURES_DIR / f"final_features_{year}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Final feature file not found for {year}:\n{path}"
        )

    df = pd.read_csv(path)

    print(
        f"  {year}: loaded {df.shape[0]:,} rows × "
        f"{df.shape[1]:,} columns"
    )

    required_columns = set(expected_features) | set(REQUIRED_ID_COLUMNS) | {
        TARGET_COLUMN
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{year} is missing required columns:\n"
            f"{sorted(missing_columns)}"
        )

    return df


def validate_target(df: pd.DataFrame) -> None:
    """Validate the home-win target."""

    print("\n" + "=" * 70)
    print("VALIDATING TARGET")
    print("=" * 70)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found."
        )

    target = df[TARGET_COLUMN]

    print(f"Target: {TARGET_COLUMN}")
    print(f"Missing values: {target.isna().sum():,}")

    if target.isna().any():
        raise ValueError(
            f"Target '{TARGET_COLUMN}' contains missing values."
        )

    unique_values = sorted(target.unique().tolist())

    print(f"Unique values: {unique_values}")

    if not set(unique_values).issubset({0, 1}):
        raise ValueError(
            f"Target '{TARGET_COLUMN}' must contain only 0/1 values. "
            f"Found: {unique_values}"
        )

    print("\nTarget distribution:")

    counts = target.value_counts().sort_index()
    percentages = target.value_counts(
        normalize=True
    ).sort_index() * 100

    for value in counts.index:
        label = "Away win" if value == 0 else "Home win"

        print(
            f"  {value} ({label:<9}): "
            f"{counts[value]:>6,} "
            f"({percentages[value]:>6.2f}%)"
        )


def validate_identifiers(df: pd.DataFrame) -> None:
    """Validate game identifiers."""

    print("\n" + "=" * 70)
    print("VALIDATING IDENTIFIERS")
    print("=" * 70)

    missing_ids = df["gameId"].isna().sum()
    duplicate_ids = df["gameId"].duplicated().sum()

    print(f"Missing game IDs:    {missing_ids:,}")
    print(f"Duplicate game IDs:  {duplicate_ids:,}")

    if missing_ids > 0:
        raise ValueError("Missing game IDs detected.")

    if duplicate_ids > 0:
        raise ValueError(
            f"Duplicate game IDs detected: {duplicate_ids:,}"
        )

    print("Game identifiers validated.")


def validate_missing_features(
    df: pd.DataFrame,
    model_features: list[str],
) -> None:
    """Report missing values in predictive features."""

    print("\n" + "=" * 70)
    print("CHECKING FEATURE MISSINGNESS")
    print("=" * 70)

    missing_counts = df[model_features].isna().sum()

    missing_features = missing_counts[missing_counts > 0]

    print(
        f"Features with missing values: "
        f"{len(missing_features):,} / {len(model_features):,}"
    )

    if len(missing_features) > 0:
        print("\nTop features by missing count:")

        missing_summary = (
            missing_features
            .sort_values(ascending=False)
            .head(20)
        )

        for feature, count in missing_summary.items():
            percentage = count / len(df) * 100

            print(
                f"  {feature:<50} "
                f"{count:>6,} "
                f"({percentage:>6.2f}%)"
            )

        print(
            "\nMissing values are NOT imputed at this stage."
        )
        print(
            "Imputation will be handled inside the modeling "
            "pipeline after temporal splitting."
        )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print("\n" + "=" * 70)
    print("PREPARING LOGISTIC REGRESSION DATA")
    print("=" * 70)

    print(f"\nProject root:")
    print(f"  {PROJECT_ROOT}")

    print(f"\nFeature classification:")
    print(f"  {CLASSIFICATION_PATH}")

    print(f"\nFeature directory:")
    print(f"  {FEATURES_DIR}")

    print(f"\nYears:")
    print(f"  {START_YEAR}-{END_YEAR}")

    # ------------------------------------------------------------------------
    # 1. Load classification
    # ------------------------------------------------------------------------

    classification = load_feature_classification(
        CLASSIFICATION_PATH
    )

    # ------------------------------------------------------------------------
    # 2. Identify model features
    # ------------------------------------------------------------------------

    model_features = identify_model_features(
        classification
    )

    # ------------------------------------------------------------------------
    # 3. Load all seasons
    # ------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("LOADING FINAL FEATURE DATA")
    print("=" * 70)

    season_data = []

    expected_schema = None

    for year in range(START_YEAR, END_YEAR + 1):

        df = load_season_features(
            year=year,
            expected_features=model_features,
        )

        # Validate that the model feature schema is consistent.
        current_schema = set(df.columns)

        if expected_schema is None:
            expected_schema = current_schema
        else:
            if current_schema != expected_schema:
                missing = expected_schema - current_schema
                extra = current_schema - expected_schema

                raise ValueError(
                    f"Schema mismatch detected in {year}.\n"
                    f"Missing columns: {sorted(missing)}\n"
                    f"Extra columns: {sorted(extra)}"
                )

        # Explicitly add/overwrite season based on the file being loaded.
        # This prevents an incorrect season value inside a file from silently
        # entering the modeling dataset.
        df["season"] = year

        season_data.append(df)

    # ------------------------------------------------------------------------
    # 4. Combine seasons
    # ------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("COMBINING SEASONS")
    print("=" * 70)

    data = pd.concat(
        season_data,
        axis=0,
        ignore_index=True,
    )

    print(
        f"Combined dataset: "
        f"{data.shape[0]:,} rows × {data.shape[1]:,} columns"
    )

    # ------------------------------------------------------------------------
    # 5. Validate target
    # ------------------------------------------------------------------------

    validate_target(data)

    # ------------------------------------------------------------------------
    # 6. Validate identifiers
    # ------------------------------------------------------------------------

    validate_identifiers(data)

    # ------------------------------------------------------------------------
    # 7. Validate feature missingness
    # ------------------------------------------------------------------------

    validate_missing_features(
        data,
        model_features,
    )

    # ------------------------------------------------------------------------
    # 8. Create final modeling dataset
    # ------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("CREATING MODELING DATASET")
    print("=" * 70)

    # Preserve identifiers and target, while using ONLY predictive-safe
    # features as predictors.
    output_columns = (
        REQUIRED_ID_COLUMNS
        + model_features
        + [TARGET_COLUMN]
    )

    # Remove accidental duplicates while preserving order.
    output_columns = list(dict.fromkeys(output_columns))

    modeling_data = data[output_columns].copy()

    print(
        f"Modeling dataset: "
        f"{modeling_data.shape[0]:,} rows × "
        f"{modeling_data.shape[1]:,} columns"
    )

    print(f"  Identifier columns: {len(REQUIRED_ID_COLUMNS)}")
    print(f"  Model features:     {len(model_features)}")
    print(f"  Target columns:     1")

    # ------------------------------------------------------------------------
    # 9. Final validation
    # ------------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)

    expected_columns = set(output_columns)

    if set(modeling_data.columns) != expected_columns:
        raise ValueError(
            "Final modeling dataset columns do not match "
            "the expected schema."
        )

    if modeling_data["gameId"].duplicated().any():
        raise ValueError(
            "Duplicate game IDs found in final modeling dataset."
        )

    if modeling_data[TARGET_COLUMN].isna().any():
        raise ValueError(
            "Missing target values found in final modeling dataset."
        )

    # Verify that no feature classified as unsafe has accidentally entered
    # the predictor set.
    classification_lookup = classification.set_index("column")

    for column in model_features:

        row = classification_lookup.loc[column]

        if not bool(row["predictive_safe"]):
            raise ValueError(
                f"Unsafe feature entered modeling dataset: {column}"
            )

        if row["category"] not in PREDICTIVE_SAFE_CATEGORIES:
            raise ValueError(
                f"Unexpected feature category for {column}: "
                f"{row['category']}"
            )

    print("All final validation checks passed.")

    # ------------------------------------------------------------------------
    # 10. Save
    # ------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    modeling_data.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 70)
    print("DATA PREPARATION COMPLETE")
    print("=" * 70)

    print(f"\nSaved modeling dataset:")
    print(f"  {OUTPUT_PATH}")

    print(f"\nFinal shape:")
    print(
        f"  {modeling_data.shape[0]:,} rows × "
        f"{modeling_data.shape[1]:,} columns"
    )

    print("\nDataset structure:")
    print(f"  Identifier columns: {len(REQUIRED_ID_COLUMNS)}")
    print(f"  Predictor columns:  {len(model_features)}")
    print(f"  Target column:      {TARGET_COLUMN}")

    print("\nSeason distribution:")

    season_counts = (
        modeling_data["season"]
        .value_counts()
        .sort_index()
    )

    for season, count in season_counts.items():
        print(f"  {season}: {count:,}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()