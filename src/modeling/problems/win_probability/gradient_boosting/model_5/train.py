"""
GRADIENT BOOSTING WIN PROBABILITY - MODEL 5

Expanded Feature Space Experiment

Purpose:
    Evaluate whether expanding from the 28-feature compact space used by
    Model 4 to the complete 310-feature predictive-safe space improves
    gradient boosting win-probability performance.

Model 5 methodology:
    - Uses ALL 310 predictive-safe features.
    - Uses the EXACT hyperparameters selected by Model 4.
    - Does NOT perform additional hyperparameter tuning.
    - Trains on 2015-2022.
    - Validates on 2023-2024.
    - Tests on 2025.

Model 4 selected hyperparameters:
    n_estimators     = 200
    learning_rate    = 0.03
    max_depth        = 4
    min_samples_leaf = 10
    subsample        = 0.75

Important:
    Model 5 currently expects all predictive-safe features to be numeric.

    Before training, this script explicitly checks the data types of all
    310 predictive-safe features. If any non-numeric features are found,
    the script stops and reports them rather than silently changing the
    feature-processing methodology.

Output artifacts:
    models/win_probability/gradient_boosting/model_5/
        model.joblib
        feature_list.csv
        training_summary.csv
        validation_predictions.csv
        test_predictions.csv
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
)
from sklearn.pipeline import Pipeline


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42

# -------------------------------------------------------------------------
# Model 4 selected hyperparameters
# -------------------------------------------------------------------------

N_ESTIMATORS = 200
LEARNING_RATE = 0.03
MAX_DEPTH = 4
MIN_SAMPLES_LEAF = 10
SUBSAMPLE = 0.75

# -------------------------------------------------------------------------
# Expected Model 5 feature space
# -------------------------------------------------------------------------

EXPECTED_FEATURE_COUNT = 310

# -------------------------------------------------------------------------
# Dataset columns
# -------------------------------------------------------------------------

TARGET = "win_home"
GAME_ID = "gameId"

# -------------------------------------------------------------------------
# Temporal splits
# -------------------------------------------------------------------------

TRAIN_SEASONS = list(range(2015, 2023))
VALIDATION_SEASONS = [2023, 2024]
TEST_SEASONS = [2025]


# =============================================================================
# PATHS
# =============================================================================

# Current file:
# src/modeling/problems/win_probability/gradient_boosting/model_5/train.py

SCRIPT_DIR = Path(__file__).resolve().parent

# Project root
PROJECT_ROOT = SCRIPT_DIR.parents[5]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_inputs"
    / "win_probability"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "final_feature_classification.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "win_probability"
    / "gradient_boosting"
    / "model_5"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TRAIN_PATH = INPUT_DIR / "train.csv"
VALIDATION_PATH = INPUT_DIR / "validation.csv"
TEST_PATH = INPUT_DIR / "test.csv"


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

warnings.filterwarnings("ignore")


def print_header(title):
    """Print a standardized section header."""

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """
    Load training, validation, test, and feature metadata datasets.
    """

    print_header("LOADING DATA")

    required_paths = [
        TRAIN_PATH,
        VALIDATION_PATH,
        TEST_PATH,
        METADATA_PATH,
    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                f"\nRequired file not found:\n{path}"
            )

    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    test = pd.read_csv(TEST_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    print(f"Training data:   {train.shape}")
    print(f"Validation data: {validation.shape}")
    print(f"Test data:       {test.shape}")
    print(f"Feature metadata:{metadata.shape}")

    return (
        train,
        validation,
        test,
        metadata,
    )


# =============================================================================
# METADATA INSPECTION
# =============================================================================

def identify_metadata_columns(metadata):
    """
    Identify the feature-name and predictive-safe classification columns.

    The metadata file currently contains:
        - column
        - predictive_safe

    Several common naming conventions are supported defensively.
    """

    feature_candidates = [
        "feature",
        "feature_name",
        "column",
        "column_name",
        "Feature",
        "Feature Name",
        "Column",
        "Column Name",
    ]

    safety_candidates = [
        "predictive_safe",
        "predictive-safe",
        "predictiveSafe",
        "is_predictive_safe",
        "Predictive Safe",
        "predictive safety",
        "safety",
    ]

    feature_col = None
    safety_col = None

    for candidate in feature_candidates:

        if candidate in metadata.columns:

            feature_col = candidate
            break

    for candidate in safety_candidates:

        if candidate in metadata.columns:

            safety_col = candidate
            break

    if feature_col is None:

        raise ValueError(
            "\nCould not identify the feature-name column in "
            "final_feature_classification.csv.\n\n"
            f"Available columns:\n{list(metadata.columns)}"
        )

    if safety_col is None:

        raise ValueError(
            "\nCould not identify the predictive-safe classification "
            "column in final_feature_classification.csv.\n\n"
            f"Available columns:\n{list(metadata.columns)}"
        )

    return (
        feature_col,
        safety_col,
    )


# =============================================================================
# PREDICTIVE-SAFE FEATURE EXTRACTION
# =============================================================================

def is_predictive_safe(value):
    """
    Interpret common representations of predictive-safe values.
    """

    if pd.isna(value):

        return False

    if isinstance(value, bool):

        return value

    normalized = str(value).strip().lower()

    safe_values = {
        "true",
        "1",
        "yes",
        "y",
        "safe",
        "predictive-safe",
        "predictive_safe",
        "predictive safe",
    }

    return normalized in safe_values


def get_predictive_safe_features(metadata):
    """
    Extract the complete predictive-safe feature set from metadata.
    """

    print_header("IDENTIFYING PREDICTIVE-SAFE FEATURES")

    (
        feature_col,
        safety_col,
    ) = identify_metadata_columns(metadata)

    print(f"Feature column:      {feature_col}")
    print(f"Safety column:       {safety_col}")

    metadata = metadata.copy()

    metadata["_predictive_safe"] = (
        metadata[safety_col]
        .apply(is_predictive_safe)
    )

    feature_list = (
        metadata.loc[
            metadata["_predictive_safe"],
            feature_col,
        ]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    # Remove accidental duplicates while preserving metadata order.
    feature_list = list(
        dict.fromkeys(feature_list)
    )

    print(
        f"\nPredictive-safe features found: "
        f"{len(feature_list)}"
    )

    if len(feature_list) != EXPECTED_FEATURE_COUNT:

        raise ValueError(
            "\nMODEL 5 FEATURE COUNT CHECK FAILED\n"
            f"Expected exactly {EXPECTED_FEATURE_COUNT} "
            f"predictive-safe features, but found "
            f"{len(feature_list)}.\n\n"
            "This prevents Model 5 from accidentally using "
            "the wrong feature space."
        )

    print(
        f"Feature count check PASSED: "
        f"{EXPECTED_FEATURE_COUNT} features"
    )

    return feature_list


# =============================================================================
# FEATURE SPACE VALIDATION
# =============================================================================

def validate_feature_list(feature_list):
    """
    Validate the Model 5 feature list.
    """

    print_header("VALIDATING MODEL 5 FEATURE SPACE")

    # -------------------------------------------------------------------------
    # Feature count
    # -------------------------------------------------------------------------

    if len(feature_list) != EXPECTED_FEATURE_COUNT:

        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} features, "
            f"found {len(feature_list)}."
        )

    # -------------------------------------------------------------------------
    # Duplicate features
    # -------------------------------------------------------------------------

    if len(feature_list) != len(set(feature_list)):

        duplicates = [
            feature
            for feature in set(feature_list)
            if feature_list.count(feature) > 1
        ]

        raise ValueError(
            "Duplicate features detected:\n"
            f"{duplicates}"
        )

    # -------------------------------------------------------------------------
    # Target / identifier leakage
    # -------------------------------------------------------------------------

    leakage_columns = {
        GAME_ID,
        "season",
        TARGET,
    }

    leakage_features = [
        feature
        for feature in feature_list
        if feature in leakage_columns
    ]

    if leakage_features:

        raise ValueError(
            "Identifier/target columns accidentally included "
            f"in feature list:\n{leakage_features}"
        )

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------

    print(
        f"Feature count: {len(feature_list)} "
        f"(expected {EXPECTED_FEATURE_COUNT})"
    )

    print("Duplicate features: NONE")
    print("Target/identifier leakage: NONE")

    print("\nFeature-space validation: PASSED")


# =============================================================================
# DATASET COLUMN VALIDATION
# =============================================================================

def validate_required_columns(
    train,
    validation,
    test,
    feature_list,
):
    """
    Validate required base columns and all 310 features.
    """

    print_header("VALIDATING DATASET COLUMNS")

    required_base_columns = {
        GAME_ID,
        "season",
        TARGET,
    }

    # -------------------------------------------------------------------------
    # Base columns
    # -------------------------------------------------------------------------

    for name, df in [
        ("Training", train),
        ("Validation", validation),
        ("Test", test),
    ]:

        missing_base = (
            required_base_columns
            - set(df.columns)
        )

        if missing_base:

            raise ValueError(
                f"{name} dataset is missing required "
                f"columns: {sorted(missing_base)}"
            )

    # -------------------------------------------------------------------------
    # Model features
    # -------------------------------------------------------------------------

    for name, df in [
        ("Training", train),
        ("Validation", validation),
        ("Test", test),
    ]:

        missing_features = [
            feature
            for feature in feature_list
            if feature not in df.columns
        ]

        if missing_features:

            raise ValueError(
                f"\n{name} dataset is missing "
                f"{len(missing_features)} Model 5 features.\n\n"
                f"First missing features:\n"
                f"{missing_features[:20]}"
            )

    print("Required base columns: PASSED")
    print("All 310 features present: PASSED")


# =============================================================================
# TARGET VALIDATION
# =============================================================================

def validate_targets(
    train,
    validation,
    test,
):
    """
    Validate target variables.
    """

    print_header("VALIDATING TARGETS")

    for name, df in [
        ("Training", train),
        ("Validation", validation),
        ("Test", test),
    ]:

        # ---------------------------------------------------------------------
        # Missing target
        # ---------------------------------------------------------------------

        if df[TARGET].isna().any():

            raise ValueError(
                f"{name} contains missing target values."
            )

        # ---------------------------------------------------------------------
        # Target values
        # ---------------------------------------------------------------------

        unique_targets = sorted(
            df[TARGET]
            .unique()
            .tolist()
        )

        if unique_targets != [0, 1]:

            raise ValueError(
                f"{name} target contains unexpected "
                f"values: {unique_targets}"
            )

        # ---------------------------------------------------------------------
        # Summary
        # ---------------------------------------------------------------------

        print(
            f"{name}: "
            f"n={len(df):,}, "
            f"home win rate={df[TARGET].mean():.4f}"
        )

    print("\nTarget validation: PASSED")


# =============================================================================
# GAME ID VALIDATION
# =============================================================================

def validate_ids(
    train,
    validation,
    test,
):
    """
    Validate game IDs and cross-split separation.
    """

    print_header("VALIDATING GAME IDS")

    # -------------------------------------------------------------------------
    # Within-split validation
    # -------------------------------------------------------------------------

    for name, df in [
        ("Training", train),
        ("Validation", validation),
        ("Test", test),
    ]:

        if df[GAME_ID].isna().any():

            raise ValueError(
                f"{name} contains missing game IDs."
            )

        if df[GAME_ID].duplicated().any():

            duplicates = (
                df.loc[
                    df[GAME_ID].duplicated(
                        keep=False
                    ),
                    GAME_ID,
                ]
                .tolist()
            )

            raise ValueError(
                f"{name} contains duplicate game IDs.\n"
                f"Examples: {duplicates[:10]}"
            )

        print(
            f"{name}: IDs unique and non-missing"
        )

    # -------------------------------------------------------------------------
    # Cross-split validation
    # -------------------------------------------------------------------------

    train_ids = set(train[GAME_ID])
    validation_ids = set(validation[GAME_ID])
    test_ids = set(test[GAME_ID])

    train_val_overlap = (
        train_ids
        & validation_ids
    )

    train_test_overlap = (
        train_ids
        & test_ids
    )

    validation_test_overlap = (
        validation_ids
        & test_ids
    )

    if train_val_overlap:

        raise ValueError(
            "Train/validation game ID overlap detected: "
            f"{len(train_val_overlap)}"
        )

    if train_test_overlap:

        raise ValueError(
            "Train/test game ID overlap detected: "
            f"{len(train_test_overlap)}"
        )

    if validation_test_overlap:

        raise ValueError(
            "Validation/test game ID overlap detected: "
            f"{len(validation_test_overlap)}"
        )

    print("Cross-split ID overlap: PASSED")


# =============================================================================
# TEMPORAL SPLIT VALIDATION
# =============================================================================

def validate_seasons(
    train,
    validation,
    test,
):
    """
    Validate the exact temporal train/validation/test split.
    """

    print_header("VALIDATING TEMPORAL SPLITS")

    train_seasons = sorted(
        train["season"]
        .unique()
        .tolist()
    )

    validation_seasons = sorted(
        validation["season"]
        .unique()
        .tolist()
    )

    test_seasons = sorted(
        test["season"]
        .unique()
        .tolist()
    )

    print(
        f"Training seasons:   {train_seasons}"
    )

    print(
        f"Validation seasons: {validation_seasons}"
    )

    print(
        f"Test seasons:       {test_seasons}"
    )

    if train_seasons != TRAIN_SEASONS:

        raise ValueError(
            "Training seasons incorrect.\n"
            f"Expected: {TRAIN_SEASONS}\n"
            f"Found:    {train_seasons}"
        )

    if validation_seasons != VALIDATION_SEASONS:

        raise ValueError(
            "Validation seasons incorrect.\n"
            f"Expected: {VALIDATION_SEASONS}\n"
            f"Found:    {validation_seasons}"
        )

    if test_seasons != TEST_SEASONS:

        raise ValueError(
            "Test seasons incorrect.\n"
            f"Expected: {TEST_SEASONS}\n"
            f"Found:    {test_seasons}"
        )

    print("\nTemporal split validation: PASSED")


# =============================================================================
# DATA TYPE DIAGNOSTIC
# =============================================================================

def inspect_feature_dtypes(
    train,
    validation,
    test,
    feature_list,
):
    """
    Inspect all Model 5 feature data types.

    This is intentionally performed BEFORE model construction/training.

    GradientBoostingClassifier with median imputation requires numeric
    feature values. Model 5's 310 predictive-safe feature set may contain
    categorical/string variables, so we explicitly identify them here.
    """

    print_header("CHECKING FEATURE DATA TYPES")

    train_features = train[feature_list]

    # -------------------------------------------------------------------------
    # Numeric vs non-numeric features
    # -------------------------------------------------------------------------

    numeric_features = (
        train_features
        .select_dtypes(
            include=[np.number]
        )
        .columns
        .tolist()
    )

    non_numeric_features = [
        feature
        for feature in feature_list
        if feature not in numeric_features
    ]

    print(
        f"Total Model 5 features: {len(feature_list)}"
    )

    print(
        f"Numeric features:       {len(numeric_features)}"
    )

    print(
        f"Non-numeric features:   {len(non_numeric_features)}"
    )

    # -------------------------------------------------------------------------
    # Numeric dtype summary
    # -------------------------------------------------------------------------

    print("\nNumeric dtype breakdown:")

    numeric_dtype_counts = (
        train_features[numeric_features]
        .dtypes
        .astype(str)
        .value_counts()
    )

    for dtype, count in numeric_dtype_counts.items():

        print(
            f"  {dtype}: {count}"
        )

    # -------------------------------------------------------------------------
    # Non-numeric feature details
    # -------------------------------------------------------------------------

    if non_numeric_features:

        print(
            "\nNon-numeric predictive-safe features:"
        )

        for feature in non_numeric_features:

            train_series = train[feature]
            validation_series = validation[feature]
            test_series = test[feature]

            train_values = (
                train_series
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            validation_values = (
                validation_series
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            test_values = (
                test_series
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            print("\n" + "-" * 80)

            print(
                f"Feature: {feature}"
            )

            print(
                f"  Training dtype:   "
                f"{train_series.dtype}"
            )

            print(
                f"  Validation dtype: "
                f"{validation_series.dtype}"
            )

            print(
                f"  Test dtype:       "
                f"{test_series.dtype}"
            )

            print(
                f"  Training missing: "
                f"{train_series.isna().sum():,}"
            )

            print(
                f"  Validation missing: "
                f"{validation_series.isna().sum():,}"
            )

            print(
                f"  Test missing: "
                f"{test_series.isna().sum():,}"
            )

            print(
                "  Training unique values "
                f"(first 20): {train_values[:20]}"
            )

            print(
                "  Validation unique values "
                f"(first 20): {validation_values[:20]}"
            )

            print(
                "  Test unique values "
                f"(first 20): {test_values[:20]}"
            )

        # ---------------------------------------------------------------------
        # Stop before training
        # ---------------------------------------------------------------------

        print("\n" + "=" * 80)
        print("MODEL 5 TRAINING STOPPED")
        print("=" * 80)

        print(
            """
Non-numeric predictive-safe features were detected.

The current Model 5 pipeline uses:

    SimpleImputer(strategy="median")
        ->
    GradientBoostingClassifier

Median imputation cannot process string/categorical values.

The script intentionally stops here so that we do NOT silently alter
the Model 5 methodology.

Next step:
    Review the non-numeric features above and determine whether they
    should be:

        1. Numerically encoded,
        2. One-hot encoded,
        3. Reclassified as metadata/non-predictive,
        4. Or otherwise transformed.

This decision should be made before training Model 5.
"""
        )

        return False

    # -------------------------------------------------------------------------
    # All features numeric
    # -------------------------------------------------------------------------

    print(
        "\nAll 310 predictive-safe features are numeric."
    )

    print(
        "Numeric feature validation: PASSED"
    )

    return True


# =============================================================================
# MISSINGNESS
# =============================================================================

def report_missingness(
    train,
    validation,
    test,
    feature_list,
):
    """
    Report missingness across Model 5 features.
    """

    print_header("FEATURE MISSINGNESS")

    for name, df in [
        ("Training", train),
        ("Validation", validation),
        ("Test", test),
    ]:

        missing = (
            df[feature_list]
            .isna()
            .sum()
        )

        missing_features = (
            missing > 0
        ).sum()

        total_missing = missing.sum()

        print(
            f"{name}: "
            f"{missing_features}/{len(feature_list)} "
            f"features with missing values | "
            f"{total_missing:,} total missing cells"
        )


# =============================================================================
# MODEL CREATION
# =============================================================================

def create_model():
    """
    Create the Model 5 pipeline.

    Hyperparameters are intentionally identical to the best Model 4
    configuration.
    """

    model = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=N_ESTIMATORS,
                    learning_rate=LEARNING_RATE,
                    max_depth=MAX_DEPTH,
                    min_samples_leaf=MIN_SAMPLES_LEAF,
                    subsample=SUBSAMPLE,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return model


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(
    y_true,
    probabilities,
    predictions,
):
    """
    Calculate classification and probability metrics.
    """

    return {
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            predictions,
        ),

        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),

        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),

        "log_loss": log_loss(
            y_true,
            probabilities,
        ),

        "brier_score": brier_score_loss(
            y_true,
            probabilities,
        ),
    }


def print_metrics(
    name,
    metrics,
):
    """
    Print model metrics.
    """

    print(f"\n{name}")
    print("-" * 40)

    print(
        f"Accuracy:          "
        f"{metrics['accuracy']:.6f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{metrics['balanced_accuracy']:.6f}"
    )

    print(
        f"Precision:         "
        f"{metrics['precision']:.6f}"
    )

    print(
        f"Recall:            "
        f"{metrics['recall']:.6f}"
    )

    print(
        f"ROC AUC:           "
        f"{metrics['roc_auc']:.6f}"
    )

    print(
        f"Log Loss:          "
        f"{metrics['log_loss']:.6f}"
    )

    print(
        f"Brier Score:       "
        f"{metrics['brier_score']:.6f}"
    )


# =============================================================================
# PREDICTION DATAFRAME
# =============================================================================

def create_prediction_dataframe(
    df,
    probabilities,
    predictions,
    split_name,
):
    """
    Create standardized prediction output.
    """

    return pd.DataFrame(
        {
            GAME_ID: df[GAME_ID].values,
            "season": df["season"].values,
            "win_home_actual": df[TARGET].values,
            "win_home_probability": probabilities,
            "win_home_prediction": predictions,
            "split": split_name,
        }
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("\n")

    print("=" * 80)
    print(
        "GRADIENT BOOSTING WIN PROBABILITY - MODEL 5"
    )
    print(
        "EXPANDED FEATURE SPACE EXPERIMENT"
    )
    print("=" * 80)

    print(
        """
Model 5 definition:

    Feature space:
        Complete predictive-safe feature set

    Expected features:
        310

    Hyperparameter tuning:
        NONE

    Hyperparameters:
        n_estimators     = 200
        learning_rate    = 0.03
        max_depth        = 4
        min_samples_leaf = 10
        subsample        = 0.75

    Training:
        2015-2022

    Validation:
        2023-2024

    Test:
        2025
"""
    )

    # =========================================================================
    # LOAD DATA
    # =========================================================================

    (
        train,
        validation,
        test,
        metadata,
    ) = load_data()

    # =========================================================================
    # IDENTIFY FEATURES
    # =========================================================================

    feature_list = (
        get_predictive_safe_features(
            metadata
        )
    )

    # =========================================================================
    # VALIDATE FEATURE SPACE
    # =========================================================================

    validate_feature_list(
        feature_list
    )

    # =========================================================================
    # VALIDATE DATASET COLUMNS
    # =========================================================================

    validate_required_columns(
        train,
        validation,
        test,
        feature_list,
    )

    # =========================================================================
    # VALIDATE TARGETS
    # =========================================================================

    validate_targets(
        train,
        validation,
        test,
    )

    # =========================================================================
    # VALIDATE GAME IDS
    # =========================================================================

    validate_ids(
        train,
        validation,
        test,
    )

    # =========================================================================
    # VALIDATE TEMPORAL SPLITS
    # =========================================================================

    validate_seasons(
        train,
        validation,
        test,
    )

    # =========================================================================
    # PREPARE DATA
    # =========================================================================

    print_header("PREPARING MODEL DATA")

    X_train = train[
        feature_list
    ].copy()

    y_train = train[
        TARGET
    ].copy()

    X_validation = validation[
        feature_list
    ].copy()

    y_validation = validation[
        TARGET
    ].copy()

    X_test = test[
        feature_list
    ].copy()

    y_test = test[
        TARGET
    ].copy()

    print(
        f"X_train:       {X_train.shape}"
    )

    print(
        f"X_validation:  {X_validation.shape}"
    )

    print(
        f"X_test:        {X_test.shape}"
    )

    # =========================================================================
    # DATA TYPE DIAGNOSTIC
    # =========================================================================

    numeric_features_valid = (
        inspect_feature_dtypes(
            train,
            validation,
            test,
            feature_list,
        )
    )

    # -------------------------------------------------------------------------
    # IMPORTANT:
    # Stop if non-numeric features are present.
    # -------------------------------------------------------------------------

    if not numeric_features_valid:

        return

    # =========================================================================
    # MISSINGNESS
    # =========================================================================

    report_missingness(
        train,
        validation,
        test,
        feature_list,
    )

    # =========================================================================
    # CREATE MODEL
    # =========================================================================

    print_header("CREATING MODEL")

    print(
        "GradientBoostingClassifier configuration:"
    )

    print(
        f"  n_estimators:     {N_ESTIMATORS}"
    )

    print(
        f"  learning_rate:    {LEARNING_RATE}"
    )

    print(
        f"  max_depth:        {MAX_DEPTH}"
    )

    print(
        f"  min_samples_leaf: {MIN_SAMPLES_LEAF}"
    )

    print(
        f"  subsample:        {SUBSAMPLE}"
    )

    print(
        f"  random_state:     {RANDOM_STATE}"
    )

    model = create_model()

    # =========================================================================
    # TRAIN
    # =========================================================================

    print_header("TRAINING MODEL 5")

    print(
        f"Training on {len(X_train):,} games "
        f"using {len(feature_list)} features..."
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Model training complete."
    )

    # =========================================================================
    # VALIDATION PREDICTIONS
    # =========================================================================

    print_header("VALIDATION PERFORMANCE")

    validation_probabilities = (
        model.predict_proba(
            X_validation
        )[:, 1]
    )

    validation_predictions = (
        validation_probabilities >= 0.5
    ).astype(int)

    validation_metrics = (
        calculate_metrics(
            y_validation,
            validation_probabilities,
            validation_predictions,
        )
    )

    print_metrics(
        "2023-2024 Validation",
        validation_metrics,
    )

    # =========================================================================
    # TEST PREDICTIONS
    # =========================================================================

    print_header("TEST PERFORMANCE")

    test_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    test_predictions = (
        test_probabilities >= 0.5
    ).astype(int)

    test_metrics = (
        calculate_metrics(
            y_test,
            test_probabilities,
            test_predictions,
        )
    )

    print_metrics(
        "2025 Test",
        test_metrics,
    )

    # =========================================================================
    # PREDICTION DISTRIBUTIONS
    # =========================================================================

    print_header(
        "PREDICTION DISTRIBUTIONS"
    )

    print(
        "Validation probabilities:"
    )

    print(
        pd.Series(
            validation_probabilities
        ).describe()
    )

    print(
        "\nTest probabilities:"
    )

    print(
        pd.Series(
            test_probabilities
        ).describe()
    )

    # =========================================================================
    # CREATE PREDICTION DATAFRAMES
    # =========================================================================

    validation_predictions_df = (
        create_prediction_dataframe(
            validation,
            validation_probabilities,
            validation_predictions,
            "validation",
        )
    )

    test_predictions_df = (
        create_prediction_dataframe(
            test,
            test_probabilities,
            test_predictions,
            "test",
        )
    )

    # =========================================================================
    # SAVE MODEL
    # =========================================================================

    print_header("SAVING ARTIFACTS")

    model_path = (
        OUTPUT_DIR
        / "model.joblib"
    )

    joblib.dump(
        model,
        model_path,
    )

    print(
        f"Saved model: {model_path}"
    )

    # =========================================================================
    # SAVE FEATURE LIST
    # =========================================================================

    feature_list_df = pd.DataFrame(
        {
            "feature": feature_list,
        }
    )

    feature_list_path = (
        OUTPUT_DIR
        / "feature_list.csv"
    )

    feature_list_df.to_csv(
        feature_list_path,
        index=False,
    )

    print(
        f"Saved feature list: "
        f"{feature_list_path}"
    )

    # =========================================================================
    # SAVE TRAINING SUMMARY
    # =========================================================================

    training_summary = pd.DataFrame(
        [
            {
                "model":
                    "gradient_boosting_model_5",

                "feature_count":
                    len(feature_list),

                "training_seasons":
                    ",".join(
                        map(
                            str,
                            TRAIN_SEASONS,
                        )
                    ),

                "validation_seasons":
                    ",".join(
                        map(
                            str,
                            VALIDATION_SEASONS,
                        )
                    ),

                "test_seasons":
                    ",".join(
                        map(
                            str,
                            TEST_SEASONS,
                        )
                    ),

                "n_estimators":
                    N_ESTIMATORS,

                "learning_rate":
                    LEARNING_RATE,

                "max_depth":
                    MAX_DEPTH,

                "min_samples_leaf":
                    MIN_SAMPLES_LEAF,

                "subsample":
                    SUBSAMPLE,

                "random_state":
                    RANDOM_STATE,

                "validation_accuracy":
                    validation_metrics[
                        "accuracy"
                    ],

                "validation_balanced_accuracy":
                    validation_metrics[
                        "balanced_accuracy"
                    ],

                "validation_precision":
                    validation_metrics[
                        "precision"
                    ],

                "validation_recall":
                    validation_metrics[
                        "recall"
                    ],

                "validation_roc_auc":
                    validation_metrics[
                        "roc_auc"
                    ],

                "validation_log_loss":
                    validation_metrics[
                        "log_loss"
                    ],

                "validation_brier_score":
                    validation_metrics[
                        "brier_score"
                    ],

                "test_accuracy":
                    test_metrics[
                        "accuracy"
                    ],

                "test_balanced_accuracy":
                    test_metrics[
                        "balanced_accuracy"
                    ],

                "test_precision":
                    test_metrics[
                        "precision"
                    ],

                "test_recall":
                    test_metrics[
                        "recall"
                    ],

                "test_roc_auc":
                    test_metrics[
                        "roc_auc"
                    ],

                "test_log_loss":
                    test_metrics[
                        "log_loss"
                    ],

                "test_brier_score":
                    test_metrics[
                        "brier_score"
                    ],
            }
        ]
    )

    training_summary_path = (
        OUTPUT_DIR
        / "training_summary.csv"
    )

    training_summary.to_csv(
        training_summary_path,
        index=False,
    )

    print(
        f"Saved training summary: "
        f"{training_summary_path}"
    )

    # =========================================================================
    # SAVE VALIDATION / TEST PREDICTIONS
    # =========================================================================

    validation_predictions_path = (
        OUTPUT_DIR
        / "validation_predictions.csv"
    )

    test_predictions_path = (
        OUTPUT_DIR
        / "test_predictions.csv"
    )

    validation_predictions_df.to_csv(
        validation_predictions_path,
        index=False,
    )

    test_predictions_df.to_csv(
        test_predictions_path,
        index=False,
    )

    print(
        f"Saved validation predictions: "
        f"{validation_predictions_path}"
    )

    print(
        f"Saved test predictions: "
        f"{test_predictions_path}"
    )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print_header(
        "MODEL 5 TRAINING COMPLETE"
    )

    print(
        f"""
Model 5:
    Features:          {len(feature_list)}
    Training seasons:  {TRAIN_SEASONS[0]}-{TRAIN_SEASONS[-1]}
    Validation:        {VALIDATION_SEASONS}
    Test:              {TEST_SEASONS}

Model 4 hyperparameters reused exactly:
    n_estimators:      {N_ESTIMATORS}
    learning_rate:     {LEARNING_RATE}
    max_depth:         {MAX_DEPTH}
    min_samples_leaf:  {MIN_SAMPLES_LEAF}
    subsample:         {SUBSAMPLE}

Validation:
    Accuracy:          {validation_metrics['accuracy']:.6f}
    ROC AUC:           {validation_metrics['roc_auc']:.6f}
    Log Loss:          {validation_metrics['log_loss']:.6f}
    Brier Score:       {validation_metrics['brier_score']:.6f}

Test:
    Accuracy:          {test_metrics['accuracy']:.6f}
    ROC AUC:           {test_metrics['roc_auc']:.6f}
    Log Loss:          {test_metrics['log_loss']:.6f}
    Brier Score:       {test_metrics['brier_score']:.6f}

Next step:
    Run the Model 5 diagnostic/stability audit and compare
    Model 4 (28 features) against Model 5 (310 features).
"""
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()