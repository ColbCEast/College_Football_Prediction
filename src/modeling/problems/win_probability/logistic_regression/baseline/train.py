"""
Train Baseline Logistic Regression
===================================

Trains a baseline logistic regression model using the pre-created
temporal train/validation/test splits.

Temporal split:
    Training:   2015-2022
    Validation: 2023-2024
    Test:       2025

Target:
    win_home

Preprocessing:
    Numeric features:
        - Median imputation
        - StandardScaler

    Categorical features:
        - Most-frequent imputation
        - OneHotEncoder

Important:
    - All preprocessing is learned from the training set only.
    - No shuffling is performed.
    - The test set is not used for model selection.
    - This is intended to establish a simple baseline before
      feature selection, hyperparameter tuning, or alternative
      modeling approaches.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================================
# CONFIGURATION
# ============================================================================

TRAIN_YEARS = list(range(2015, 2023))
VALIDATION_YEARS = [2023, 2024]
TEST_YEARS = [2025]

TARGET_COLUMN = "win_home"

IDENTIFIER_COLUMNS = [
    "gameId",
    "season",
]

DATE_COLUMNS = [
    "startDate",
]

BASE_DIR = Path(__file__).resolve().parents[6]

MODEL_DATA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
    / "model_inputs"
    / "win_probabilities"
    / "logistic_regression"
)

TRAIN_PATH = MODEL_DATA_DIR / "logistic_regression_train.csv"
VALIDATION_PATH = MODEL_DATA_DIR / "logistic_regression_validation.csv"
TEST_PATH = MODEL_DATA_DIR / "logistic_regression_test.csv"

OUTPUT_DIR = (
    BASE_DIR
    / "models"
    / "win_probability"
    / "logistic_regression"
    / "baseline"
)

MODEL_OUTPUT_PATH = OUTPUT_DIR / "baseline.joblib"
PREDICTIONS_OUTPUT_PATH = (
    OUTPUT_DIR / "predictions.csv"
)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_section(title):
    """Print a standardized section header."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def load_split(path, name):
    """Load and validate a modeling split."""
    if not path.exists():
        raise FileNotFoundError(
            f"{name} split does not exist:\n  {path}"
        )

    df = pd.read_csv(path)

    print(
        f"{name:<12}: "
        f"{len(df):,} rows × {len(df.columns):,} columns"
    )

    return df


# ============================================================================
# VALIDATION
# ============================================================================

def validate_splits(train, validation, test):
    """Validate the temporal modeling splits."""

    print_section("VALIDATING MODELING SPLITS")

    required_columns = {
        TARGET_COLUMN,
    }

    for name, df in [
        ("Training", train),
        ("Validation", validation),
        ("Test", test),
    ]:
        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"{name} split is missing required columns: "
                f"{sorted(missing)}"
            )

        if df[TARGET_COLUMN].isna().any():
            raise ValueError(
                f"{name} split contains missing target values."
            )

        unique_targets = sorted(df[TARGET_COLUMN].unique())

        if unique_targets != [0, 1]:
            raise ValueError(
                f"{name} split has unexpected target values: "
                f"{unique_targets}"
            )

    # ------------------------------------------------------------------
    # Validate game IDs
    # ------------------------------------------------------------------

    available_id = None

    for candidate in IDENTIFIER_COLUMNS:
        if candidate in train.columns:
            available_id = candidate
            break

    if available_id is None:
        raise ValueError(
            "Could not find a game identifier column. "
            f"Expected one of: {IDENTIFIER_COLUMNS}"
        )

    print(f"Game identifier: {available_id}")

    for name, df in [
        ("Training", train),
        ("Validation", validation),
        ("Test", test),
    ]:
        if df[available_id].isna().any():
            raise ValueError(
                f"{name} split contains missing game IDs."
            )

        if df[available_id].duplicated().any():
            raise ValueError(
                f"{name} split contains duplicate game IDs."
            )

    train_ids = set(train[available_id])
    validation_ids = set(validation[available_id])
    test_ids = set(test[available_id])

    if train_ids & validation_ids:
        raise ValueError(
            "Game IDs overlap between training and validation."
        )

    if train_ids & test_ids:
        raise ValueError(
            "Game IDs overlap between training and test."
        )

    if validation_ids & test_ids:
        raise ValueError(
            "Game IDs overlap between validation and test."
        )

    print("Game ID validation passed.")

    # ------------------------------------------------------------------
    # Validate seasons
    # ------------------------------------------------------------------

    if "season" in train.columns:

        train_seasons = sorted(train["season"].unique())
        validation_seasons = sorted(validation["season"].unique())
        test_seasons = sorted(test["season"].unique())

        print()
        print("Season validation:")
        print(f"  Training:   {train_seasons}")
        print(f"  Validation: {validation_seasons}")
        print(f"  Test:       {test_seasons}")

        if train_seasons != TRAIN_YEARS:
            raise ValueError(
                f"Unexpected training seasons: {train_seasons}"
            )

        if validation_seasons != VALIDATION_YEARS:
            raise ValueError(
                f"Unexpected validation seasons: {validation_seasons}"
            )

        if test_seasons != TEST_YEARS:
            raise ValueError(
                f"Unexpected test seasons: {test_seasons}"
            )

    print()
    print("Split validation passed.")


# ============================================================================
# PREPARE FEATURES
# ============================================================================

def identify_predictors(train):
    """
    Identify predictor columns.

    Excludes:
        - Target
        - Game identifiers
        - Raw date columns

    Date columns are deliberately excluded from the baseline model.
    """

    excluded_columns = (
        set(IDENTIFIER_COLUMNS)
        | {TARGET_COLUMN}
        | set(DATE_COLUMNS)
    )

    predictor_columns = [
        column
        for column in train.columns
        if column not in excluded_columns
    ]

    if not predictor_columns:
        raise ValueError("No predictor columns identified.")

    return predictor_columns


def identify_feature_types(train, predictor_columns):
    """
    Separate predictors into numeric and categorical features.

    Feature type is determined from the training data only.
    """

    numeric_features = []
    categorical_features = []

    for column in predictor_columns:

        if pd.api.types.is_numeric_dtype(train[column]):
            numeric_features.append(column)

        else:
            categorical_features.append(column)

    return numeric_features, categorical_features


# ============================================================================
# CREATE MODEL PIPELINE
# ============================================================================

def create_model_pipeline(
    numeric_features,
    categorical_features,
):
    """Create the baseline preprocessing + logistic regression pipeline."""

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    transformers = []

    if numeric_features:
        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    logistic_regression = LogisticRegression(
        max_iter=2000,
        random_state=42,
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                logistic_regression,
            ),
        ]
    )

    return model


# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(y_true, y_probability, threshold=0.50):
    """Calculate binary classification metrics."""

    y_prediction = (
        y_probability >= threshold
    ).astype(int)

    metrics = {
        "accuracy": accuracy_score(
            y_true,
            y_prediction,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_prediction,
        ),
        "precision": precision_score(
            y_true,
            y_prediction,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_prediction,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            y_probability,
        ),
        "log_loss": log_loss(
            y_true,
            y_probability,
        ),
        "brier_score": brier_score_loss(
            y_true,
            y_probability,
        ),
    }

    return metrics


def print_metrics(
    name,
    y_true,
    y_probability,
):
    """Print classification metrics."""

    metrics = calculate_metrics(
        y_true,
        y_probability,
    )

    y_prediction = (
        y_probability >= 0.50
    ).astype(int)

    matrix = confusion_matrix(
        y_true,
        y_prediction,
    )

    print_section(f"{name.upper()} PERFORMANCE")

    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(
        f"Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}"
    )
    print(f"Precision:          {metrics['precision']:.4f}")
    print(f"Recall:             {metrics['recall']:.4f}")
    print(f"ROC AUC:            {metrics['roc_auc']:.4f}")
    print(f"Log Loss:           {metrics['log_loss']:.4f}")
    print(f"Brier Score:        {metrics['brier_score']:.4f}")

    print()
    print("Confusion Matrix:")
    print(
        "                 Predicted"
    )
    print(
        "                 Away   Home"
    )
    print(
        f"Actual Away      {matrix[0, 0]:5d}  {matrix[0, 1]:5d}"
    )
    print(
        f"Actual Home      {matrix[1, 0]:5d}  {matrix[1, 1]:5d}"
    )

    return metrics


# ============================================================================
# PREDICTIONS
# ============================================================================

def create_prediction_dataframe(
    df,
    y_probability,
    split_name,
):
    """Create a dataframe containing predictions and probabilities."""

    result = pd.DataFrame(index=df.index)

    # Preserve identifiers
    for column in IDENTIFIER_COLUMNS:
        if column in df.columns:
            result[column] = df[column].values

    # Preserve season
    if "season" in df.columns:
        result["season"] = df["season"].values

    # Actual outcome
    result["win_home_actual"] = df[TARGET_COLUMN].values

    # Probability home team wins
    result["win_home_probability"] = y_probability

    # Predicted outcome
    result["win_home_prediction"] = (
        y_probability >= 0.50
    ).astype(int)

    result["split"] = split_name

    return result


# ============================================================================
# SAVE MODEL
# ============================================================================

def save_model(model):
    """Save fitted model using joblib."""

    import joblib

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_OUTPUT_PATH,
    )

    print()
    print("Saved fitted model:")
    print(f"  {MODEL_OUTPUT_PATH}")


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 70)
    print("TRAINING BASELINE LOGISTIC REGRESSION")
    print("=" * 70)

    # ==================================================================
    # LOAD DATA
    # ==================================================================

    print_section("LOADING TEMPORAL SPLITS")

    train = load_split(
        TRAIN_PATH,
        "Training",
    )

    validation = load_split(
        VALIDATION_PATH,
        "Validation",
    )

    test = load_split(
        TEST_PATH,
        "Test",
    )

    # ==================================================================
    # VALIDATE DATA
    # ==================================================================

    validate_splits(
        train,
        validation,
        test,
    )

    # ==================================================================
    # PREPARE FEATURES
    # ==================================================================

    print_section("PREPARING FEATURES AND TARGET")

    predictor_columns = identify_predictors(
        train
    )

    print(
        f"Predictor count: {len(predictor_columns):,}"
    )

    X_train = train[predictor_columns]
    X_validation = validation[predictor_columns]
    X_test = test[predictor_columns]

    y_train = train[TARGET_COLUMN]
    y_validation = validation[TARGET_COLUMN]
    y_test = test[TARGET_COLUMN]

    print(
        f"Training rows:   {len(X_train):,}"
    )
    print(
        f"Validation rows: {len(X_validation):,}"
    )
    print(
        f"Test rows:       {len(X_test):,}"
    )

    # ==================================================================
    # IDENTIFY FEATURE TYPES
    # ==================================================================

    print_section("IDENTIFYING FEATURE TYPES")

    numeric_features, categorical_features = (
        identify_feature_types(
            X_train,
            predictor_columns,
        )
    )

    print(
        f"Numeric features:     {len(numeric_features):,}"
    )

    print(
        f"Categorical features: {len(categorical_features):,}"
    )

    if categorical_features:
        print()
        print("Categorical features:")

        for column in categorical_features:
            unique_values = (
                X_train[column]
                .dropna()
                .unique()
            )

            print(
                f"  {column:<35} "
                f"{len(unique_values):>3} unique values"
            )

            if len(unique_values) <= 10:
                print(
                    f"      Values: "
                    f"{sorted(map(str, unique_values))}"
                )

    if DATE_COLUMNS:
        date_columns_present = [
            column
            for column in DATE_COLUMNS
            if column in train.columns
        ]

        if date_columns_present:
            print()
            print(
                "Excluded date columns:"
            )

            for column in date_columns_present:
                print(f"  {column}")

    # ==================================================================
    # CHECK FEATURE TYPES
    # ==================================================================

    print_section("VALIDATING FEATURE TYPES")

    if set(numeric_features) & set(categorical_features):
        raise ValueError(
            "A feature was classified as both numeric "
            "and categorical."
        )

    if (
        len(numeric_features)
        + len(categorical_features)
        != len(predictor_columns)
    ):
        raise ValueError(
            "Feature type counts do not match predictor count."
        )

    print(
        "All predictors have been assigned a valid "
        "numeric or categorical treatment."
    )

    # ==================================================================
    # CREATE PIPELINE
    # ==================================================================

    print_section("CREATING MODEL PIPELINE")

    model = create_model_pipeline(
        numeric_features,
        categorical_features,
    )

    print("Pipeline:")

    print(
        "  Numeric:"
    )
    print(
        "    1. Median imputation"
    )
    print(
        "    2. StandardScaler"
    )

    print(
        "  Categorical:"
    )
    print(
        "    1. Most-frequent imputation"
    )
    print(
        "    2. OneHotEncoder"
    )

    print(
        "  Final estimator:"
    )
    print(
        "    LogisticRegression"
    )

    print()
    print(
        "All preprocessing will be learned from "
        "training data only."
    )

    # ==================================================================
    # FIT MODEL
    # ==================================================================

    print_section("FITTING LOGISTIC REGRESSION")

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Logistic regression fitted successfully."
    )

    # ==================================================================
    # GENERATE PREDICTIONS
    # ==================================================================

    print_section("GENERATING PREDICTIONS")

    train_probability = model.predict_proba(
        X_train
    )[:, 1]

    validation_probability = model.predict_proba(
        X_validation
    )[:, 1]

    test_probability = model.predict_proba(
        X_test
    )[:, 1]

    print(
        "Predictions generated for:"
    )
    print(
        f"  Training:   {len(train_probability):,}"
    )
    print(
        f"  Validation: {len(validation_probability):,}"
    )
    print(
        f"  Test:       {len(test_probability):,}"
    )

    # ==================================================================
    # PERFORMANCE
    # ==================================================================

    train_metrics = print_metrics(
        "Training",
        y_train,
        train_probability,
    )

    validation_metrics = print_metrics(
        "Validation",
        y_validation,
        validation_probability,
    )

    test_metrics = print_metrics(
        "Test",
        y_test,
        test_probability,
    )

    # ==================================================================
    # PROBABILITY SUMMARY
    # ==================================================================

    print("\n" + "=" * 70)
    print("PREDICTED PROBABILITY SUMMARY")
    print("=" * 70)

    probability_summary = pd.DataFrame(
        {
            "Split": ["Training", "Validation", "Test"],
            "Rows": [
                len(train_probability),
                len(validation_probability),
                len(test_probability),
            ],
            "Mean Probability": [
                train_probability.mean(),
                validation_probability.mean(),
                test_probability.mean(),
            ],
            "Median Probability": [
                np.median(train_probability),
                np.median(validation_probability),
                np.median(test_probability),
            ],
            "Minimum Probability": [
                train_probability.min(),
                validation_probability.min(),
                test_probability.min(),
            ],
            "Maximum Probability": [
                train_probability.max(),
                validation_probability.max(),
                test_probability.max(),
            ],
        }
    )

    print(
        probability_summary.to_string(
            index=False,
            formatters={
                "Mean Probability": "{:.4f}".format,
                "Median Probability": "{:.4f}".format,
                "Minimum Probability": "{:.4f}".format,
                "Maximum Probability": "{:.4f}".format,
            },
        )
    )

    # ==================================================================
    # PREDICTION SUMMARY
    # ==================================================================

    print_section("PREDICTION SUMMARY")

    for name, y_true, probabilities in [
        (
            "Training",
            y_train,
            train_probability,
        ),
        (
            "Validation",
            y_validation,
            validation_probability,
        ),
        (
            "Test",
            y_test,
            test_probability,
        ),
    ]:

        predictions = (
            probabilities >= 0.50
        ).astype(int)

        home_prediction_rate = predictions.mean()
        actual_home_rate = y_true.mean()

        print(f"{name}:")
        print(
            f"  Actual home win rate:     "
            f"{actual_home_rate:.4f}"
        )
        print(
            f"  Predicted home win rate:  "
            f"{home_prediction_rate:.4f}"
        )
        print()

    # ==================================================================
    # CREATE PREDICTION OUTPUT
    # ==================================================================

    print_section("CREATING PREDICTION OUTPUT")

    train_predictions = create_prediction_dataframe(
        train,
        train_probability,
        "train",
    )

    validation_predictions = (
        create_prediction_dataframe(
            validation,
            validation_probability,
            "validation",
        )
    )

    test_predictions = create_prediction_dataframe(
        test,
        test_probability,
        "test",
    )

    predictions = pd.concat(
        [
            train_predictions,
            validation_predictions,
            test_predictions,
        ],
        ignore_index=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        PREDICTIONS_OUTPUT_PATH,
        index=False,
    )

    print(
        f"Saved predictions:"
    )
    print(
        f"  {PREDICTIONS_OUTPUT_PATH}"
    )

    print(
        f"Prediction rows: {len(predictions):,}"
    )

    # ==================================================================
    # SAVE MODEL
    # ==================================================================

    save_model(model)

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================

    print_section("BASELINE LOGISTIC REGRESSION COMPLETE")

    print("Model:")
    print("  Logistic Regression")
    print()
    print("Temporal split:")
    print("  Training:   2015-2022")
    print("  Validation: 2023-2024")
    print("  Test:       2025")
    print()
    print("Preprocessing:")
    print("  Numeric:")
    print("    Median imputation")
    print("    StandardScaler")
    print("  Categorical:")
    print("    Most-frequent imputation")
    print("    OneHotEncoder")
    print()
    print("Validation performance:")
    print(
        f"  Accuracy:    "
        f"{validation_metrics['accuracy']:.4f}"
    )
    print(
        f"  ROC AUC:     "
        f"{validation_metrics['roc_auc']:.4f}"
    )
    print(
        f"  Log Loss:    "
        f"{validation_metrics['log_loss']:.4f}"
    )
    print(
        f"  Brier Score: "
        f"{validation_metrics['brier_score']:.4f}"
    )
    print()
    print("Test performance:")
    print(
        f"  Accuracy:    "
        f"{test_metrics['accuracy']:.4f}"
    )
    print(
        f"  ROC AUC:     "
        f"{test_metrics['roc_auc']:.4f}"
    )
    print(
        f"  Log Loss:    "
        f"{test_metrics['log_loss']:.4f}"
    )
    print(
        f"  Brier Score: "
        f"{test_metrics['brier_score']:.4f}"
    )
    print()
    print(
        "This is the baseline model. "
        "No feature selection or hyperparameter tuning "
        "has been performed."
    )


if __name__ == "__main__":
    main()