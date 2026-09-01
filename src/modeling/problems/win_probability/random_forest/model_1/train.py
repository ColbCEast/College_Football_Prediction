"""
Random Forest Model 1 — Win Probability

Trains and evaluates a baseline Random Forest classifier for
pre-game college football home-win probability.

Data:
    Training:   2015–2022
    Validation: 2023–2024
    Test:       2025

Input:
    data/processed/model_inputs/win_probability/

Outputs:
    models/win_probability/random_forest/model_1/

Primary evaluation metric:
    Log Loss

Secondary metrics:
    Brier Score
    ROC AUC
    Accuracy

Preprocessing:
    Median imputation fitted on training data only.

Notes:
    - No feature scaling is used because Random Forest does not require it.
    - The imputer is included in the saved sklearn Pipeline so that the exact
      same preprocessing is automatically applied when the model is loaded.
"""


# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


# =============================================================================
# PATHS
# =============================================================================

# Project root:
# College_Football_Prediction/
#
# train.py is located at:
# src/modeling/problems/win_probability/random_forest/model_1/train.py
#
# parents[6] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[6]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_inputs"
    / "win_probability"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "win_probability"
    / "random_forest"
    / "model_1"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

TARGET = "win_home"

# Columns excluded from the Random Forest predictor set.
#
# season:
#     Temporal identifier; excluded to prevent the model from learning
#     season-specific patterns that may not generalize.
#
# gameId:
#     Unique game identifier; not predictive.
#
# startDate:
#     Date/metadata field; not used as a predictor.
#
# seasonType:
#     Categorical metadata; excluded for consistency with the Logistic V4
#     predictor set.
#
# win_home:
#     Target variable.
EXCLUDED_COLUMNS = [
    "season",
    "gameId",
    "startDate",
    "seasonType",
    TARGET,
]

RANDOM_STATE = 42

RF_PARAMS = {
    "n_estimators": 500,
    "max_features": "sqrt",
    "min_samples_leaf": 5,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    """
    Load the pre-defined temporal train/validation/test datasets.

    Returns:
        train_df
        validation_df
        test_df
    """

    train_path = INPUT_DIR / "train.csv"
    validation_path = INPUT_DIR / "validation.csv"
    test_path = INPUT_DIR / "test.csv"

    print("=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    print(f"Training:   {train_path}")
    print(f"Validation: {validation_path}")
    print(f"Test:       {test_path}")

    train_df = pd.read_csv(train_path)
    validation_df = pd.read_csv(validation_path)
    test_df = pd.read_csv(test_path)

    print()
    print(f"Training shape:   {train_df.shape}")
    print(f"Validation shape: {validation_df.shape}")
    print(f"Test shape:       {test_df.shape}")

    return train_df, validation_df, test_df


# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_datasets(train_df, validation_df, test_df):
    """
    Validate that train, validation, and test datasets are structurally
    compatible and contain valid binary target values.
    """

    print()
    print("=" * 80)
    print("VALIDATING DATASETS")
    print("=" * 80)

    datasets = {
        "Training": train_df,
        "Validation": validation_df,
        "Test": test_df,
    }

    # -------------------------------------------------------------------------
    # Target existence
    # -------------------------------------------------------------------------

    for name, df in datasets.items():
        if TARGET not in df.columns:
            raise ValueError(
                f"{TARGET!r} is missing from the {name.lower()} dataset."
            )

    # -------------------------------------------------------------------------
    # Column consistency
    # -------------------------------------------------------------------------

    train_columns = set(train_df.columns)
    validation_columns = set(validation_df.columns)
    test_columns = set(test_df.columns)

    if train_columns != validation_columns:
        raise ValueError(
            "Training and validation datasets do not contain identical columns."
        )

    if train_columns != test_columns:
        raise ValueError(
            "Training and test datasets do not contain identical columns."
        )

    # Check column order as well.
    if list(train_df.columns) != list(validation_df.columns):
        raise ValueError(
            "Training and validation columns are not in the same order."
        )

    if list(train_df.columns) != list(test_df.columns):
        raise ValueError(
            "Training and test columns are not in the same order."
        )

    # -------------------------------------------------------------------------
    # Excluded-column validation
    # -------------------------------------------------------------------------

    missing_excluded = [
        column
        for column in EXCLUDED_COLUMNS
        if column not in train_df.columns
    ]

    if missing_excluded:
        raise ValueError(
            "The following expected columns are missing: "
            f"{missing_excluded}"
        )

    print("Dataset column validation: PASSED")

    # -------------------------------------------------------------------------
    # Target validation
    # -------------------------------------------------------------------------

    for name, df in datasets.items():

        if df[TARGET].isna().any():
            raise ValueError(
                f"{name} dataset contains missing target values."
            )

        unique_targets = set(df[TARGET].unique())

        if not unique_targets.issubset({0, 1}):
            raise ValueError(
                f"{name} target contains unexpected values: "
                f"{unique_targets}"
            )

    print("Target validation: PASSED")


# =============================================================================
# FEATURE PREPARATION
# =============================================================================

def prepare_features(train_df, validation_df, test_df):
    """
    Separate predictors from the target.

    Missing predictor values are intentionally retained at this stage.
    They will be handled by the median imputer inside the model Pipeline.

    Returns:
        X_train
        y_train
        X_validation
        y_validation
        X_test
        y_test
        feature_columns
    """

    feature_columns = [
        column
        for column in train_df.columns
        if column not in EXCLUDED_COLUMNS
    ]

    if not feature_columns:
        raise ValueError("No predictor columns were identified.")

    X_train = train_df[feature_columns].copy()
    y_train = train_df[TARGET].copy()

    X_validation = validation_df[feature_columns].copy()
    y_validation = validation_df[TARGET].copy()

    X_test = test_df[feature_columns].copy()
    y_test = test_df[TARGET].copy()

    print()
    print("=" * 80)
    print("FEATURE PREPARATION")
    print("=" * 80)

    print(f"Predictors used: {len(feature_columns)}")
    print(f"Excluded columns: {len(EXCLUDED_COLUMNS)}")

    print()
    print("Excluded:")
    for column in EXCLUDED_COLUMNS:
        print(f"  - {column}")

    print()
    print("Feature matrix shapes:")
    print(f"  Training:   {X_train.shape}")
    print(f"  Validation: {X_validation.shape}")
    print(f"  Test:       {X_test.shape}")

    # -------------------------------------------------------------------------
    # Missingness summary
    # -------------------------------------------------------------------------

    train_missing = int(X_train.isna().sum().sum())
    validation_missing = int(X_validation.isna().sum().sum())
    test_missing = int(X_test.isna().sum().sum())

    train_total = X_train.shape[0] * X_train.shape[1]
    validation_total = X_validation.shape[0] * X_validation.shape[1]
    test_total = X_test.shape[0] * X_test.shape[1]

    print()
    print("Missing predictor values:")
    print(
        f"  Training:   {train_missing:,} "
        f"({train_missing / train_total:.2%})"
    )
    print(
        f"  Validation: {validation_missing:,} "
        f"({validation_missing / validation_total:.2%})"
    )
    print(
        f"  Test:       {test_missing:,} "
        f"({test_missing / test_total:.2%})"
    )

    print()
    print(
        "Missing values will be handled using median imputation "
        "fitted on training data only."
    )

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        feature_columns,
    )


# =============================================================================
# TARGET DISTRIBUTION
# =============================================================================

def print_target_distribution(name, y):
    """Print target counts and proportions."""

    counts = y.value_counts().sort_index()
    proportions = y.value_counts(normalize=True).sort_index()

    print()
    print(f"{name} target distribution:")
    print(
        f"  Away win (0): "
        f"{counts.get(0, 0):,} "
        f"({proportions.get(0, 0):.2%})"
    )
    print(
        f"  Home win (1): "
        f"{counts.get(1, 0):,} "
        f"({proportions.get(1, 0):.2%})"
    )


# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_predictions(name, y_true, probabilities):
    """
    Calculate probability and classification metrics.

    Returns:
        Dictionary containing evaluation metrics.
    """

    predictions = (probabilities >= 0.50).astype(int)

    metrics = {
        "dataset": name,
        "n_games": len(y_true),
        "log_loss": log_loss(y_true, probabilities),
        "brier_score": brier_score_loss(y_true, probabilities),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "accuracy": accuracy_score(y_true, predictions),
    }

    print()
    print("=" * 80)
    print(f"{name.upper()} PERFORMANCE")
    print("=" * 80)

    print(f"Games:       {metrics['n_games']:,}")
    print(f"Log Loss:    {metrics['log_loss']:.6f}")
    print(f"Brier Score: {metrics['brier_score']:.6f}")
    print(f"ROC AUC:     {metrics['roc_auc']:.6f}")
    print(f"Accuracy:    {metrics['accuracy']:.4%}")

    return metrics


# =============================================================================
# SAVE PREDICTIONS
# =============================================================================

def save_predictions(
    dataset_name,
    original_df,
    probabilities,
):
    """
    Save predicted probabilities alongside game-identifying information.
    """

    predictions_df = pd.DataFrame({
        "gameId": original_df["gameId"].values,
        "season": original_df["season"].values,
        "startDate": original_df["startDate"].values,
        "seasonType": original_df["seasonType"].values,
        "actual_win_home": original_df[TARGET].values,
        "predicted_home_win_probability": probabilities,
    })

    output_path = (
        OUTPUT_DIR
        / f"{dataset_name.lower()}_predictions.csv"
    )

    predictions_df.to_csv(output_path, index=False)

    print(
        f"Saved {dataset_name.lower()} predictions: "
        f"{output_path}"
    )

    return predictions_df


# =============================================================================
# MAIN
# =============================================================================

def main():

    print()
    print("=" * 80)
    print("RANDOM FOREST MODEL 1 — WIN PROBABILITY")
    print("=" * 80)

    print()
    print("Project root:")
    print(f"  {PROJECT_ROOT}")

    print()
    print("Input directory:")
    print(f"  {INPUT_DIR}")

    print()
    print("Output directory:")
    print(f"  {OUTPUT_DIR}")

    # =========================================================================
    # 1. Load data
    # =========================================================================

    train_df, validation_df, test_df = load_data()

    # =========================================================================
    # 2. Validate datasets
    # =========================================================================

    validate_datasets(
        train_df,
        validation_df,
        test_df,
    )

    # =========================================================================
    # 3. Prepare features
    # =========================================================================

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        feature_columns,
    ) = prepare_features(
        train_df,
        validation_df,
        test_df,
    )

    # =========================================================================
    # 4. Print target distributions
    # =========================================================================

    print()
    print("=" * 80)
    print("TARGET DISTRIBUTIONS")
    print("=" * 80)

    print_target_distribution("Training", y_train)
    print_target_distribution("Validation", y_validation)
    print_target_distribution("Test", y_test)

    # =========================================================================
    # 5. Build Random Forest Pipeline
    # =========================================================================

    print()
    print("=" * 80)
    print("RANDOM FOREST CONFIGURATION")
    print("=" * 80)

    print("Preprocessing:")
    print("  - SimpleImputer(strategy='median')")
    print("  - No feature scaling")

    print()
    print("Random Forest parameters:")

    for parameter, value in RF_PARAMS.items():
        print(f"  {parameter}: {value}")

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
                RandomForestClassifier(
                    **RF_PARAMS
                ),
            ),
        ]
    )

    # =========================================================================
    # 6. Train model
    # =========================================================================

    print()
    print("=" * 80)
    print("TRAINING RANDOM FOREST")
    print("=" * 80)

    print(
        "Fitting median imputer on training data "
        "and training Random Forest..."
    )

    model.fit(
        X_train,
        y_train,
    )

    print("Training complete.")

    # =========================================================================
    # 7. Generate probability predictions
    # =========================================================================

    print()
    print("=" * 80)
    print("GENERATING PREDICTIONS")
    print("=" * 80)

    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    print("Validation predictions generated.")
    print("Test predictions generated.")

    # =========================================================================
    # 8. Evaluate validation performance
    # =========================================================================

    validation_metrics = evaluate_predictions(
        "Validation",
        y_validation,
        validation_probabilities,
    )

    # =========================================================================
    # 9. Evaluate test performance
    # =========================================================================

    test_metrics = evaluate_predictions(
        "Test",
        y_test,
        test_probabilities,
    )

    # =========================================================================
    # 10. Save complete model pipeline
    # =========================================================================

    model_path = OUTPUT_DIR / "model.joblib"

    joblib.dump(
        model,
        model_path,
    )

    print()
    print("=" * 80)
    print("MODEL SAVED")
    print("=" * 80)

    print(f"Complete model pipeline: {model_path}")

    # =========================================================================
    # 11. Save predictions
    # =========================================================================

    print()
    print("=" * 80)
    print("SAVING PREDICTIONS")
    print("=" * 80)

    save_predictions(
        "Validation",
        validation_df,
        validation_probabilities,
    )

    save_predictions(
        "Test",
        test_df,
        test_probabilities,
    )

    # =========================================================================
    # 12. Save metrics
    # =========================================================================

    metrics_df = pd.DataFrame([
        validation_metrics,
        test_metrics,
    ])

    metrics_path = OUTPUT_DIR / "metrics.csv"

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    print(f"Saved metrics: {metrics_path}")

    # =========================================================================
    # 13. Save feature list
    # =========================================================================

    feature_list_df = pd.DataFrame({
        "feature": feature_columns,
    })

    feature_list_path = OUTPUT_DIR / "features.csv"

    feature_list_df.to_csv(
        feature_list_path,
        index=False,
    )

    print(f"Saved feature list: {feature_list_path}")

    # =========================================================================
    # 14. Save model configuration
    # =========================================================================

    config_df = pd.DataFrame({
        "parameter": list(RF_PARAMS.keys()),
        "value": [
            str(value)
            for value in RF_PARAMS.values()
        ],
    })

    # Add preprocessing information.
    preprocessing_df = pd.DataFrame({
        "parameter": [
            "imputation_strategy",
            "feature_scaling",
        ],
        "value": [
            "median",
            "none",
        ],
    })

    config_df = pd.concat(
        [
            config_df,
            preprocessing_df,
        ],
        ignore_index=True,
    )

    config_path = OUTPUT_DIR / "model_config.csv"

    config_df.to_csv(
        config_path,
        index=False,
    )

    print(f"Saved model configuration: {config_path}")

    # =========================================================================
    # 15. Final summary
    # =========================================================================

    print()
    print("=" * 80)
    print("RANDOM FOREST MODEL 1 COMPLETE")
    print("=" * 80)

    print()
    print("Validation:")
    print(
        f"  Log Loss:    "
        f"{validation_metrics['log_loss']:.6f}"
    )
    print(
        f"  Brier Score: "
        f"{validation_metrics['brier_score']:.6f}"
    )
    print(
        f"  ROC AUC:     "
        f"{validation_metrics['roc_auc']:.6f}"
    )
    print(
        f"  Accuracy:    "
        f"{validation_metrics['accuracy']:.4%}"
    )

    print()
    print("Test:")
    print(
        f"  Log Loss:    "
        f"{test_metrics['log_loss']:.6f}"
    )
    print(
        f"  Brier Score: "
        f"{test_metrics['brier_score']:.6f}"
    )
    print(
        f"  ROC AUC:     "
        f"{test_metrics['roc_auc']:.6f}"
    )
    print(
        f"  Accuracy:    "
        f"{test_metrics['accuracy']:.4%}"
    )

    print()
    print("Artifacts saved to:")
    print(f"  {OUTPUT_DIR}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()