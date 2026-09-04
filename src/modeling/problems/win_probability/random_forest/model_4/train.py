"""
Random Forest Model 4 — Hyperparameter Tuning

Purpose
-------
Tune the Random Forest hyperparameters using the Model 3 compact
28-feature specification.

Experimental controls
---------------------
- Same 28 predictors as Model 3
- Same temporal train/validation/test split
- Same target
- Same median imputation strategy
- Same random seed
- Test set remains untouched during model selection

Model selection criterion
-------------------------
Primary: Validation Log Loss

Secondary metrics:
- Brier Score
- ROC AUC
- Accuracy
- Balanced Accuracy

Outputs
-------
models/win_probability/random_forest/model_4/
    model.joblib
    feature_list.csv
    training_summary.csv
    validation_predictions.csv
    test_predictions.csv
    tuning_results.csv
    best_params.csv
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42

TARGET = "win_home"
GAME_ID = "gameId"
SEASON = "season"

# Input data
TRAIN_PATH = Path(
    "data/processed/model_inputs/win_probability/train.csv"
)

VALIDATION_PATH = Path(
    "data/processed/model_inputs/win_probability/validation.csv"
)

TEST_PATH = Path(
    "data/processed/model_inputs/win_probability/test.csv"
)

# Model output directory
OUTPUT_DIR = Path(
    "models/win_probability/random_forest/model_4"
)

# Tuning configuration
N_ITER = 50

# Cross-validation is NOT being used to select the final temporal model.
# The official 2023–2024 validation set is used for model selection after
# fitting each candidate on the fixed 2015–2022 training set.
#
# RandomizedSearchCV still requires CV internally. We use a small CV only
# within the training period so that the validation and test periods remain
# completely untouched.
CV_FOLDS = 3


# =============================================================================
# MODEL 3 FEATURE SET
# =============================================================================

FEATURES = [
    # -------------------------------------------------------------------------
    # Core strength — 8
    # -------------------------------------------------------------------------
    "homePregameElo",
    "awayPregameElo",
    "winPctBefore_home",
    "winPctBefore_away",
    "pointDifferentialBefore_home",
    "pointDifferentialBefore_away",
    "pointDifferentialAvgBefore_home",
    "pointDifferentialAvgBefore_away",

    # -------------------------------------------------------------------------
    # Recent form — 8
    # -------------------------------------------------------------------------
    "pointDifferentialAvgLast3_home",
    "pointDifferentialAvgLast3_away",
    "pointDifferentialAvgLast5_home",
    "pointDifferentialAvgLast5_away",
    "pointsForAvgLast5_home",
    "pointsForAvgLast5_away",
    "pointsAgainstAvgLast5_home",
    "pointsAgainstAvgLast5_away",

    # -------------------------------------------------------------------------
    # Offensive efficiency — 8
    # -------------------------------------------------------------------------
    "home_pregame_offense_successRate",
    "away_pregame_offense_successRate",
    "home_pregame_offense_ppa",
    "away_pregame_offense_ppa",
    "yardsPerPassAttemptBefore_home",
    "yardsPerPassAttemptBefore_away",
    "yardsPerRushAttemptBefore_home",
    "yardsPerRushAttemptBefore_away",

    # -------------------------------------------------------------------------
    # Defensive efficiency — 4
    # -------------------------------------------------------------------------
    "home_pregame_defense_successRate",
    "away_pregame_defense_successRate",
    "home_pregame_defense_ppa",
    "away_pregame_defense_ppa",
]


# =============================================================================
# HYPERPARAMETER SEARCH SPACE
# =============================================================================

PARAM_DISTRIBUTIONS = {
    "model__n_estimators": [
        300,
        500,
        750,
        1000,
    ],
    "model__max_depth": [
        None,
        8,
        12,
        16,
        20,
        30,
    ],
    "model__min_samples_split": [
        2,
        5,
        10,
        20,
        30,
    ],
    "model__min_samples_leaf": [
        1,
        2,
        5,
        10,
        15,
        20,
    ],
    "model__max_features": [
        "sqrt",
        "log2",
        0.25,
        0.50,
        0.75,
        1.0,
    ],
    "model__bootstrap": [
        True,
        False,
    ],
}


# =============================================================================
# FUNCTIONS
# =============================================================================

def load_data():
    """Load temporal train, validation, and test datasets."""

    print("Loading data...")

    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    test = pd.read_csv(TEST_PATH)

    print(f"  Training:   {train.shape}")
    print(f"  Validation: {validation.shape}")
    print(f"  Test:       {test.shape}")

    return train, validation, test


def validate_data(train, validation, test):
    """Validate required columns and feature availability."""

    print("\nValidating data...")

    required_columns = set(FEATURES + [TARGET, GAME_ID, SEASON])

    for name, df in [
        ("training", train),
        ("validation", validation),
        ("test", test),
    ]:
        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"{name.capitalize()} dataset is missing required columns: "
                f"{sorted(missing)}"
            )

        if df[GAME_ID].duplicated().any():
            raise ValueError(
                f"{name.capitalize()} dataset contains duplicate game IDs."
            )

    if train[TARGET].isna().any():
        raise ValueError("Training target contains missing values.")

    if validation[TARGET].isna().any():
        raise ValueError("Validation target contains missing values.")

    if test[TARGET].isna().any():
        raise ValueError("Test target contains missing values.")

    train_seasons = sorted(train[SEASON].unique())
    validation_seasons = sorted(validation[SEASON].unique())
    test_seasons = sorted(test[SEASON].unique())

    print(f"  Training seasons:   {train_seasons}")
    print(f"  Validation seasons: {validation_seasons}")
    print(f"  Test seasons:       {test_seasons}")

    print(f"  Features:           {len(FEATURES)}")

    if len(FEATURES) != 28:
        raise ValueError(
            f"Expected exactly 28 Model 3 features, found {len(FEATURES)}."
        )


def build_pipeline():
    """Build the preprocessing + Random Forest pipeline."""

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "model",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    class_weight=None,
                ),
            ),
        ]
    )


def calculate_metrics(y_true, probabilities, predictions):
    """Calculate probability and classification metrics."""

    return {
        "log_loss": log_loss(y_true, probabilities),
        "brier_score": brier_score_loss(y_true, probabilities),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            predictions,
        ),
    }


def create_predictions(
    model,
    df,
    split_name,
):
    """Create standardized prediction output."""

    X = df[FEATURES]
    y = df[TARGET]

    probabilities = model.predict_proba(X)[:, 1]
    predictions = model.predict(X)

    return pd.DataFrame(
        {
            GAME_ID: df[GAME_ID].values,
            SEASON: df[SEASON].values,
            "win_home_actual": y.values,
            "win_home_probability": probabilities,
            "win_home_prediction": predictions,
            "split": split_name,
        }
    )


def main():

    print("=" * 80)
    print("RANDOM FOREST MODEL 4 — HYPERPARAMETER TUNING")
    print("=" * 80)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load and validate
    # -------------------------------------------------------------------------

    train, validation, test = load_data()

    validate_data(
        train,
        validation,
        test,
    )

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_validation = validation[FEATURES]
    y_validation = validation[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    # -------------------------------------------------------------------------
    # Training target summary
    # -------------------------------------------------------------------------

    print("\nTraining target distribution:")
    print(
        y_train.value_counts()
        .sort_index()
        .to_string()
    )

    print(
        f"\nTraining home win rate: "
        f"{y_train.mean():.4f}"
    )

    # -------------------------------------------------------------------------
    # Build pipeline
    # -------------------------------------------------------------------------

    pipeline = build_pipeline()

    # -------------------------------------------------------------------------
    # Hyperparameter tuning
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("HYPERPARAMETER SEARCH")
    print("=" * 80)

    print(f"\nRandomized search iterations: {N_ITER}")
    print(f"Internal CV folds:            {CV_FOLDS}")
    print("Scoring:                      neg_log_loss")
    print("\nSearch space:")

    for parameter, values in PARAM_DISTRIBUTIONS.items():
        print(f"  {parameter}: {values}")

    print(
        "\nIMPORTANT: "
        "The official validation and test datasets are not used during "
        "hyperparameter search."
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER,
        scoring="neg_log_loss",
        cv=CV_FOLDS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
        verbose=2,
    )

    search.fit(
        X_train,
        y_train,
    )

    print("\n" + "=" * 80)
    print("BEST HYPERPARAMETERS")
    print("=" * 80)

    print(f"\nBest internal CV Log Loss: {-search.best_score_:.6f}")

    for parameter, value in search.best_params_.items():
        print(f"  {parameter}: {value}")

    # -------------------------------------------------------------------------
    # Save tuning results
    # -------------------------------------------------------------------------

    print("\nSaving tuning results...")

    cv_results = pd.DataFrame(search.cv_results_)

    # Convert negative Log Loss to positive Log Loss for readability.
    cv_results["mean_test_log_loss"] = (
        -cv_results["mean_test_score"]
    )

    cv_results["mean_train_log_loss"] = (
        -cv_results["mean_train_score"]
    )

    cv_results = cv_results.sort_values(
        "mean_test_log_loss"
    )

    cv_results.to_csv(
        OUTPUT_DIR / "tuning_results.csv",
        index=False,
    )

    best_params_df = pd.DataFrame(
        [
            {
                parameter.replace("model__", ""): value
                for parameter, value in search.best_params_.items()
            }
        ]
    )

    best_params_df["internal_cv_log_loss"] = -search.best_score_

    best_params_df.to_csv(
        OUTPUT_DIR / "best_params.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Evaluate on official validation set
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("OFFICIAL VALIDATION EVALUATION")
    print("=" * 80)

    validation_probabilities = search.best_estimator_.predict_proba(
        X_validation
    )[:, 1]

    validation_predictions = search.best_estimator_.predict(
        X_validation
    )

    validation_metrics = calculate_metrics(
        y_validation,
        validation_probabilities,
        validation_predictions,
    )

    for metric, value in validation_metrics.items():
        print(f"  {metric:<20}: {value:.6f}")

    # -------------------------------------------------------------------------
    # Evaluate on untouched test set
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("OFFICIAL TEST EVALUATION")
    print("=" * 80)

    test_probabilities = search.best_estimator_.predict_proba(
        X_test
    )[:, 1]

    test_predictions = search.best_estimator_.predict(
        X_test
    )

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        test_predictions,
    )

    for metric, value in test_metrics.items():
        print(f"  {metric:<20}: {value:.6f}")

    # -------------------------------------------------------------------------
    # Save model
    # -------------------------------------------------------------------------

    print("\nSaving final tuned model...")

    joblib.dump(
        search.best_estimator_,
        OUTPUT_DIR / "model.joblib",
    )

    # -------------------------------------------------------------------------
    # Save feature list
    # -------------------------------------------------------------------------

    feature_list = pd.DataFrame(
        {
            "feature": FEATURES,
            "feature_order": range(1, len(FEATURES) + 1),
        }
    )

    feature_list.to_csv(
        OUTPUT_DIR / "feature_list.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save predictions
    # -------------------------------------------------------------------------

    validation_output = create_predictions(
        search.best_estimator_,
        validation,
        "validation",
    )

    test_output = create_predictions(
        search.best_estimator_,
        test,
        "test",
    )

    validation_output.to_csv(
        OUTPUT_DIR / "validation_predictions.csv",
        index=False,
    )

    test_output.to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save training summary
    # -------------------------------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "model": "random_forest_model_4",
                "feature_count": len(FEATURES),
                "train_rows": len(train),
                "validation_rows": len(validation),
                "test_rows": len(test),
                "train_seasons": ",".join(
                    map(str, sorted(train[SEASON].unique()))
                ),
                "validation_seasons": ",".join(
                    map(str, sorted(validation[SEASON].unique()))
                ),
                "test_seasons": ",".join(
                    map(str, sorted(test[SEASON].unique()))
                ),
                "internal_cv_log_loss": -search.best_score_,
                "validation_log_loss": validation_metrics["log_loss"],
                "validation_brier_score": validation_metrics[
                    "brier_score"
                ],
                "validation_roc_auc": validation_metrics["roc_auc"],
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_balanced_accuracy": validation_metrics[
                    "balanced_accuracy"
                ],
                "test_log_loss": test_metrics["log_loss"],
                "test_brier_score": test_metrics["brier_score"],
                "test_roc_auc": test_metrics["roc_auc"],
                "test_accuracy": test_metrics["accuracy"],
                "test_balanced_accuracy": test_metrics[
                    "balanced_accuracy"
                ],
                "random_state": RANDOM_STATE,
                "tuning_iterations": N_ITER,
                "cv_folds": CV_FOLDS,
            }
        ]
    )

    summary.to_csv(
        OUTPUT_DIR / "training_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Final output
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("MODEL 4 COMPLETE")
    print("=" * 80)

    print("\nArtifacts saved to:")
    print(f"  {OUTPUT_DIR.resolve()}")

    print("\nFiles:")
    print("  model.joblib")
    print("  feature_list.csv")
    print("  training_summary.csv")
    print("  validation_predictions.csv")
    print("  test_predictions.csv")
    print("  tuning_results.csv")
    print("  best_params.csv")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()