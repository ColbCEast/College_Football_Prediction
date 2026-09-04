"""
Random Forest Model 3 — Model 2 vs Model 3 Audit

Purpose
-------
Compare Random Forest Model 2 and Model 3 as a controlled A/B experiment.

Model 2:
    44 predictors
    Compact base features + Trend + Prior SOS

Model 3:
    28 predictors
    Compact base features only

The primary question is whether removing Trend and Prior SOS features:

    1. Preserves or improves probability quality
    2. Preserves or improves discrimination
    3. Improves calibration
    4. Produces a more stable / simpler model
    5. Avoids unnecessary complexity

Primary probability-quality metrics:
    - Log Loss
    - Brier Score

Secondary metrics:
    - ROC AUC
    - Accuracy
    - Balanced Accuracy
    - Precision
    - Recall

Additional diagnostics:
    - Calibration by probability bin
    - Expected Calibration Error (ECE)
    - Maximum Calibration Error (MCE)
    - Prediction distribution comparison
    - Game-level prediction differences
    - Prediction agreement / disagreement
    - Feature importance
    - Feature importance concentration
    - Performance by season
    - Model complexity comparison

Important experimental rule
---------------------------
The 2025 test set is evaluated but MUST NOT be used to make the feature
selection decision. Model selection should primarily be based on validation
performance (2023–2024).

Prediction file schema
----------------------
The training scripts save:

    gameId
    season
    win_home_actual
    win_home_probability
    win_home_prediction
    split

Outputs
-------
models/win_probability/random_forest/model_3/audit/

    model_2_vs_model_3_summary.csv
    model_2_vs_model_3_metrics.csv
    model_2_vs_model_3_validation_metrics.csv
    model_2_vs_model_3_test_metrics.csv
    model_2_vs_model_3_calibration.csv
    model_2_vs_model_3_calibration_summary.csv
    model_2_vs_model_3_prediction_distribution.csv
    model_2_vs_model_3_prediction_differences.csv
    model_2_vs_model_3_agreement.csv
    model_2_feature_importance.csv
    model_3_feature_importance.csv
    model_2_vs_model_3_feature_importance_comparison.csv
    model_2_vs_model_3_feature_importance_concentration.csv
    model_2_vs_model_3_season_metrics.csv
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[6]

MODEL_ROOT = (
    PROJECT_ROOT
    / "models"
    / "win_probability"
    / "random_forest"
)

MODEL_2_DIR = MODEL_ROOT / "model_2"
MODEL_3_DIR = MODEL_ROOT / "model_3"

AUDIT_DIR = MODEL_3_DIR / "audit"


# =============================================================================
# PREDICTION FILE SCHEMA
# =============================================================================

GAME_ID = "gameId"
SEASON = "season"

ACTUAL_COLUMN = "win_home_actual"
PROBABILITY_COLUMN = "win_home_probability"
PREDICTION_COLUMN = "win_home_prediction"

SPLIT_COLUMN = "split"


# =============================================================================
# EXPERIMENT CONSTANTS
# =============================================================================

MODEL_2_FEATURE_COUNT = 44
MODEL_3_FEATURE_COUNT = 28

TREND_FEATURE_COUNT = 12
SOS_FEATURE_COUNT = 4

CALIBRATION_BINS = 10


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def ensure_output_directory():
    """Create the audit output directory if necessary."""

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_predictions(model_dir, split_name):
    """
    Load saved prediction file and validate its schema.
    """

    path = (
        model_dir
        / f"{split_name}_predictions.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{path}"
        )

    df = pd.read_csv(path)

    required_columns = {
        GAME_ID,
        SEASON,
        ACTUAL_COLUMN,
        PROBABILITY_COLUMN,
        PREDICTION_COLUMN,
        SPLIT_COLUMN,
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{path.name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if df[GAME_ID].duplicated().any():
        duplicate_count = (
            df[GAME_ID].duplicated().sum()
        )

        raise ValueError(
            f"{path.name} contains "
            f"{duplicate_count:,} duplicate game IDs."
        )

    if df[ACTUAL_COLUMN].isna().any():
        raise ValueError(
            f"{path.name} contains missing actual outcomes."
        )

    if df[PROBABILITY_COLUMN].isna().any():
        raise ValueError(
            f"{path.name} contains missing probabilities."
        )

    if df[PREDICTION_COLUMN].isna().any():
        raise ValueError(
            f"{path.name} contains missing predictions."
        )

    probabilities = df[PROBABILITY_COLUMN]

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError(
            f"{path.name} contains probabilities outside [0, 1]."
        )

    actual_values = set(
        df[ACTUAL_COLUMN].dropna().unique()
    )

    if not actual_values.issubset({0, 1}):
        raise ValueError(
            f"{path.name} contains unexpected actual values: "
            f"{sorted(actual_values)}"
        )

    prediction_values = set(
        df[PREDICTION_COLUMN].dropna().unique()
    )

    if not prediction_values.issubset({0, 1}):
        raise ValueError(
            f"{path.name} contains unexpected prediction values: "
            f"{sorted(prediction_values)}"
        )

    return df


def load_feature_list(model_dir):
    """Load the feature list saved during model training."""

    path = model_dir / "feature_list.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Feature list not found:\n{path}"
        )

    df = pd.read_csv(path)

    if "feature" not in df.columns:
        raise ValueError(
            f"{path.name} must contain a 'feature' column."
        )

    return df["feature"].tolist()


def load_model(model_dir):
    """Load the trained Random Forest pipeline."""

    path = model_dir / "model.joblib"

    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{path}"
        )

    return joblib.load(path)


def calculate_metrics(
    y_true,
    probabilities,
    predictions,
):
    """Calculate probability and classification metrics."""

    return {
        "log_loss": log_loss(
            y_true,
            probabilities,
        ),
        "brier_score": brier_score_loss(
            y_true,
            probabilities,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
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
    }


def calibration_table(
    y_true,
    probabilities,
    model_name,
    split_name,
):
    """
    Create a calibration table using ten fixed probability bins.
    """

    df = pd.DataFrame(
        {
            "actual": y_true,
            "probability": probabilities,
        }
    )

    bins = np.linspace(
        0,
        1,
        CALIBRATION_BINS + 1,
    )

    df["bin"] = pd.cut(
        df["probability"],
        bins=bins,
        include_lowest=True,
        right=True,
    )

    rows = []

    for interval, group in df.groupby(
        "bin",
        observed=False,
    ):

        if len(group) == 0:
            continue

        mean_probability = (
            group["probability"].mean()
        )

        observed_rate = (
            group["actual"].mean()
        )

        calibration_error = abs(
            mean_probability
            - observed_rate
        )

        rows.append(
            {
                "model": model_name,
                "split": split_name,
                "probability_bin": str(interval),
                "count": len(group),
                "mean_predicted_probability": (
                    mean_probability
                ),
                "observed_home_win_rate": (
                    observed_rate
                ),
                "absolute_calibration_error": (
                    calibration_error
                ),
            }
        )

    return pd.DataFrame(rows)


def calculate_calibration_summary(
    y_true,
    probabilities,
    model_name,
    split_name,
):
    """Calculate Expected Calibration Error and Maximum Calibration Error."""

    df = pd.DataFrame(
        {
            "actual": y_true,
            "probability": probabilities,
        }
    )

    bins = np.linspace(
        0,
        1,
        CALIBRATION_BINS + 1,
    )

    df["bin"] = pd.cut(
        df["probability"],
        bins=bins,
        include_lowest=True,
        right=True,
    )

    errors = []

    for _, group in df.groupby(
        "bin",
        observed=False,
    ):

        if len(group) == 0:
            continue

        predicted = (
            group["probability"].mean()
        )

        observed = (
            group["actual"].mean()
        )

        errors.append(
            {
                "count": len(group),
                "error": abs(
                    predicted
                    - observed
                ),
            }
        )

    if not errors:
        ece = np.nan
        mce = np.nan

    else:
        total = sum(
            row["count"]
            for row in errors
        )

        ece = (
            sum(
                row["count"] * row["error"]
                for row in errors
            )
            / total
        )

        mce = max(
            row["error"]
            for row in errors
        )

    return {
        "model": model_name,
        "split": split_name,
        "ece": ece,
        "mce": mce,
    }


def prediction_distribution(
    probabilities,
    model_name,
    split_name,
):
    """Calculate prediction distribution statistics."""

    p = probabilities

    return {
        "model": model_name,
        "split": split_name,
        "mean": p.mean(),
        "std": p.std(),
        "min": p.min(),
        "p01": p.quantile(0.01),
        "p05": p.quantile(0.05),
        "p10": p.quantile(0.10),
        "p25": p.quantile(0.25),
        "p50": p.quantile(0.50),
        "p75": p.quantile(0.75),
        "p90": p.quantile(0.90),
        "p95": p.quantile(0.95),
        "p99": p.quantile(0.99),
        "max": p.max(),
    }


def get_season_metrics(
    prediction_df,
    model_name,
    split_name,
):
    """
    Calculate performance metrics separately for each season.
    """

    rows = []

    for season, group in prediction_df.groupby(
        SEASON
    ):

        y_true = group[ACTUAL_COLUMN]

        probabilities = (
            group[PROBABILITY_COLUMN]
        )

        predictions = (
            group[PREDICTION_COLUMN]
        )

        if y_true.nunique() >= 2:
            auc = roc_auc_score(
                y_true,
                probabilities,
            )

            balanced_accuracy = (
                balanced_accuracy_score(
                    y_true,
                    predictions,
                )
            )

        else:
            auc = np.nan
            balanced_accuracy = np.nan

        rows.append(
            {
                "model": model_name,
                "split": split_name,
                "season": season,
                "rows": len(group),
                "home_win_rate": y_true.mean(),
                "log_loss": log_loss(
                    y_true,
                    probabilities,
                ),
                "brier_score": brier_score_loss(
                    y_true,
                    probabilities,
                ),
                "roc_auc": auc,
                "accuracy": accuracy_score(
                    y_true,
                    predictions,
                ),
                "balanced_accuracy": (
                    balanced_accuracy
                ),
                "mean_probability": (
                    probabilities.mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def extract_feature_importance(
    model,
    feature_names,
):
    """
    Extract Random Forest feature importance from
    the fitted Pipeline.
    """

    if not hasattr(
        model,
        "named_steps",
    ):
        raise ValueError(
            "Expected the saved model to be a sklearn Pipeline."
        )

    if "model" not in model.named_steps:
        raise ValueError(
            "Pipeline does not contain a step named 'model'."
        )

    rf = model.named_steps["model"]

    if not hasattr(
        rf,
        "feature_importances_",
    ):
        raise ValueError(
            "Final model does not expose feature_importances_."
        )

    importances = (
        rf.feature_importances_
    )

    if len(importances) != len(feature_names):
        raise ValueError(
            "Feature importance count does not match "
            "feature list count."
        )

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    result = result.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    result["rank"] = (
        np.arange(
            1,
            len(result) + 1,
        )
    )

    result["cumulative_importance"] = (
        result["importance"].cumsum()
    )

    return result


def top_k_importance(
    importance_df,
    k,
):
    """Calculate cumulative importance of the top k features."""

    return (
        importance_df
        .head(k)["importance"]
        .sum()
    )


# =============================================================================
# MAIN AUDIT
# =============================================================================


def main():

    print("=" * 80)
    print("RANDOM FOREST MODEL 2 vs MODEL 3 — AUDIT")
    print("=" * 80)

    ensure_output_directory()

    # =========================================================================
    # REQUIRED FILES
    # =========================================================================

    print("\nChecking required files...")

    required_files = [
        MODEL_2_DIR / "model.joblib",
        MODEL_3_DIR / "model.joblib",
        MODEL_2_DIR / "feature_list.csv",
        MODEL_3_DIR / "feature_list.csv",
        MODEL_2_DIR / "validation_predictions.csv",
        MODEL_3_DIR / "validation_predictions.csv",
        MODEL_2_DIR / "test_predictions.csv",
        MODEL_3_DIR / "test_predictions.csv",
    ]

    for path in required_files:

        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    print("All required model artifacts found.")

    # =========================================================================
    # LOAD PREDICTIONS
    # =========================================================================

    print("\nLoading predictions...")

    model_2_validation = load_predictions(
        MODEL_2_DIR,
        "validation",
    )

    model_3_validation = load_predictions(
        MODEL_3_DIR,
        "validation",
    )

    model_2_test = load_predictions(
        MODEL_2_DIR,
        "test",
    )

    model_3_test = load_predictions(
        MODEL_3_DIR,
        "test",
    )

    print(
        f"  Model 2 validation: "
        f"{len(model_2_validation):,} rows"
    )

    print(
        f"  Model 3 validation: "
        f"{len(model_3_validation):,} rows"
    )

    print(
        f"  Model 2 test:       "
        f"{len(model_2_test):,} rows"
    )

    print(
        f"  Model 3 test:       "
        f"{len(model_3_test):,} rows"
    )

    # =========================================================================
    # GAME ALIGNMENT
    # =========================================================================

    print("\nValidating game alignment...")

    for split_name, df2, df3 in [
        (
            "validation",
            model_2_validation,
            model_3_validation,
        ),
        (
            "test",
            model_2_test,
            model_3_test,
        ),
    ]:

        ids2 = set(
            df2[GAME_ID]
        )

        ids3 = set(
            df3[GAME_ID]
        )

        if ids2 != ids3:

            only_model_2 = (
                ids2 - ids3
            )

            only_model_3 = (
                ids3 - ids2
            )

            raise ValueError(
                f"Game alignment mismatch in {split_name}.\n"
                f"Only Model 2: {len(only_model_2)}\n"
                f"Only Model 3: {len(only_model_3)}"
            )

        actual_mismatch = (
            df2[
                [
                    GAME_ID,
                    ACTUAL_COLUMN,
                ]
            ]
            .merge(
                df3[
                    [
                        GAME_ID,
                        ACTUAL_COLUMN,
                    ]
                ],
                on=GAME_ID,
                suffixes=(
                    "_model_2",
                    "_model_3",
                ),
            )
        )

        if (
            actual_mismatch[
                f"{ACTUAL_COLUMN}_model_2"
            ]
            != actual_mismatch[
                f"{ACTUAL_COLUMN}_model_3"
            ]
        ).any():

            raise ValueError(
                f"Actual outcomes do not align in "
                f"{split_name}."
            )

        print(
            f"  {split_name.title():<12}: "
            f"{len(ids2):,} games aligned"
        )

    # =========================================================================
    # FEATURE LIST COMPARISON
    # =========================================================================

    model_2_features = load_feature_list(
        MODEL_2_DIR
    )

    model_3_features = load_feature_list(
        MODEL_3_DIR
    )

    model_2_set = set(
        model_2_features
    )

    model_3_set = set(
        model_3_features
    )

    shared_features = sorted(
        model_2_set & model_3_set
    )

    removed_features = sorted(
        model_2_set - model_3_set
    )

    added_features = sorted(
        model_3_set - model_2_set
    )

    print("\nFeature comparison:")

    print(
        f"  Model 2 features : "
        f"{len(model_2_features)}"
    )

    print(
        f"  Model 3 features : "
        f"{len(model_3_features)}"
    )

    print(
        f"  Shared features  : "
        f"{len(shared_features)}"
    )

    print(
        f"  Removed features : "
        f"{len(removed_features)}"
    )

    print(
        f"  Added features   : "
        f"{len(added_features)}"
    )

    if len(model_2_features) != MODEL_2_FEATURE_COUNT:
        raise ValueError(
            "Model 2 feature count does not match "
            f"expected {MODEL_2_FEATURE_COUNT}."
        )

    if len(model_3_features) != MODEL_3_FEATURE_COUNT:
        raise ValueError(
            "Model 3 feature count does not match "
            f"expected {MODEL_3_FEATURE_COUNT}."
        )

    expected_removed_count = (
        TREND_FEATURE_COUNT
        + SOS_FEATURE_COUNT
    )

    if len(removed_features) != expected_removed_count:
        raise ValueError(
            "Model 3 does not remove exactly "
            f"{expected_removed_count} features."
        )

    if added_features:
        raise ValueError(
            "Model 3 contains features that were not "
            "present in Model 2."
        )

    # =========================================================================
    # PERFORMANCE METRICS
    # =========================================================================

    metric_rows = []

    model_split_data = [
        (
            "model_2",
            model_2_validation,
            "validation",
        ),
        (
            "model_3",
            model_3_validation,
            "validation",
        ),
        (
            "model_2",
            model_2_test,
            "test",
        ),
        (
            "model_3",
            model_3_test,
            "test",
        ),
    ]

    for model_name, df, split_name in model_split_data:

        metrics = calculate_metrics(
            df[ACTUAL_COLUMN],
            df[PROBABILITY_COLUMN],
            df[PREDICTION_COLUMN],
        )

        metrics["model"] = model_name
        metrics["split"] = split_name
        metrics["rows"] = len(df)

        metric_rows.append(metrics)

    metrics_df = pd.DataFrame(
        metric_rows
    )

    validation_metrics = (
        metrics_df[
            metrics_df["split"]
            == "validation"
        ]
        .copy()
    )

    test_metrics = (
        metrics_df[
            metrics_df["split"]
            == "test"
        ]
        .copy()
    )

    # =========================================================================
    # CONSOLE PERFORMANCE COMPARISON
    # =========================================================================

    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)

    for split_name, df in [
        (
            "Validation",
            validation_metrics,
        ),
        (
            "Test",
            test_metrics,
        ),
    ]:

        m2 = df[
            df["model"] == "model_2"
        ].iloc[0]

        m3 = df[
            df["model"] == "model_3"
        ].iloc[0]

        print(f"\n{split_name}")

        print(
            f"  Log Loss       "
            f"Model 2: {m2['log_loss']:.6f} | "
            f"Model 3: {m3['log_loss']:.6f} | "
            f"Δ: "
            f"{m3['log_loss'] - m2['log_loss']:+.6f}"
        )

        print(
            f"  Brier Score    "
            f"Model 2: {m2['brier_score']:.6f} | "
            f"Model 3: {m3['brier_score']:.6f} | "
            f"Δ: "
            f"{m3['brier_score'] - m2['brier_score']:+.6f}"
        )

        print(
            f"  ROC AUC        "
            f"Model 2: {m2['roc_auc']:.6f} | "
            f"Model 3: {m3['roc_auc']:.6f} | "
            f"Δ: "
            f"{m3['roc_auc'] - m2['roc_auc']:+.6f}"
        )

        print(
            f"  Accuracy       "
            f"Model 2: {m2['accuracy']:.6f} | "
            f"Model 3: {m3['accuracy']:.6f} | "
            f"Δ: "
            f"{m3['accuracy'] - m2['accuracy']:+.6f}"
        )

        print(
            f"  Balanced Acc.  "
            f"Model 2: {m2['balanced_accuracy']:.6f} | "
            f"Model 3: {m3['balanced_accuracy']:.6f} | "
            f"Δ: "
            f"{m3['balanced_accuracy'] - m2['balanced_accuracy']:+.6f}"
        )

    # =========================================================================
    # CALIBRATION
    # =========================================================================

    calibration_frames = []

    calibration_summary_rows = []

    for model_name, df, split_name in model_split_data:

        calibration_frames.append(
            calibration_table(
                df[ACTUAL_COLUMN],
                df[PROBABILITY_COLUMN],
                model_name,
                split_name,
            )
        )

        calibration_summary_rows.append(
            calculate_calibration_summary(
                df[ACTUAL_COLUMN],
                df[PROBABILITY_COLUMN],
                model_name,
                split_name,
            )
        )

    calibration_df = pd.concat(
        calibration_frames,
        ignore_index=True,
    )

    calibration_summary_df = pd.DataFrame(
        calibration_summary_rows
    )

    # =========================================================================
    # PREDICTION DISTRIBUTIONS
    # =========================================================================

    distribution_rows = []

    for model_name, df, split_name in model_split_data:

        distribution_rows.append(
            prediction_distribution(
                df[PROBABILITY_COLUMN],
                model_name,
                split_name,
            )
        )

    distribution_df = pd.DataFrame(
        distribution_rows
    )

    # =========================================================================
    # GAME-LEVEL PREDICTION COMPARISON
    # =========================================================================

    print(
        "\nCalculating game-level prediction differences..."
    )

    prediction_difference_frames = []

    for split_name, df2, df3 in [
        (
            "validation",
            model_2_validation,
            model_3_validation,
        ),
        (
            "test",
            model_2_test,
            model_3_test,
        ),
    ]:

        comparison = (
            df2[
                [
                    GAME_ID,
                    SEASON,
                    ACTUAL_COLUMN,
                    PROBABILITY_COLUMN,
                    PREDICTION_COLUMN,
                ]
            ]
            .merge(
                df3[
                    [
                        GAME_ID,
                        PROBABILITY_COLUMN,
                        PREDICTION_COLUMN,
                    ]
                ],
                on=GAME_ID,
                suffixes=(
                    "_model_2",
                    "_model_3",
                ),
            )
        )

        comparison["split"] = split_name

        comparison[
            "probability_difference"
        ] = (
            comparison[
                f"{PROBABILITY_COLUMN}_model_3"
            ]
            - comparison[
                f"{PROBABILITY_COLUMN}_model_2"
            ]
        )

        comparison[
            "absolute_probability_difference"
        ] = (
            comparison[
                "probability_difference"
            ]
            .abs()
        )

        comparison[
            "prediction_agreement"
        ] = (
            comparison[
                f"{PREDICTION_COLUMN}_model_2"
            ]
            ==
            comparison[
                f"{PREDICTION_COLUMN}_model_3"
            ]
        )

        comparison["model_2_correct"] = (
            comparison[
                f"{PREDICTION_COLUMN}_model_2"
            ]
            == comparison[ACTUAL_COLUMN]
        )

        comparison["model_3_correct"] = (
            comparison[
                f"{PREDICTION_COLUMN}_model_3"
            ]
            == comparison[ACTUAL_COLUMN]
        )

        prediction_difference_frames.append(
            comparison
        )

    prediction_difference_df = pd.concat(
        prediction_difference_frames,
        ignore_index=True,
    )

    # =========================================================================
    # AGREEMENT SUMMARY
    # =========================================================================

    agreement_rows = []

    for split_name, group in (
        prediction_difference_df
        .groupby("split")
    ):

        agreement_rate = (
            group[
                "prediction_agreement"
            ]
            .mean()
        )

        disagreement_count = (
            ~group[
                "prediction_agreement"
            ]
        ).sum()

        model_2_only_correct = (
            group["model_2_correct"]
            & ~group["model_3_correct"]
        ).sum()

        model_3_only_correct = (
            group["model_3_correct"]
            & ~group["model_2_correct"]
        ).sum()

        both_correct = (
            group["model_2_correct"]
            & group["model_3_correct"]
        ).sum()

        both_wrong = (
            ~group["model_2_correct"]
            & ~group["model_3_correct"]
        ).sum()

        agreement_rows.append(
            {
                "split": split_name,
                "rows": len(group),
                "agreement_rate": (
                    agreement_rate
                ),
                "disagreement_rate": (
                    1 - agreement_rate
                ),
                "disagreement_count": (
                    disagreement_count
                ),
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "model_2_only_correct": (
                    model_2_only_correct
                ),
                "model_3_only_correct": (
                    model_3_only_correct
                ),
                "mean_absolute_probability_difference": (
                    group[
                        "absolute_probability_difference"
                    ].mean()
                ),
                "median_absolute_probability_difference": (
                    group[
                        "absolute_probability_difference"
                    ].median()
                ),
                "max_absolute_probability_difference": (
                    group[
                        "absolute_probability_difference"
                    ].max()
                ),
            }
        )

    agreement_df = pd.DataFrame(
        agreement_rows
    )

    # =========================================================================
    # FEATURE IMPORTANCE
    # =========================================================================

    print(
        "\nExtracting feature importance..."
    )

    model_2 = load_model(
        MODEL_2_DIR
    )

    model_3 = load_model(
        MODEL_3_DIR
    )

    model_2_importance = (
        extract_feature_importance(
            model_2,
            model_2_features,
        )
    )

    model_3_importance = (
        extract_feature_importance(
            model_3,
            model_3_features,
        )
    )

    # =========================================================================
    # FEATURE IMPORTANCE COMPARISON
    # =========================================================================

    importance_comparison = (
        model_2_importance[
            [
                "feature",
                "importance",
                "rank",
                "cumulative_importance",
            ]
        ]
        .rename(
            columns={
                "importance": "model_2_importance",
                "rank": "model_2_rank",
                "cumulative_importance": (
                    "model_2_cumulative_importance"
                ),
            }
        )
    )

    importance_comparison = (
        importance_comparison
        .merge(
            model_3_importance[
                [
                    "feature",
                    "importance",
                    "rank",
                    "cumulative_importance",
                ]
            ].rename(
                columns={
                    "importance": "model_3_importance",
                    "rank": "model_3_rank",
                    "cumulative_importance": (
                        "model_3_cumulative_importance"
                    ),
                }
            ),
            on="feature",
            how="outer",
        )
    )

    importance_comparison[
        "model_2_importance"
    ] = (
        importance_comparison[
            "model_2_importance"
        ]
        .fillna(0)
    )

    importance_comparison[
        "model_3_importance"
    ] = (
        importance_comparison[
            "model_3_importance"
        ]
        .fillna(0)
    )

    importance_comparison[
        "importance_difference"
    ] = (
        importance_comparison[
            "model_3_importance"
        ]
        - importance_comparison[
            "model_2_importance"
        ]
    )

    importance_comparison = (
        importance_comparison
        .sort_values(
            "model_3_importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # =========================================================================
    # FEATURE IMPORTANCE CONCENTRATION
    # =========================================================================

    importance_concentration_rows = []

    for model_name, importance_df in [
        (
            "model_2",
            model_2_importance,
        ),
        (
            "model_3",
            model_3_importance,
        ),
    ]:

        importance_concentration_rows.append(
            {
                "model": model_name,
                "feature_count": len(
                    importance_df
                ),
                "top_5_importance": (
                    top_k_importance(
                        importance_df,
                        5,
                    )
                ),
                "top_10_importance": (
                    top_k_importance(
                        importance_df,
                        10,
                    )
                ),
                "top_20_importance": (
                    top_k_importance(
                        importance_df,
                        min(
                            20,
                            len(importance_df),
                        ),
                    )
                ),
                "top_50pct_feature_count": (
                    (
                        importance_df[
                            importance_df[
                                "cumulative_importance"
                            ]
                            <= 0.50
                        ]
                        .shape[0]
                        + 1
                    )
                ),
                "top_80pct_feature_count": (
                    (
                        importance_df[
                            importance_df[
                                "cumulative_importance"
                            ]
                            <= 0.80
                        ]
                        .shape[0]
                        + 1
                    )
                ),
            }
        )

    importance_concentration_df = (
        pd.DataFrame(
            importance_concentration_rows
        )
    )

    # =========================================================================
    # SEASON-LEVEL METRICS
    # =========================================================================

    season_frames = []

    for model_name, df, split_name in model_split_data:

        season_df = get_season_metrics(
            df,
            model_name,
            split_name,
        )

        season_frames.append(
            season_df
        )

    season_metrics_df = pd.concat(
        season_frames,
        ignore_index=True,
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    validation_m2 = validation_metrics[
        validation_metrics["model"]
        == "model_2"
    ].iloc[0]

    validation_m3 = validation_metrics[
        validation_metrics["model"]
        == "model_3"
    ].iloc[0]

    test_m2 = test_metrics[
        test_metrics["model"]
        == "model_2"
    ].iloc[0]

    test_m3 = test_metrics[
        test_metrics["model"]
        == "model_3"
    ].iloc[0]

    val_cal_m2 = calibration_summary_df[
        (
            calibration_summary_df["model"]
            == "model_2"
        )
        &
        (
            calibration_summary_df["split"]
            == "validation"
        )
    ].iloc[0]

    val_cal_m3 = calibration_summary_df[
        (
            calibration_summary_df["model"]
            == "model_3"
        )
        &
        (
            calibration_summary_df["split"]
            == "validation"
        )
    ].iloc[0]

    test_cal_m2 = calibration_summary_df[
        (
            calibration_summary_df["model"]
            == "model_2"
        )
        &
        (
            calibration_summary_df["split"]
            == "test"
        )
    ].iloc[0]

    test_cal_m3 = calibration_summary_df[
        (
            calibration_summary_df["model"]
            == "model_3"
        )
        &
        (
            calibration_summary_df["split"]
            == "test"
        )
    ].iloc[0]

    validation_agreement = agreement_df[
        agreement_df["split"]
        == "validation"
    ].iloc[0]

    test_agreement = agreement_df[
        agreement_df["split"]
        == "test"
    ].iloc[0]

    summary = pd.DataFrame(
        [
            {
                # -------------------------------------------------------------
                # Complexity
                # -------------------------------------------------------------

                "model_2_feature_count": (
                    len(model_2_features)
                ),

                "model_3_feature_count": (
                    len(model_3_features)
                ),

                "features_removed": (
                    len(removed_features)
                ),

                "trend_features_removed": (
                    TREND_FEATURE_COUNT
                ),

                "sos_features_removed": (
                    SOS_FEATURE_COUNT
                ),

                "feature_reduction_percent": (
                    (
                        len(removed_features)
                        / len(model_2_features)
                    )
                    * 100
                ),

                # -------------------------------------------------------------
                # Validation — primary metrics
                # -------------------------------------------------------------

                "validation_model_2_log_loss": (
                    validation_m2[
                        "log_loss"
                    ]
                ),

                "validation_model_3_log_loss": (
                    validation_m3[
                        "log_loss"
                    ]
                ),

                "validation_log_loss_delta_model_3_minus_model_2": (
                    validation_m3[
                        "log_loss"
                    ]
                    - validation_m2[
                        "log_loss"
                    ]
                ),

                "validation_model_2_brier": (
                    validation_m2[
                        "brier_score"
                    ]
                ),

                "validation_model_3_brier": (
                    validation_m3[
                        "brier_score"
                    ]
                ),

                "validation_brier_delta_model_3_minus_model_2": (
                    validation_m3[
                        "brier_score"
                    ]
                    - validation_m2[
                        "brier_score"
                    ]
                ),

                # -------------------------------------------------------------
                # Validation — discrimination
                # -------------------------------------------------------------

                "validation_model_2_auc": (
                    validation_m2[
                        "roc_auc"
                    ]
                ),

                "validation_model_3_auc": (
                    validation_m3[
                        "roc_auc"
                    ]
                ),

                "validation_auc_delta_model_3_minus_model_2": (
                    validation_m3[
                        "roc_auc"
                    ]
                    - validation_m2[
                        "roc_auc"
                    ]
                ),

                # -------------------------------------------------------------
                # Validation — calibration
                # -------------------------------------------------------------

                "validation_model_2_ece": (
                    val_cal_m2["ece"]
                ),

                "validation_model_3_ece": (
                    val_cal_m3["ece"]
                ),

                "validation_ece_delta_model_3_minus_model_2": (
                    val_cal_m3["ece"]
                    - val_cal_m2["ece"]
                ),

                "validation_model_2_mce": (
                    val_cal_m2["mce"]
                ),

                "validation_model_3_mce": (
                    val_cal_m3["mce"]
                ),

                "validation_mce_delta_model_3_minus_model_2": (
                    val_cal_m3["mce"]
                    - val_cal_m2["mce"]
                ),

                # -------------------------------------------------------------
                # Test — primary metrics
                # -------------------------------------------------------------

                "test_model_2_log_loss": (
                    test_m2["log_loss"]
                ),

                "test_model_3_log_loss": (
                    test_m3["log_loss"]
                ),

                "test_log_loss_delta_model_3_minus_model_2": (
                    test_m3["log_loss"]
                    - test_m2["log_loss"]
                ),

                "test_model_2_brier": (
                    test_m2["brier_score"]
                ),

                "test_model_3_brier": (
                    test_m3["brier_score"]
                ),

                "test_brier_delta_model_3_minus_model_2": (
                    test_m3["brier_score"]
                    - test_m2["brier_score"]
                ),

                # -------------------------------------------------------------
                # Test — discrimination
                # -------------------------------------------------------------

                "test_model_2_auc": (
                    test_m2["roc_auc"]
                ),

                "test_model_3_auc": (
                    test_m3["roc_auc"]
                ),

                "test_auc_delta_model_3_minus_model_2": (
                    test_m3["roc_auc"]
                    - test_m2["roc_auc"]
                ),

                # -------------------------------------------------------------
                # Test — calibration
                # -------------------------------------------------------------

                "test_model_2_ece": (
                    test_cal_m2["ece"]
                ),

                "test_model_3_ece": (
                    test_cal_m3["ece"]
                ),

                "test_ece_delta_model_3_minus_model_2": (
                    test_cal_m3["ece"]
                    - test_cal_m2["ece"]
                ),

                "test_model_2_mce": (
                    test_cal_m2["mce"]
                ),

                "test_model_3_mce": (
                    test_cal_m3["mce"]
                ),

                "test_mce_delta_model_3_minus_model_2": (
                    test_cal_m3["mce"]
                    - test_cal_m2["mce"]
                ),

                # -------------------------------------------------------------
                # Prediction agreement
                # -------------------------------------------------------------

                "validation_prediction_agreement": (
                    validation_agreement[
                        "agreement_rate"
                    ]
                ),

                "validation_mean_abs_probability_difference": (
                    validation_agreement[
                        "mean_absolute_probability_difference"
                    ]
                ),

                "test_prediction_agreement": (
                    test_agreement[
                        "agreement_rate"
                    ]
                ),

                "test_mean_abs_probability_difference": (
                    test_agreement[
                        "mean_absolute_probability_difference"
                    ]
                ),
            }
        ]
    )

    # =========================================================================
    # SAVE OUTPUTS
    # =========================================================================

    print(
        "\nSaving audit outputs..."
    )

    metrics_df.to_csv(
        AUDIT_DIR
        / "model_2_vs_model_3_metrics.csv",
        index=False,
    )

    validation_metrics.to_csv(
        AUDIT_DIR
        / "model_2_vs_model_3_validation_metrics.csv",
        index=False,
    )

    test_metrics.to_csv(
        AUDIT_DIR
        / "model_2_vs_model_3_test_metrics.csv",
        index=False,
    )

    calibration_df.to_csv(
        AUDIT_DIR
        / "model_2_vs_model_3_calibration.csv",
        index=False,
    )

    calibration_summary_df.to_csv(
        AUDIT_DIR
        / "model_2_vs_model_3_calibration_summary.csv",
        index=False,
    )

    distribution_df.to_csv(
        AUDIT_DIR
        / "model_2_vs_model_3_prediction_distribution.csv",
        index=False,
    )

    prediction_difference_df.to_csv(
        AUDIT_DIR
        / "model_2_vs_model_3_prediction_differences.csv",
        index=False,
    )

    agreement_df.to_csv(
        AUDIT_DIR
        / "model_2_vs_model_3_agreement.csv",
        index=False,
    )

    model_2_importance.to_csv(
        AUDIT_DIR
        / "model_2_feature_importance.csv",
        index=False,
    )

    model_3_importance.to_csv(
        AUDIT_DIR
        / "model_3_feature_importance.csv",
        index=False,
    )

    importance_comparison.to_csv(
        AUDIT_DIR
        / "model_2_vs_model_3_feature_importance_comparison.csv",
        index=False,
    )

    importance_concentration_df.to_csv(
        AUDIT_DIR
        / "model_2_vs_model_3_feature_importance_concentration.csv",
        index=False,
    )

    season_metrics_df.to_csv(
        AUDIT_DIR
        / "model_2_vs_model_3_season_metrics.csv",
        index=False,
    )

    summary.to_csv(
        AUDIT_DIR
        / "model_2_vs_model_3_summary.csv",
        index=False,
    )

    # =========================================================================
    # FINAL CONSOLE SUMMARY
    # =========================================================================

    print("\n" + "=" * 80)
    print("FINAL AUDIT SUMMARY")
    print("=" * 80)

    print("\nFeature reduction:")

    print(
        f"  Model 2: "
        f"{len(model_2_features)} features"
    )

    print(
        f"  Model 3: "
        f"{len(model_3_features)} features"
    )

    print(
        f"  Removed: "
        f"{len(removed_features)} features "
        f"("
        f"Trend={TREND_FEATURE_COUNT}, "
        f"SOS={SOS_FEATURE_COUNT}"
        f")"
    )

    print(
        f"  Reduction: "
        f"{summary.iloc[0]['feature_reduction_percent']:.1f}%"
    )

    print(
        "\nValidation — primary probability metrics:"
    )

    print(
        f"  Log Loss:"
        f"  Model 2 = "
        f"{validation_m2['log_loss']:.6f}"
        f" | Model 3 = "
        f"{validation_m3['log_loss']:.6f}"
        f" | Δ = "
        f"{validation_m3['log_loss'] - validation_m2['log_loss']:+.6f}"
    )

    print(
        f"  Brier:"
        f"     Model 2 = "
        f"{validation_m2['brier_score']:.6f}"
        f" | Model 3 = "
        f"{validation_m3['brier_score']:.6f}"
        f" | Δ = "
        f"{validation_m3['brier_score'] - validation_m2['brier_score']:+.6f}"
    )

    print(
        "\nValidation — discrimination:"
    )

    print(
        f"  ROC AUC:"
        f"    Model 2 = "
        f"{validation_m2['roc_auc']:.6f}"
        f" | Model 3 = "
        f"{validation_m3['roc_auc']:.6f}"
        f" | Δ = "
        f"{validation_m3['roc_auc'] - validation_m2['roc_auc']:+.6f}"
    )

    print(
        "\nValidation — calibration:"
    )

    print(
        f"  ECE:"
        f"        Model 2 = "
        f"{val_cal_m2['ece']:.6f}"
        f" | Model 3 = "
        f"{val_cal_m3['ece']:.6f}"
        f" | Δ = "
        f"{val_cal_m3['ece'] - val_cal_m2['ece']:+.6f}"
    )

    print(
        f"  MCE:"
        f"        Model 2 = "
        f"{val_cal_m2['mce']:.6f}"
        f" | Model 3 = "
        f"{val_cal_m3['mce']:.6f}"
        f" | Δ = "
        f"{val_cal_m3['mce'] - val_cal_m2['mce']:+.6f}"
    )

    print(
        "\nTest — primary probability metrics:"
    )

    print(
        f"  Log Loss:"
        f"  Model 2 = "
        f"{test_m2['log_loss']:.6f}"
        f" | Model 3 = "
        f"{test_m3['log_loss']:.6f}"
        f" | Δ = "
        f"{test_m3['log_loss'] - test_m2['log_loss']:+.6f}"
    )

    print(
        f"  Brier:"
        f"     Model 2 = "
        f"{test_m2['brier_score']:.6f}"
        f" | Model 3 = "
        f"{test_m3['brier_score']:.6f}"
        f" | Δ = "
        f"{test_m3['brier_score'] - test_m2['brier_score']:+.6f}"
    )

    print(
        "\nTest — discrimination:"
    )

    print(
        f"  ROC AUC:"
        f"    Model 2 = "
        f"{test_m2['roc_auc']:.6f}"
        f" | Model 3 = "
        f"{test_m3['roc_auc']:.6f}"
        f" | Δ = "
        f"{test_m3['roc_auc'] - test_m2['roc_auc']:+.6f}"
    )

    print(
        "\nTest — calibration:"
    )

    print(
        f"  ECE:"
        f"        Model 2 = "
        f"{test_cal_m2['ece']:.6f}"
        f" | Model 3 = "
        f"{test_cal_m3['ece']:.6f}"
        f" | Δ = "
        f"{test_cal_m3['ece'] - test_cal_m2['ece']:+.6f}"
    )

    print(
        f"  MCE:"
        f"        Model 2 = "
        f"{test_cal_m2['mce']:.6f}"
        f" | Model 3 = "
        f"{test_cal_m3['mce']:.6f}"
        f" | Δ = "
        f"{test_cal_m3['mce'] - test_cal_m2['mce']:+.6f}"
    )

    print(
        "\nPrediction agreement:"
    )

    print(
        f"  Validation: "
        f"{validation_agreement['agreement_rate']:.2%}"
    )

    print(
        f"  Test:       "
        f"{test_agreement['agreement_rate']:.2%}"
    )

    print(
        "\nAudit outputs saved to:"
    )

    print(
        f"  {AUDIT_DIR}"
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "AUDIT COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()