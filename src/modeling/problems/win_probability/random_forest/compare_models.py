"""
Compare Random Forest Win Probability Models 1-5

Purpose
-------
Compare Random Forest win-probability models using identical validation and
test datasets.

Primary model-selection criterion:
    Test Log Loss

Supporting criteria:
    - Brier Score
    - ROC AUC
    - Expected Calibration Error (ECE)
    - Maximum Calibration Error (MCE)
    - Accuracy
    - Balanced Accuracy
    - Precision
    - Recall

Additional analysis:
    - Prediction agreement between models
    - Mean absolute probability difference
    - Correlation of predicted probabilities
    - Champion model identification

Expected prediction file schema
--------------------------------
Each model's validation_predictions.csv and test_predictions.csv should
contain:

    gameId
    season
    startDate
    seasonType
    actual_win_home
    predicted_home_win_probability

Output files
------------
models/win_probability/random_forest/model_comparison.csv
models/win_probability/random_forest/model_agreement.csv
"""

from pathlib import Path
from itertools import combinations

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
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[5]

MODEL_ROOT = (
    PROJECT_ROOT
    / "models"
    / "win_probability"
    / "random_forest"
)

MODEL_NUMBERS = [1, 2, 3, 4, 5]

DATASETS = ["validation", "test"]

OUTPUT_COMPARISON = MODEL_ROOT / "model_comparison.csv"
OUTPUT_AGREEMENT = MODEL_ROOT / "model_agreement.csv"


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def print_header(title):
    """Print a major section header."""

    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_subheader(title):
    """Print a subsection header."""

    print()
    print("-" * 78)
    print(title)
    print("-" * 78)


# =============================================================================
# COLUMN IDENTIFICATION
# =============================================================================

def identify_prediction_columns(df):
    """
    Identify the prediction-file schema.

    The Random Forest models currently use two prediction schemas:

    Current schema:
        gameId
        actual_win_home
        predicted_home_win_probability

    Legacy Model 2 schema:
        gameId
        win_home_actual
        win_home_probability
        win_home_prediction
        split

    Returns
    -------
    dict
        Standardized mapping of logical column names to dataframe columns.
    """

    # -------------------------------------------------------------------------
    # Current schema
    # -------------------------------------------------------------------------

    current_schema = {
        "game_id": "gameId",
        "actual": "actual_win_home",
        "probability": "predicted_home_win_probability",
        "prediction": None,
    }

    if all(
        column in df.columns
        for column in [
            "gameId",
            "actual_win_home",
            "predicted_home_win_probability",
        ]
    ):
        return current_schema

    # -------------------------------------------------------------------------
    # Legacy Model 2 schema
    # -------------------------------------------------------------------------

    legacy_schema = {
        "game_id": "gameId",
        "actual": "win_home_actual",
        "probability": "win_home_probability",
        "prediction": (
            "win_home_prediction"
            if "win_home_prediction" in df.columns
            else None
        ),
    }

    if all(
        column in df.columns
        for column in [
            "gameId",
            "win_home_actual",
            "win_home_probability",
        ]
    ):
        return legacy_schema

    # -------------------------------------------------------------------------
    # Unknown schema
    # -------------------------------------------------------------------------

    raise ValueError(
        "Could not identify prediction-file schema.\n\n"
        "Expected either:\n"
        "  Current schema:\n"
        "    gameId\n"
        "    actual_win_home\n"
        "    predicted_home_win_probability\n\n"
        "  Legacy schema:\n"
        "    gameId\n"
        "    win_home_actual\n"
        "    win_home_probability\n\n"
        f"Available columns:\n{list(df.columns)}"
    )


# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_prediction_dataframe(
    df,
    model_number,
    dataset_name,
):
    """
    Validate an individual prediction dataframe.

    Checks:
        - Required columns exist
        - Game IDs are unique
        - Actual target contains only 0/1
        - Probabilities are numeric
        - Probabilities fall within [0, 1]
        - No missing IDs, targets, or probabilities
    """

    cols = identify_prediction_columns(df)

    game_id_column = cols["game_id"]
    actual_column = cols["actual"]
    probability_column = cols["probability"]

    # -------------------------------------------------------------------------
    # Missing values
    # -------------------------------------------------------------------------

    required_for_validation = [
        game_id_column,
        actual_column,
        probability_column,
    ]

    missing_values = df[required_for_validation].isna().sum()

    if missing_values.any():

        raise ValueError(
            f"MODEL {model_number} {dataset_name.upper()} "
            f"CONTAINS MISSING VALUES:\n"
            f"{missing_values[missing_values > 0]}"
        )

    # -------------------------------------------------------------------------
    # Game ID uniqueness
    # -------------------------------------------------------------------------

    duplicate_ids = df[game_id_column].duplicated().sum()

    if duplicate_ids > 0:

        raise ValueError(
            f"MODEL {model_number} {dataset_name.upper()} "
            f"CONTAINS {duplicate_ids} DUPLICATE GAME IDS."
        )

    # -------------------------------------------------------------------------
    # Target validation
    # -------------------------------------------------------------------------

    actual_values = set(
        pd.Series(df[actual_column]).unique()
    )

    if not actual_values.issubset({0, 1}):

        raise ValueError(
            f"MODEL {model_number} {dataset_name.upper()} "
            f"CONTAINS INVALID TARGET VALUES:\n"
            f"{sorted(actual_values)}"
        )

    # -------------------------------------------------------------------------
    # Probability validation
    # -------------------------------------------------------------------------

    if not pd.api.types.is_numeric_dtype(
        df[probability_column]
    ):

        raise ValueError(
            f"MODEL {model_number} {dataset_name.upper()} "
            f"PROBABILITY COLUMN IS NOT NUMERIC."
        )

    probabilities = df[probability_column]

    invalid_probabilities = (
        (probabilities < 0)
        | (probabilities > 1)
    ).sum()

    if invalid_probabilities > 0:

        raise ValueError(
            f"MODEL {model_number} {dataset_name.upper()} "
            f"CONTAINS {invalid_probabilities} "
            f"PROBABILITIES OUTSIDE [0, 1]."
        )

    # -------------------------------------------------------------------------
    # Basic information
    # -------------------------------------------------------------------------

    print(
        f"Model {model_number:>2} | "
        f"{dataset_name.capitalize():>10} | "
        f"Rows: {len(df):>5} | "
        f"Target: {actual_column} | "
        f"Probability: {probability_column}"
    )

    return cols


def validate_common_dataset(
    prediction_data,
    dataset_name,
):
    """
    Verify that all models use exactly the same games and actual outcomes.

    Model 1 is used as the reference dataset.
    """

    reference_model = MODEL_NUMBERS[0]

    reference_df = prediction_data[
        reference_model
    ][dataset_name]

    reference_cols = identify_prediction_columns(
        reference_df
    )

    reference_actual = (
        reference_df[
            [
                reference_cols["game_id"],
                reference_cols["actual"],
            ]
        ]
        .copy()
        .sort_values(reference_cols["game_id"])
        .reset_index(drop=True)
    )

    for model_number in MODEL_NUMBERS[1:]:

        df = prediction_data[
            model_number
        ][dataset_name]

        cols = identify_prediction_columns(df)

        # ---------------------------------------------------------------------
        # Check row counts
        # ---------------------------------------------------------------------

        if len(df) != len(reference_df):

            raise ValueError(
                f"{dataset_name.upper()} ROW COUNT MISMATCH\n"
                f"Model 1: {len(reference_df)} rows\n"
                f"Model {model_number}: {len(df)} rows"
            )

        # ---------------------------------------------------------------------
        # Check game IDs
        # ---------------------------------------------------------------------

        reference_ids = set(
            reference_df[reference_cols["game_id"]]
        )

        model_ids = set(
            df[cols["game_id"]]
        )

        if reference_ids != model_ids:

            missing_from_model = (
                reference_ids - model_ids
            )

            extra_in_model = (
                model_ids - reference_ids
            )

            raise ValueError(
                f"{dataset_name.upper()} GAME ID MISMATCH\n"
                f"Model 1 vs Model {model_number}\n"
                f"Missing from Model {model_number}: "
                f"{len(missing_from_model)}\n"
                f"Extra in Model {model_number}: "
                f"{len(extra_in_model)}"
            )

        # ---------------------------------------------------------------------
        # Check actual outcomes
        # ---------------------------------------------------------------------

        comparison = (
            df[
                [
                    cols["game_id"],
                    cols["actual"],
                ]
            ]
            .copy()
            .sort_values(cols["game_id"])
            .reset_index(drop=True)
        )

        comparison.columns = [
            reference_cols["game_id"],
            reference_cols["actual"],
        ]

        if not reference_actual.equals(comparison):

            merged = reference_actual.merge(
                comparison,
                on=reference_cols["game_id"],
                suffixes=(
                    "_reference",
                    "_model",
                ),
            )

            reference_actual_column = (
                f"{reference_cols['actual']}_reference"
            )

            model_actual_column = (
                f"{reference_cols['actual']}_model"
            )

            mismatches = merged[
                merged[reference_actual_column]
                != merged[model_actual_column]
            ]

            raise ValueError(
                f"{dataset_name.upper()} ACTUAL TARGET MISMATCH\n"
                f"Model 1 vs Model {model_number}\n"
                f"Mismatched games: {len(mismatches)}"
            )

    print(
        f"{dataset_name.capitalize()} dataset consistency check: PASSED"
    )


# =============================================================================
# CALIBRATION
# =============================================================================

def calculate_calibration_metrics(
    y_true,
    y_probability,
    n_bins=10,
):
    """
    Calculate Expected Calibration Error (ECE) and Maximum Calibration
    Error (MCE).

    Uses equal-width probability bins from 0 to 1.
    """

    y_true = np.asarray(y_true)
    y_probability = np.asarray(y_probability)

    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    ece = 0.0
    mce = 0.0

    for i in range(n_bins):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == n_bins - 1:

            mask = (
                (y_probability >= lower)
                & (y_probability <= upper)
            )

        else:

            mask = (
                (y_probability >= lower)
                & (y_probability < upper)
            )

        if not np.any(mask):
            continue

        bin_probability = y_probability[mask].mean()
        bin_accuracy = y_true[mask].mean()

        calibration_error = abs(
            bin_accuracy - bin_probability
        )

        bin_weight = mask.mean()

        ece += (
            bin_weight
            * calibration_error
        )

        mce = max(
            mce,
            calibration_error,
        )

    return ece, mce


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(
    y_true,
    y_probability,
):
    """
    Calculate all standardized model-comparison metrics.

    Classification predictions use a fixed 0.50 probability threshold.
    """

    y_true = np.asarray(y_true)
    y_probability = np.asarray(y_probability)

    # -------------------------------------------------------------------------
    # Classification prediction
    # -------------------------------------------------------------------------

    y_prediction = (
        y_probability >= 0.50
    ).astype(int)

    # -------------------------------------------------------------------------
    # Probability metrics
    # -------------------------------------------------------------------------

    log_loss_value = log_loss(
        y_true,
        y_probability,
    )

    brier_value = brier_score_loss(
        y_true,
        y_probability,
    )

    roc_auc_value = roc_auc_score(
        y_true,
        y_probability,
    )

    # -------------------------------------------------------------------------
    # Classification metrics
    # -------------------------------------------------------------------------

    accuracy_value = accuracy_score(
        y_true,
        y_prediction,
    )

    balanced_accuracy_value = (
        balanced_accuracy_score(
            y_true,
            y_prediction,
        )
    )

    precision_value = precision_score(
        y_true,
        y_prediction,
        zero_division=0,
    )

    recall_value = recall_score(
        y_true,
        y_prediction,
        zero_division=0,
    )

    # -------------------------------------------------------------------------
    # Calibration
    # -------------------------------------------------------------------------

    ece_value, mce_value = (
        calculate_calibration_metrics(
            y_true,
            y_probability,
        )
    )

    return {
        "log_loss": log_loss_value,
        "brier_score": brier_value,
        "roc_auc": roc_auc_value,
        "accuracy": accuracy_value,
        "balanced_accuracy": balanced_accuracy_value,
        "precision": precision_value,
        "recall": recall_value,
        "ece": ece_value,
        "mce": mce_value,
    }


# =============================================================================
# FEATURE COUNT
# =============================================================================

def load_feature_count(model_number):
    """
    Load feature count from feature_list.csv.

    Returns
    -------
    int
        Number of features listed in feature_list.csv.
    """

    feature_file = (
        MODEL_ROOT
        / f"model_{model_number}"
        / "feature_list.csv"
    )

    if not feature_file.exists():

        raise FileNotFoundError(
            f"Feature list not found:\n{feature_file}"
        )

    feature_df = pd.read_csv(
        feature_file
    )

    return len(feature_df)


# =============================================================================
# LOAD MODEL PREDICTIONS
# =============================================================================

def load_model_predictions():
    """
    Load validation and test predictions for Models 1-5.

    Returns
    -------
    dict
        Nested dictionary:

        {
            model_number: {
                "validation": dataframe,
                "test": dataframe
            }
        }
    """

    print_header(
        "LOADING RANDOM FOREST MODEL PREDICTIONS"
    )

    prediction_data = {}

    for model_number in MODEL_NUMBERS:

        model_directory = (
            MODEL_ROOT
            / f"model_{model_number}"
        )

        prediction_data[
            model_number
        ] = {}

        print_subheader(
            f"MODEL {model_number}"
        )

        for dataset_name in DATASETS:

            prediction_file = (
                model_directory
                / f"{dataset_name}_predictions.csv"
            )

            if not prediction_file.exists():

                raise FileNotFoundError(
                    f"Prediction file not found:\n"
                    f"{prediction_file}"
                )

            df = pd.read_csv(
                prediction_file
            )

            validate_prediction_dataframe(
                df,
                model_number,
                dataset_name,
            )

            prediction_data[
                model_number
            ][dataset_name] = df

    return prediction_data


# =============================================================================
# RANKING
# =============================================================================

def rank_metric(
    results,
    metric,
    higher_is_better,
):
    """
    Rank models for a single metric.
    """

    values = results[metric]

    if higher_is_better:

        return values.rank(
            ascending=False,
            method="min",
        ).astype(int)

    return values.rank(
        ascending=True,
        method="min",
    ).astype(int)


def create_rankings(results):
    """
    Create rankings for each metric.

    Lower is better:
        log_loss
        brier_score
        ece
        mce

    Higher is better:
        roc_auc
        accuracy
        balanced_accuracy
        precision
        recall
    """

    higher_is_better = {
        "log_loss": False,
        "brier_score": False,
        "roc_auc": True,
        "accuracy": True,
        "balanced_accuracy": True,
        "precision": True,
        "recall": True,
        "ece": False,
        "mce": False,
    }

    for metric, direction in higher_is_better.items():

        results[
            f"{metric}_rank"
        ] = rank_metric(
            results,
            metric,
            direction,
        )

    ranking_columns = [
        f"{metric}_rank"
        for metric in higher_is_better
    ]

    results["average_rank"] = (
        results[ranking_columns]
        .mean(axis=1)
    )

    return results


# =============================================================================
# MODEL AGREEMENT
# =============================================================================

def calculate_model_agreement(
    prediction_data,
):
    """
    Calculate pairwise agreement between models.

    Metrics:
        - mean absolute probability difference
        - median absolute probability difference
        - maximum absolute probability difference
        - probability correlation
        - classification agreement
    """

    rows = []

    for dataset_name in DATASETS:

        for model_a, model_b in combinations(
            MODEL_NUMBERS,
            2,
        ):

            df_a = prediction_data[
                model_a
            ][dataset_name]

            df_b = prediction_data[
                model_b
            ][dataset_name]

            cols_a = identify_prediction_columns(
                df_a
            )

            cols_b = identify_prediction_columns(
                df_b
            )

            # -----------------------------------------------------------------
            # Select and standardize the columns BEFORE merging.
            #
            # This is important because Model 1 uses:
            #   predicted_home_win_probability
            #
            # while Models 2-5 use:
            #   win_home_probability
            # -----------------------------------------------------------------

            model_a_data = df_a[
                [
                    cols_a["game_id"],
                    cols_a["probability"],
                ]
            ].copy()

            model_b_data = df_b[
                [
                    cols_b["game_id"],
                    cols_b["probability"],
                ]
            ].copy()

            model_a_data.columns = [
                "game_id",
                "probability_a",
            ]

            model_b_data.columns = [
                "game_id",
                "probability_b",
            ]

            # -----------------------------------------------------------------
            # Merge on game ID
            # -----------------------------------------------------------------

            merged = model_a_data.merge(
                model_b_data,
                on="game_id",
                how="inner",
                validate="one_to_one",
            )

            # -----------------------------------------------------------------
            # Verify all games are present
            # -----------------------------------------------------------------

            expected_games = len(df_a)

            if len(merged) != expected_games:

                raise ValueError(
                    f"{dataset_name.upper()} MODEL AGREEMENT "
                    f"GAME COUNT MISMATCH\n"
                    f"Model {model_a} vs Model {model_b}\n"
                    f"Expected: {expected_games}\n"
                    f"Merged: {len(merged)}"
                )

            # -----------------------------------------------------------------
            # Extract probabilities
            # -----------------------------------------------------------------

            probability_a = merged[
                "probability_a"
            ]

            probability_b = merged[
                "probability_b"
            ]

            # -----------------------------------------------------------------
            # Classification predictions
            #
            # Use the same 0.50 threshold for every model.
            # -----------------------------------------------------------------

            prediction_a = (
                probability_a >= 0.50
            ).astype(int)

            prediction_b = (
                probability_b >= 0.50
            ).astype(int)

            # -----------------------------------------------------------------
            # Probability differences
            # -----------------------------------------------------------------

            absolute_difference = (
                probability_a
                - probability_b
            ).abs()

            # -----------------------------------------------------------------
            # Probability correlation
            # -----------------------------------------------------------------

            correlation = (
                probability_a.corr(
                    probability_b
                )
            )

            # -----------------------------------------------------------------
            # Classification agreement
            # -----------------------------------------------------------------

            classification_agreement = (
                prediction_a
                == prediction_b
            ).mean()

            # -----------------------------------------------------------------
            # Save results
            # -----------------------------------------------------------------

            rows.append(
                {
                    "dataset": dataset_name,
                    "model_a": model_a,
                    "model_b": model_b,
                    "games": len(merged),
                    "mean_abs_probability_difference":
                        absolute_difference.mean(),
                    "median_abs_probability_difference":
                        absolute_difference.median(),
                    "max_abs_probability_difference":
                        absolute_difference.max(),
                    "probability_correlation":
                        correlation,
                    "classification_agreement":
                        classification_agreement,
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# PRINT RESULTS
# =============================================================================

def print_metric_table(
    results,
    dataset_name,
):
    """Print standardized metric comparison."""

    dataset_results = results[
        results["dataset"]
        == dataset_name
    ].copy()

    display_columns = [
        "model",
        "features",
        "log_loss",
        "brier_score",
        "roc_auc",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "ece",
        "mce",
    ]

    table = dataset_results[
        display_columns
    ].copy()

    print(
        table.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )


def print_rankings(
    results,
    dataset_name,
):
    """Print model rankings."""

    dataset_results = results[
        results["dataset"]
        == dataset_name
    ].copy()

    ranking_columns = [
        "model",
        "log_loss_rank",
        "brier_score_rank",
        "roc_auc_rank",
        "accuracy_rank",
        "balanced_accuracy_rank",
        "precision_rank",
        "recall_rank",
        "ece_rank",
        "mce_rank",
        "average_rank",
    ]

    print(
        dataset_results[
            ranking_columns
        ]
        .sort_values(
            "average_rank"
        )
        .to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )


def print_best_models(
    results,
    dataset_name,
):
    """
    Print the best model for each major evaluation metric.
    """

    dataset_results = results[
        results["dataset"]
        == dataset_name
    ].copy()

    print_subheader(
        f"BEST MODELS — {dataset_name.upper()}"
    )

    lower_is_better = [
        "log_loss",
        "brier_score",
        "ece",
        "mce",
    ]

    higher_is_better = [
        "roc_auc",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
    ]

    for metric in lower_is_better:

        best_index = (
            dataset_results[metric]
            .idxmin()
        )

        best_model = (
            dataset_results
            .loc[best_index, "model"]
        )

        best_value = (
            dataset_results
            .loc[best_index, metric]
        )

        print(
            f"{metric:<22} "
            f"Model {int(best_model)} "
            f"({best_value:.6f})"
        )

    for metric in higher_is_better:

        best_index = (
            dataset_results[metric]
            .idxmax()
        )

        best_model = (
            dataset_results
            .loc[best_index, "model"]
        )

        best_value = (
            dataset_results
            .loc[best_index, metric]
        )

        print(
            f"{metric:<22} "
            f"Model {int(best_model)} "
            f"({best_value:.6f})"
        )


# =============================================================================
# BUILD RESULTS
# =============================================================================

def build_results(
    prediction_data,
):
    """
    Calculate metrics for every model and dataset.
    """

    rows = []

    for model_number in MODEL_NUMBERS:

        feature_count = (
            load_feature_count(
                model_number
            )
        )

        for dataset_name in DATASETS:

            df = prediction_data[
                model_number
            ][dataset_name]

            cols = identify_prediction_columns(
                df
            )

            y_true = df[
                cols["actual"]
            ]

            y_probability = df[
                cols["probability"]
            ]

            metrics = calculate_metrics(
                y_true,
                y_probability,
            )

            row = {
                "dataset": dataset_name,
                "model": model_number,
                "features": feature_count,
                "rows": len(df),
            }

            row.update(metrics)

            rows.append(row)

    results = pd.DataFrame(rows)

    return results


# =============================================================================
# CHAMPION SELECTION
# =============================================================================

def select_champion(results):
    """
    Select the overall champion.

    Primary criterion:
        Test log loss

    Supporting information:
        Test Brier score
        Test ROC AUC
        Test ECE
        Test MCE
    """

    test_results = results[
        results["dataset"] == "test"
    ].copy()

    champion_index = (
        test_results["log_loss"]
        .idxmin()
    )

    champion = (
        test_results
        .loc[champion_index]
    )

    return champion


def print_champion(results):
    """Print the overall champion model."""

    champion = select_champion(
        results
    )

    model_number = int(
        champion["model"]
    )

    print_header(
        "OVERALL CHAMPION MODEL"
    )

    print(
        f"Champion: Model {model_number}"
    )

    print(
        f"Features: {int(champion['features'])}"
    )

    print(
        f"Test Log Loss: "
        f"{champion['log_loss']:.6f}"
    )

    print(
        f"Test Brier Score: "
        f"{champion['brier_score']:.6f}"
    )

    print(
        f"Test ROC AUC: "
        f"{champion['roc_auc']:.6f}"
    )

    print(
        f"Test ECE: "
        f"{champion['ece']:.6f}"
    )

    print(
        f"Test MCE: "
        f"{champion['mce']:.6f}"
    )

    print()
    print(
        "Champion selection is based primarily "
        "on test log loss because this project "
        "is focused on probability estimation."
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print_header(
        "RANDOM FOREST WIN PROBABILITY"
    )

    print(
        "MODEL 1 VS MODEL 2 VS MODEL 3 VS MODEL 4 VS MODEL 5"
    )

    print()
    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Model root:   {MODEL_ROOT}"
    )

    # -------------------------------------------------------------------------
    # Load predictions
    # -------------------------------------------------------------------------

    prediction_data = (
        load_model_predictions()
    )

    # -------------------------------------------------------------------------
    # Validate common datasets
    # -------------------------------------------------------------------------

    print_header(
        "DATASET CONSISTENCY CHECK"
    )

    for dataset_name in DATASETS:

        validate_common_dataset(
            prediction_data,
            dataset_name,
        )

    # -------------------------------------------------------------------------
    # Build metrics
    # -------------------------------------------------------------------------

    print_header(
        "CALCULATING MODEL METRICS"
    )

    results = build_results(
        prediction_data
    )

    # -------------------------------------------------------------------------
    # Rankings
    # -------------------------------------------------------------------------

    results = create_rankings(
        results
    )

    # -------------------------------------------------------------------------
    # Validation results
    # -------------------------------------------------------------------------

    print_header(
        "VALIDATION RESULTS"
    )

    print_metric_table(
        results,
        "validation",
    )

    print_rankings(
        results,
        "validation",
    )

    print_best_models(
        results,
        "validation",
    )

    # -------------------------------------------------------------------------
    # Test results
    # -------------------------------------------------------------------------

    print_header(
        "TEST RESULTS"
    )

    print_metric_table(
        results,
        "test",
    )

    print_rankings(
        results,
        "test",
    )

    print_best_models(
        results,
        "test",
    )

    # -------------------------------------------------------------------------
    # Model agreement
    # -------------------------------------------------------------------------

    print_header(
        "PAIRWISE MODEL AGREEMENT"
    )

    agreement = calculate_model_agreement(
        prediction_data
    )

    for dataset_name in DATASETS:

        print_subheader(
            dataset_name.upper()
        )

        dataset_agreement = agreement[
            agreement["dataset"]
            == dataset_name
        ]

        print(
            dataset_agreement.to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

    # -------------------------------------------------------------------------
    # Champion
    # -------------------------------------------------------------------------

    print_champion(
        results
    )

    # -------------------------------------------------------------------------
    # Save comparison results
    # -------------------------------------------------------------------------

    results.to_csv(
        OUTPUT_COMPARISON,
        index=False,
    )

    agreement.to_csv(
        OUTPUT_AGREEMENT,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Final output
    # -------------------------------------------------------------------------

    print_header(
        "FILES SAVED"
    )

    print(
        f"Model comparison:\n"
        f"{OUTPUT_COMPARISON}"
    )

    print()
    print(
        f"Model agreement:\n"
        f"{OUTPUT_AGREEMENT}"
    )

    print_header(
        "COMPARISON COMPLETE"
    )


if __name__ == "__main__":
    main()