"""
This audit script is being used to compare
model 4 and model 5. These models both use
the same feature space, but use different
hyperparameter tuning methods, leading to
slightly differing models.
"""

from pathlib import Path
import os

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
# CONFIGURATION
# =============================================================================

MODEL_4_DIR = Path(
    "models/win_probability/random_forest/model_4"
)

MODEL_5_DIR = Path(
    "models/win_probability/random_forest/model_5"
)

OUTPUT_DIR = MODEL_5_DIR / "audit"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RANDOM_STATE = 42

PROBABILITY_BINS = np.linspace(
    0.0,
    1.0,
    11,
)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def print_header(title):
    """Print a formatted section header."""
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def calculate_metrics(y_true, probabilities, threshold=0.5):
    """Calculate classification and probability metrics."""

    probabilities = np.asarray(probabilities)
    y_true = np.asarray(y_true)

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "log_loss": log_loss(
            y_true,
            probabilities,
            labels=[0, 1],
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


def calculate_calibration(
    y_true,
    probabilities,
    n_bins=10,
):
    """Calculate calibration table, ECE, and MCE."""

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    bins = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    rows = []

    ece = 0.0
    mce = 0.0

    for i in range(n_bins):

        lower = bins[i]
        upper = bins[i + 1]

        if i == n_bins - 1:

            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )

        else:

            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        count = mask.sum()

        if count == 0:

            rows.append(
                {
                    "bin": i + 1,
                    "bin_lower": lower,
                    "bin_upper": upper,
                    "count": 0,
                    "mean_predicted_probability": np.nan,
                    "observed_win_rate": np.nan,
                    "calibration_error": np.nan,
                    "absolute_calibration_error": np.nan,
                }
            )

            continue

        mean_probability = (
            probabilities[mask].mean()
        )

        observed_rate = (
            y_true[mask].mean()
        )

        calibration_error = (
            observed_rate
            - mean_probability
        )

        absolute_error = abs(
            calibration_error
        )

        weight = (
            count / len(y_true)
        )

        ece += (
            weight
            * absolute_error
        )

        mce = max(
            mce,
            absolute_error,
        )

        rows.append(
            {
                "bin": i + 1,
                "bin_lower": lower,
                "bin_upper": upper,
                "count": count,
                "mean_predicted_probability": mean_probability,
                "observed_win_rate": observed_rate,
                "calibration_error": calibration_error,
                "absolute_calibration_error": absolute_error,
            }
        )

    calibration_table = pd.DataFrame(
        rows
    )

    return (
        calibration_table,
        ece,
        mce,
    )


def load_predictions(model_dir):
    """Load validation and test prediction files."""

    validation_path = (
        model_dir
        / "validation_predictions.csv"
    )

    test_path = (
        model_dir
        / "test_predictions.csv"
    )

    if not validation_path.exists():

        raise FileNotFoundError(
            "Missing validation predictions:\n"
            f"{validation_path}"
        )

    if not test_path.exists():

        raise FileNotFoundError(
            "Missing test predictions:\n"
            f"{test_path}"
        )

    validation = pd.read_csv(
        validation_path
    )

    test = pd.read_csv(
        test_path
    )

    return (
        validation,
        test,
    )


def validate_prediction_schema(
    df,
    model_name,
    split,
):
    """Validate expected prediction columns."""

    required_columns = {
        "gameId",
        "season",
        "win_home_actual",
        "win_home_probability",
        "win_home_prediction",
        "split",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"{model_name} {split} predictions "
            f"are missing columns: "
            f"{sorted(missing)}"
        )

    if df["gameId"].duplicated().any():

        raise ValueError(
            f"{model_name} {split} predictions "
            "contain duplicate gameId values."
        )

    if (
        df["win_home_probability"]
        .isna()
        .any()
    ):

        raise ValueError(
            f"{model_name} {split} predictions "
            "contain missing probabilities."
        )

    if (
        (
            df["win_home_probability"]
            < 0
        ).any()
        or
        (
            df["win_home_probability"]
            > 1
        ).any()
    ):

        raise ValueError(
            f"{model_name} {split} probabilities "
            "contain values outside [0, 1]."
        )

    if not set(
        df["win_home_actual"].unique()
    ).issubset({0, 1}):

        raise ValueError(
            f"{model_name} {split} actual outcomes "
            "are not binary."
        )


def compare_prediction_alignment(
    model_4_df,
    model_5_df,
    split,
):
    """Compare predictions after aligning on gameId."""

    merged = model_4_df[
        [
            "gameId",
            "season",
            "win_home_actual",
            "win_home_probability",
            "win_home_prediction",
        ]
    ].merge(
        model_5_df[
            [
                "gameId",
                "season",
                "win_home_actual",
                "win_home_probability",
                "win_home_prediction",
            ]
        ],
        on="gameId",
        how="outer",
        suffixes=(
            "_model4",
            "_model5",
        ),
        indicator=True,
    )

    if not (
        merged["_merge"]
        == "both"
    ).all():

        missing_4 = merged.loc[
            merged["_merge"]
            == "right_only",
            "gameId",
        ].tolist()

        missing_5 = merged.loc[
            merged["_merge"]
            == "left_only",
            "gameId",
        ].tolist()

        raise ValueError(
            f"{split} prediction alignment mismatch.\n"
            f"Missing from Model 4: "
            f"{missing_4[:10]}\n"
            f"Missing from Model 5: "
            f"{missing_5[:10]}"
        )

    merged = merged.loc[
        merged["_merge"]
        == "both"
    ].copy()

    if not (
        merged[
            "win_home_actual_model4"
        ]
        ==
        merged[
            "win_home_actual_model5"
        ]
    ).all():

        raise ValueError(
            f"{split} actual outcomes differ "
            "between Model 4 and Model 5."
        )

    if not (
        merged["season_model4"]
        ==
        merged["season_model5"]
    ).all():

        raise ValueError(
            f"{split} season assignments differ "
            "between Model 4 and Model 5."
        )

    merged[
        "probability_difference"
    ] = (
        merged[
            "win_home_probability_model5"
        ]
        -
        merged[
            "win_home_probability_model4"
        ]
    )

    merged[
        "absolute_probability_difference"
    ] = (
        merged[
            "probability_difference"
        ].abs()
    )

    merged[
        "prediction_agreement"
    ] = (
        merged[
            "win_home_prediction_model4"
        ]
        ==
        merged[
            "win_home_prediction_model5"
        ]
    )

    merged["probability_rank"] = (
        merged[
            "probability_difference"
        ]
        .abs()
        .rank(
            ascending=False,
            method="first",
        )
    )

    return merged


def calculate_season_metrics(df):
    """Calculate metrics separately for each season."""

    rows = []

    for season in sorted(
        df["season"].unique()
    ):

        season_df = df.loc[
            df["season"] == season
        ].copy()

        metrics = calculate_metrics(
            season_df[
                "win_home_actual"
            ],
            season_df[
                "win_home_probability"
            ],
        )

        rows.append(
            {
                "season": season,
                "games": len(
                    season_df
                ),
                **metrics,
            }
        )

    return pd.DataFrame(
        rows
    )


def load_feature_list(model_dir):
    """Load feature list."""

    path = (
        model_dir
        / "feature_list.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing feature list:\n{path}"
        )

    return pd.read_csv(
        path
    )


def load_best_params(model_dir):
    path = os.path.join(model_dir, "best_params.csv")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find best_params.csv at: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"best_params.csv is empty: {path}")

    return df


def find_feature_column(df):
    """Identify feature-name column."""

    candidates = [
        "feature",
        "feature_name",
        "Feature",
        "Feature Name",
    ]

    for column in candidates:

        if column in df.columns:

            return column

    raise ValueError(
        "Could not identify feature-name "
        "column in feature importance file. "
        f"Available columns: {list(df.columns)}"
    )


def find_importance_column(df):
    """Identify importance column."""

    candidates = [
        "importance",
        "feature_importance",
        "mean_importance",
        "Importance",
    ]

    for column in candidates:

        if column in df.columns:

            return column

    raise ValueError(
        "Could not identify importance "
        "column in feature importance file. "
        f"Available columns: {list(df.columns)}"
    )


def load_feature_importance(
    model_dir,
    model_name,
):
    """
    Load feature importance.

    If feature_importance.csv exists, use it.

    Otherwise, derive feature importance directly
    from model.joblib and feature_list.csv.
    """

    importance_path = (
        model_dir
        / "feature_importance.csv"
    )

    if importance_path.exists():

        print(
            f"{model_name}: loading feature "
            f"importance from CSV."
        )

        return pd.read_csv(
            importance_path
        )

    print(
        f"{model_name}: feature_importance.csv "
        "not found."
    )

    print(
        f"{model_name}: deriving feature "
        "importance from model.joblib."
    )

    model_path = (
        model_dir
        / "model.joblib"
    )

    feature_list_path = (
        model_dir
        / "feature_list.csv"
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"{model_name}: missing model:\n"
            f"{model_path}"
        )

    if not feature_list_path.exists():

        raise FileNotFoundError(
            f"{model_name}: missing feature list:\n"
            f"{feature_list_path}"
        )

    model = joblib.load(
        model_path
    )

    feature_list = pd.read_csv(
        feature_list_path
    )

    feature_column_candidates = [
        "feature",
        "feature_name",
        "Feature",
        "Feature Name",
    ]

    feature_column = None

    for column in (
        feature_column_candidates
    ):

        if column in feature_list.columns:

            feature_column = column
            break

    if feature_column is None:

        feature_column = (
            feature_list.columns[0]
        )

    feature_names = (
        feature_list[
            feature_column
        ]
        .astype(str)
        .tolist()
    )

    # The saved model is expected to be a
    # sklearn Pipeline containing the imputer
    # followed by the Random Forest.
    if hasattr(
        model,
        "named_steps",
    ):

        if "model" in model.named_steps:

            estimator = (
                model.named_steps["model"]
            )

        else:

            estimator = (
                model.steps[-1][1]
            )

    else:

        estimator = model

    if not hasattr(
        estimator,
        "feature_importances_",
    ):

        raise AttributeError(
            f"{model_name}: could not locate "
            "feature_importances_ in saved model."
        )

    importances = (
        estimator.feature_importances_
    )

    if len(feature_names) != len(
        importances
    ):

        raise ValueError(
            f"{model_name}: feature count mismatch.\n"
            f"Feature list: {len(feature_names)}\n"
            f"Model importances: {len(importances)}"
        )

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    importance = (
        importance
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    # Persist the derived artifact so future
    # audits can load it directly.
    importance.to_csv(
        importance_path,
        index=False,
    )

    print(
        f"{model_name}: derived feature importance "
        f"saved to:\n{importance_path}"
    )

    return importance


# =============================================================================
# LOAD DATA
# =============================================================================

print_header(
    "RANDOM FOREST WIN PROBABILITY - "
    "MODEL 4 VS MODEL 5 AUDIT"
)

print(
    "Comparing conventional CV hyperparameter "
    "tuning against temporal CV hyperparameter tuning."
)

print()
print(
    "Loading prediction files..."
)

(
    model_4_validation,
    model_4_test,
) = load_predictions(
    MODEL_4_DIR
)

(
    model_5_validation,
    model_5_test,
) = load_predictions(
    MODEL_5_DIR
)

validate_prediction_schema(
    model_4_validation,
    "Model 4",
    "validation",
)

validate_prediction_schema(
    model_4_test,
    "Model 4",
    "test",
)

validate_prediction_schema(
    model_5_validation,
    "Model 5",
    "validation",
)

validate_prediction_schema(
    model_5_test,
    "Model 5",
    "test",
)

print(
    "Prediction files loaded successfully."
)


# =============================================================================
# DATASET SIZE CHECK
# =============================================================================

print_header(
    "PREDICTION DATASET CHECK"
)

print(
    f"Model 4 validation rows: "
    f"{len(model_4_validation)}"
)

print(
    f"Model 5 validation rows: "
    f"{len(model_5_validation)}"
)

print(
    f"Model 4 test rows:       "
    f"{len(model_4_test)}"
)

print(
    f"Model 5 test rows:       "
    f"{len(model_5_test)}"
)

if (
    len(model_4_validation)
    != len(model_5_validation)
):

    raise ValueError(
        "Model 4 and Model 5 validation "
        "row counts differ."
    )

if (
    len(model_4_test)
    != len(model_5_test)
):

    raise ValueError(
        "Model 4 and Model 5 test "
        "row counts differ."
    )


# =============================================================================
# FEATURE COMPARISON
# =============================================================================

print_header(
    "FEATURE CONFIGURATION"
)

model_4_features = load_feature_list(
    MODEL_4_DIR
)

model_5_features = load_feature_list(
    MODEL_5_DIR
)

model_4_feature_column = (
    model_4_features.columns[0]
)

model_5_feature_column = (
    model_5_features.columns[0]
)

features_4 = set(
    model_4_features[
        model_4_feature_column
    ]
)

features_5 = set(
    model_5_features[
        model_5_feature_column
    ]
)

only_model_4 = sorted(
    features_4 - features_5
)

only_model_5 = sorted(
    features_5 - features_4
)

shared_features = sorted(
    features_4 & features_5
)

print(
    f"Model 4 features: {len(features_4)}"
)

print(
    f"Model 5 features: {len(features_5)}"
)

print(
    f"Shared features:   {len(shared_features)}"
)

print(
    f"Model 4 only:      {len(only_model_4)}"
)

print(
    f"Model 5 only:      {len(only_model_5)}"
)

if only_model_4:

    print(
        "\nFeatures only in Model 4:"
    )

    for feature in only_model_4:

        print(
            f"  - {feature}"
        )

if only_model_5:

    print(
        "\nFeatures only in Model 5:"
    )

    for feature in only_model_5:

        print(
            f"  - {feature}"
        )

if features_4 != features_5:

    raise ValueError(
        "Model 4 and Model 5 do not use "
        "identical feature sets."
    )

feature_comparison = pd.DataFrame(
    {
        "feature_count_model_4": [
            len(features_4)
        ],
        "feature_count_model_5": [
            len(features_5)
        ],
        "shared_feature_count": [
            len(shared_features)
        ],
        "model_4_only_count": [
            len(only_model_4)
        ],
        "model_5_only_count": [
            len(only_model_5)
        ],
        "feature_sets_identical": [
            features_4 == features_5
        ],
    }
)

feature_comparison.to_csv(
    OUTPUT_DIR
    / "feature_comparison.csv",
    index=False,
)


# =============================================================================
# HYPERPARAMETER COMPARISON
# =============================================================================

print_header(
    "HYPERPARAMETER COMPARISON"
)

model_4_params = load_best_params(
    MODEL_4_DIR
)

model_5_params = load_best_params(
    MODEL_5_DIR
)

params_4 = dict(
    zip(
        model_4_params.iloc[:, 0],
        model_4_params.iloc[:, 1],
    )
)

params_5 = dict(
    zip(
        model_5_params.iloc[:, 0],
        model_5_params.iloc[:, 1],
    )
)

# Normalize parameter names to strings.
params_4 = {
    str(key): value
    for key, value in params_4.items()
}

params_5 = {
    str(key): value
    for key, value in params_5.items()
}

all_parameter_names = sorted(
    set(params_4)
    | set(params_5)
)

parameter_rows = []

for parameter in all_parameter_names:

    value_4 = params_4.get(
        parameter
    )

    value_5 = params_5.get(
        parameter
    )

    parameter_rows.append(
        {
            "parameter": parameter,
            "model_4": value_4,
            "model_5": value_5,
            "same": (
                str(value_4)
                ==
                str(value_5)
            ),
        }
    )

    print(
        f"{parameter}: "
        f"Model 4 = {value_4} | "
        f"Model 5 = {value_5}"
    )

parameter_comparison = pd.DataFrame(
    parameter_rows
)

parameter_comparison.to_csv(
    OUTPUT_DIR
    / "hyperparameter_comparison.csv",
    index=False,
)


# =============================================================================
# OFFICIAL METRICS
# =============================================================================

print_header(
    "OFFICIAL VALIDATION METRICS"
)

model_4_validation_metrics = (
    calculate_metrics(
        model_4_validation[
            "win_home_actual"
        ],
        model_4_validation[
            "win_home_probability"
        ],
    )
)

model_5_validation_metrics = (
    calculate_metrics(
        model_5_validation[
            "win_home_actual"
        ],
        model_5_validation[
            "win_home_probability"
        ],
    )
)

validation_metric_rows = []

for metric in (
    model_4_validation_metrics
):

    value_4 = (
        model_4_validation_metrics[
            metric
        ]
    )

    value_5 = (
        model_5_validation_metrics[
            metric
        ]
    )

    validation_metric_rows.append(
        {
            "metric": metric,
            "model_4": value_4,
            "model_5": value_5,
            "model_5_minus_model_4": (
                value_5 - value_4
            ),
        }
    )

    print(
        f"{metric}: "
        f"Model 4 = {value_4:.6f} | "
        f"Model 5 = {value_5:.6f} | "
        f"Difference = "
        f"{value_5 - value_4:+.6f}"
    )

validation_metrics = pd.DataFrame(
    validation_metric_rows
)

validation_metrics.to_csv(
    OUTPUT_DIR
    / "validation_metrics.csv",
    index=False,
)


# =============================================================================
# TEST METRICS
# =============================================================================

print_header(
    "OFFICIAL TEST METRICS"
)

model_4_test_metrics = (
    calculate_metrics(
        model_4_test[
            "win_home_actual"
        ],
        model_4_test[
            "win_home_probability"
        ],
    )
)

model_5_test_metrics = (
    calculate_metrics(
        model_5_test[
            "win_home_actual"
        ],
        model_5_test[
            "win_home_probability"
        ],
    )
)

test_metric_rows = []

for metric in (
    model_4_test_metrics
):

    value_4 = (
        model_4_test_metrics[
            metric
        ]
    )

    value_5 = (
        model_5_test_metrics[
            metric
        ]
    )

    test_metric_rows.append(
        {
            "metric": metric,
            "model_4": value_4,
            "model_5": value_5,
            "model_5_minus_model_4": (
                value_5 - value_4
            ),
        }
    )

    print(
        f"{metric}: "
        f"Model 4 = {value_4:.6f} | "
        f"Model 5 = {value_5:.6f} | "
        f"Difference = "
        f"{value_5 - value_4:+.6f}"
    )

test_metrics = pd.DataFrame(
    test_metric_rows
)

test_metrics.to_csv(
    OUTPUT_DIR
    / "test_metrics.csv",
    index=False,
)


# =============================================================================
# CALIBRATION
# =============================================================================

print_header(
    "CALIBRATION ANALYSIS"
)

calibration_summary_rows = []

for (
    split_name,
    model_4_df,
    model_5_df,
) in [
    (
        "validation",
        model_4_validation,
        model_5_validation,
    ),
    (
        "test",
        model_4_test,
        model_5_test,
    ),
]:

    (
        cal_4,
        ece_4,
        mce_4,
    ) = calculate_calibration(
        model_4_df[
            "win_home_actual"
        ],
        model_4_df[
            "win_home_probability"
        ],
    )

    (
        cal_5,
        ece_5,
        mce_5,
    ) = calculate_calibration(
        model_5_df[
            "win_home_actual"
        ],
        model_5_df[
            "win_home_probability"
        ],
    )

    cal_4.insert(
        0,
        "model",
        "model_4",
    )

    cal_4.insert(
        1,
        "split",
        split_name,
    )

    cal_5.insert(
        0,
        "model",
        "model_5",
    )

    cal_5.insert(
        1,
        "split",
        split_name,
    )

    cal_4.to_csv(
        OUTPUT_DIR
        / f"model_4_{split_name}_calibration.csv",
        index=False,
    )

    cal_5.to_csv(
        OUTPUT_DIR
        / f"model_5_{split_name}_calibration.csv",
        index=False,
    )

    calibration_summary_rows.append(
        {
            "split": split_name,
            "model_4_ece": ece_4,
            "model_5_ece": ece_5,
            "model_5_minus_model_4_ece": (
                ece_5 - ece_4
            ),
            "model_4_mce": mce_4,
            "model_5_mce": mce_5,
            "model_5_minus_model_4_mce": (
                mce_5 - mce_4
            ),
        }
    )

    print(
        f"\n{split_name.upper()}"
    )

    print(
        f"  ECE: "
        f"Model 4 = {ece_4:.6f} | "
        f"Model 5 = {ece_5:.6f}"
    )

    print(
        f"  MCE: "
        f"Model 4 = {mce_4:.6f} | "
        f"Model 5 = {mce_5:.6f}"
    )

calibration_summary = pd.DataFrame(
    calibration_summary_rows
)

calibration_summary.to_csv(
    OUTPUT_DIR
    / "calibration_summary.csv",
    index=False,
)


# =============================================================================
# PREDICTION ALIGNMENT
# =============================================================================

print_header(
    "PREDICTION AGREEMENT AND DIFFERENCES"
)

validation_alignment = (
    compare_prediction_alignment(
        model_4_validation,
        model_5_validation,
        "validation",
    )
)

test_alignment = (
    compare_prediction_alignment(
        model_4_test,
        model_5_test,
        "test",
    )
)

alignment_rows = []

for (
    split_name,
    alignment,
) in [
    (
        "validation",
        validation_alignment,
    ),
    (
        "test",
        test_alignment,
    ),
]:

    mean_abs_difference = (
        alignment[
            "absolute_probability_difference"
        ].mean()
    )

    median_abs_difference = (
        alignment[
            "absolute_probability_difference"
        ].median()
    )

    max_abs_difference = (
        alignment[
            "absolute_probability_difference"
        ].max()
    )

    mean_difference = (
        alignment[
            "probability_difference"
        ].mean()
    )

    agreement_rate = (
        alignment[
            "prediction_agreement"
        ].mean()
    )

    correlation = (
        alignment[
            [
                "win_home_probability_model4",
                "win_home_probability_model5",
            ]
        ]
        .corr()
        .iloc[0, 1]
    )

    alignment_rows.append(
        {
            "split": split_name,
            "games": len(
                alignment
            ),
            "mean_probability_difference": (
                mean_difference
            ),
            "mean_absolute_probability_difference": (
                mean_abs_difference
            ),
            "median_absolute_probability_difference": (
                median_abs_difference
            ),
            "max_absolute_probability_difference": (
                max_abs_difference
            ),
            "prediction_agreement_rate": (
                agreement_rate
            ),
            "probability_correlation": (
                correlation
            ),
        }
    )

    print(
        f"\n{split_name.upper()}"
    )

    print(
        "  Mean probability difference "
        "(Model 5 - Model 4): "
        f"{mean_difference:+.6f}"
    )

    print(
        f"  Mean absolute difference: "
        f"{mean_abs_difference:.6f}"
    )

    print(
        f"  Median absolute difference: "
        f"{median_abs_difference:.6f}"
    )

    print(
        f"  Maximum absolute difference: "
        f"{max_abs_difference:.6f}"
    )

    print(
        f"  Prediction agreement: "
        f"{agreement_rate:.2%}"
    )

    print(
        f"  Probability correlation: "
        f"{correlation:.6f}"
    )

    alignment.to_csv(
        OUTPUT_DIR
        / f"{split_name}_prediction_comparison.csv",
        index=False,
    )

alignment_summary = pd.DataFrame(
    alignment_rows
)

alignment_summary.to_csv(
    OUTPUT_DIR
    / "prediction_alignment_summary.csv",
    index=False,
)


# =============================================================================
# LARGEST PREDICTION DIFFERENCES
# =============================================================================

print_header(
    "LARGEST MODEL DISAGREEMENTS"
)

for (
    split_name,
    alignment,
) in [
    (
        "validation",
        validation_alignment,
    ),
    (
        "test",
        test_alignment,
    ),
]:

    largest = (
        alignment
        .sort_values(
            "absolute_probability_difference",
            ascending=False,
        )
        .head(25)
        .copy()
    )

    largest.insert(
        0,
        "rank",
        range(
            1,
            len(largest) + 1,
        ),
    )

    largest.to_csv(
        OUTPUT_DIR
        / f"{split_name}_largest_prediction_differences.csv",
        index=False,
    )

    print(
        f"\n{split_name.upper()}"
    )

    print(
        largest[
            [
                "rank",
                "gameId",
                "season_model4",
                "win_home_actual_model4",
                "win_home_probability_model4",
                "win_home_probability_model5",
                "probability_difference",
                "win_home_prediction_model4",
                "win_home_prediction_model5",
            ]
        ].to_string(
            index=False
        )
    )


# =============================================================================
# PROBABILITY DISTRIBUTION
# =============================================================================

print_header(
    "PROBABILITY DISTRIBUTION"
)

distribution_rows = []

for (
    split_name,
    model_4_df,
    model_5_df,
) in [
    (
        "validation",
        model_4_validation,
        model_5_validation,
    ),
    (
        "test",
        model_4_test,
        model_5_test,
    ),
]:

    for (
        model_name,
        df,
    ) in [
        (
            "model_4",
            model_4_df,
        ),
        (
            "model_5",
            model_5_df,
        ),
    ]:

        probabilities = df[
            "win_home_probability"
        ]

        distribution_rows.append(
            {
                "split": split_name,
                "model": model_name,
                "mean": probabilities.mean(),
                "std": probabilities.std(),
                "min": probabilities.min(),
                "p01": probabilities.quantile(
                    0.01
                ),
                "p05": probabilities.quantile(
                    0.05
                ),
                "p25": probabilities.quantile(
                    0.25
                ),
                "median": probabilities.median(),
                "p75": probabilities.quantile(
                    0.75
                ),
                "p95": probabilities.quantile(
                    0.95
                ),
                "p99": probabilities.quantile(
                    0.99
                ),
                "max": probabilities.max(),
                "below_0_10": (
                    probabilities
                    < 0.10
                ).mean(),
                "below_0_20": (
                    probabilities
                    < 0.20
                ).mean(),
                "above_0_80": (
                    probabilities
                    > 0.80
                ).mean(),
                "above_0_90": (
                    probabilities
                    > 0.90
                ).mean(),
                "above_0_95": (
                    probabilities
                    > 0.95
                ).mean(),
            }
        )

distribution_summary = pd.DataFrame(
    distribution_rows
)

distribution_summary.to_csv(
    OUTPUT_DIR
    / "probability_distribution.csv",
    index=False,
)

print(
    distribution_summary.to_string(
        index=False
    )
)


# =============================================================================
# SEASON-LEVEL PERFORMANCE
# =============================================================================

print_header(
    "SEASON-LEVEL PERFORMANCE"
)

model_4_validation_season = (
    calculate_season_metrics(
        model_4_validation
    )
)

model_4_validation_season[
    "model"
] = "model_4"

model_4_validation_season[
    "split"
] = "validation"

model_5_validation_season = (
    calculate_season_metrics(
        model_5_validation
    )
)

model_5_validation_season[
    "model"
] = "model_5"

model_5_validation_season[
    "split"
] = "validation"

model_4_test_season = (
    calculate_season_metrics(
        model_4_test
    )
)

model_4_test_season[
    "model"
] = "model_4"

model_4_test_season[
    "split"
] = "test"

model_5_test_season = (
    calculate_season_metrics(
        model_5_test
    )
)

model_5_test_season[
    "model"
] = "model_5"

model_5_test_season[
    "split"
] = "test"

season_metrics = pd.concat(
    [
        model_4_validation_season,
        model_5_validation_season,
        model_4_test_season,
        model_5_test_season,
    ],
    ignore_index=True,
)

season_metrics = season_metrics[
    [
        "split",
        "season",
        "model",
        "games",
        "log_loss",
        "brier_score",
        "roc_auc",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
    ]
]

season_metrics.to_csv(
    OUTPUT_DIR
    / "season_metrics.csv",
    index=False,
)

print(
    season_metrics.to_string(
        index=False
    )
)


# =============================================================================
# SEASON-LEVEL MODEL DIFFERENCES
# =============================================================================

season_comparison_rows = []

for split_name in [
    "validation",
    "test",
]:

    model_4_split = (
        season_metrics.loc[
            (
                season_metrics[
                    "split"
                ]
                == split_name
            )
            &
            (
                season_metrics[
                    "model"
                ]
                == "model_4"
            )
        ].copy()
    )

    model_5_split = (
        season_metrics.loc[
            (
                season_metrics[
                    "split"
                ]
                == split_name
            )
            &
            (
                season_metrics[
                    "model"
                ]
                == "model_5"
            )
        ].copy()
    )

    merged_season = (
        model_4_split.merge(
            model_5_split,
            on=[
                "split",
                "season",
                "games",
            ],
            suffixes=(
                "_model4",
                "_model5",
            ),
        )
    )

    for _, row in (
        merged_season.iterrows()
    ):

        season_comparison_rows.append(
            {
                "split": split_name,
                "season": row[
                    "season"
                ],
                "games": row[
                    "games"
                ],
                "log_loss_model4": row[
                    "log_loss_model4"
                ],
                "log_loss_model5": row[
                    "log_loss_model5"
                ],
                "log_loss_difference": (
                    row[
                        "log_loss_model5"
                    ]
                    -
                    row[
                        "log_loss_model4"
                    ]
                ),
                "brier_model4": row[
                    "brier_score_model4"
                ],
                "brier_model5": row[
                    "brier_score_model5"
                ],
                "brier_difference": (
                    row[
                        "brier_score_model5"
                    ]
                    -
                    row[
                        "brier_score_model4"
                    ]
                ),
                "auc_model4": row[
                    "roc_auc_model4"
                ],
                "auc_model5": row[
                    "roc_auc_model5"
                ],
                "auc_difference": (
                    row[
                        "roc_auc_model5"
                    ]
                    -
                    row[
                        "roc_auc_model4"
                    ]
                ),
                "accuracy_model4": row[
                    "accuracy_model4"
                ],
                "accuracy_model5": row[
                    "accuracy_model5"
                ],
                "accuracy_difference": (
                    row[
                        "accuracy_model5"
                    ]
                    -
                    row[
                        "accuracy_model4"
                    ]
                ),
            }
        )

season_comparison = pd.DataFrame(
    season_comparison_rows
)

season_comparison.to_csv(
    OUTPUT_DIR
    / "season_comparison.csv",
    index=False,
)


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

print_header(
    "FEATURE IMPORTANCE COMPARISON"
)

model_4_importance = (
    load_feature_importance(
        MODEL_4_DIR,
        "Model 4",
    )
)

model_5_importance = (
    load_feature_importance(
        MODEL_5_DIR,
        "Model 5",
    )
)

feature_column_4 = (
    find_feature_column(
        model_4_importance
    )
)

importance_column_4 = (
    find_importance_column(
        model_4_importance
    )
)

feature_column_5 = (
    find_feature_column(
        model_5_importance
    )
)

importance_column_5 = (
    find_importance_column(
        model_5_importance
    )
)

importance_4 = (
    model_4_importance[
        [
            feature_column_4,
            importance_column_4,
        ]
    ].copy()
)

importance_4.columns = [
    "feature",
    "importance_model4",
]

importance_5 = (
    model_5_importance[
        [
            feature_column_5,
            importance_column_5,
        ]
    ].copy()
)

importance_5.columns = [
    "feature",
    "importance_model5",
]

# Normalize feature names to strings so the merge
# cannot fail because of CSV dtype differences.
importance_4[
    "feature"
] = importance_4[
    "feature"
].astype(str)

importance_5[
    "feature"
] = importance_5[
    "feature"
].astype(str)

importance_comparison = (
    importance_4.merge(
        importance_5,
        on="feature",
        how="outer",
    )
)

importance_comparison[
    [
        "importance_model4",
        "importance_model5",
    ]
] = importance_comparison[
    [
        "importance_model4",
        "importance_model5",
    ]
].fillna(0)

importance_comparison[
    "importance_difference"
] = (
    importance_comparison[
        "importance_model5"
    ]
    -
    importance_comparison[
        "importance_model4"
    ]
)

importance_comparison[
    "absolute_importance_difference"
] = (
    importance_comparison[
        "importance_difference"
    ].abs()
)

importance_comparison[
    "rank_model4"
] = (
    importance_comparison[
        "importance_model4"
    ]
    .rank(
        ascending=False,
        method="min",
    )
)

importance_comparison[
    "rank_model5"
] = (
    importance_comparison[
        "importance_model5"
    ]
    .rank(
        ascending=False,
        method="min",
    )
)

importance_comparison[
    "rank_change_model5_minus_model4"
] = (
    importance_comparison[
        "rank_model5"
    ]
    -
    importance_comparison[
        "rank_model4"
    ]
)

importance_comparison = (
    importance_comparison
    .sort_values(
        "importance_model5",
        ascending=False,
    )
)

importance_comparison.to_csv(
    OUTPUT_DIR
    / "feature_importance_comparison.csv",
    index=False,
)

print(
    importance_comparison[
        [
            "feature",
            "importance_model4",
            "importance_model5",
            "importance_difference",
            "rank_model4",
            "rank_model5",
            "rank_change_model5_minus_model4",
        ]
    ]
    .head(28)
    .to_string(
        index=False
    )
)


# =============================================================================
# FEATURE IMPORTANCE CONCENTRATION
# =============================================================================

print_header(
    "FEATURE IMPORTANCE CONCENTRATION"
)

concentration_rows = []

for (
    model_name,
    importance_column,
) in [
    (
        "model_4",
        "importance_model4",
    ),
    (
        "model_5",
        "importance_model5",
    ),
]:

    values = (
        importance_comparison[
            importance_column
        ]
        .sort_values(
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    total = values.sum()

    concentration_rows.append(
        {
            "model": model_name,
            "top_1_share": (
                values.head(1).sum()
                / total
            ),
            "top_3_share": (
                values.head(3).sum()
                / total
            ),
            "top_5_share": (
                values.head(5).sum()
                / total
            ),
            "top_10_share": (
                values.head(10).sum()
                / total
            ),
        }
    )

importance_concentration = (
    pd.DataFrame(
        concentration_rows
    )
)

importance_concentration.to_csv(
    OUTPUT_DIR
    / "feature_importance_concentration.csv",
    index=False,
)

print(
    importance_concentration.to_string(
        index=False
    )
)


# =============================================================================
# MODEL 5 TEMPORAL CV
# =============================================================================

print_header(
    "MODEL 5 TEMPORAL CV RESULTS"
)

temporal_cv_path = (
    MODEL_5_DIR
    / "temporal_cv_fold_results.csv"
)

if temporal_cv_path.exists():

    temporal_cv = pd.read_csv(
        temporal_cv_path
    )

    print(
        temporal_cv.to_string(
            index=False
        )
    )

    temporal_cv.to_csv(
        OUTPUT_DIR
        / "model_5_temporal_cv_results.csv",
        index=False,
    )

    possible_log_loss_columns = [
        column
        for column in temporal_cv.columns
        if (
            "log" in column.lower()
            and
            "loss" in column.lower()
        )
    ]

    if possible_log_loss_columns:

        cv_log_loss_column = (
            possible_log_loss_columns[0]
        )

        print()
        print(
            "Temporal CV mean Log Loss: "
            f"{temporal_cv[cv_log_loss_column].mean():.6f}"
        )

        print(
            "Temporal CV standard deviation: "
            f"{temporal_cv[cv_log_loss_column].std():.6f}"
        )

        print(
            "Temporal CV best fold: "
            f"{temporal_cv[cv_log_loss_column].min():.6f}"
        )

        print(
            "Temporal CV worst fold: "
            f"{temporal_cv[cv_log_loss_column].max():.6f}"
        )

else:

    print(
        "WARNING: Model 5 temporal CV results "
        "file not found."
    )


# =============================================================================
# TRAINING SUMMARY
# =============================================================================

print_header(
    "TRAINING SUMMARY COMPARISON"
)

summary_rows = []

for (
    model_name,
    model_dir,
) in [
    (
        "model_4",
        MODEL_4_DIR,
    ),
    (
        "model_5",
        MODEL_5_DIR,
    ),
]:

    summary_path = (
        model_dir
        / "training_summary.csv"
    )

    if summary_path.exists():

        summary = pd.read_csv(
            summary_path
        )

        if "model" in summary.columns:
            summary = summary.drop(columns = ["model"])

        summary.insert(
            0,
            "model",
            model_name,
        )

        summary_rows.append(
            summary
        )

if summary_rows:

    training_summary = pd.concat(
        summary_rows,
        ignore_index=True,
    )

    training_summary.to_csv(
        OUTPUT_DIR
        / "training_summary_comparison.csv",
        index=False,
    )

    print(
        training_summary.to_string(
            index=False
        )
    )


# =============================================================================
# OVERALL SCORECARD
# =============================================================================

print_header(
    "MODEL 4 VS MODEL 5 SCORECARD"
)

scorecard_rows = []

metrics_to_evaluate = [
    "log_loss",
    "brier_score",
    "roc_auc",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
]

for (
    split_name,
    metric_dict_4,
    metric_dict_5,
) in [
    (
        "validation",
        model_4_validation_metrics,
        model_5_validation_metrics,
    ),
    (
        "test",
        model_4_test_metrics,
        model_5_test_metrics,
    ),
]:

    for metric in metrics_to_evaluate:

        value_4 = (
            metric_dict_4[
                metric
            ]
        )

        value_5 = (
            metric_dict_5[
                metric
            ]
        )

        if metric in [
            "log_loss",
            "brier_score",
        ]:

            better_model = (
                "Model 4"
                if value_4 < value_5
                else
                "Model 5"
                if value_5 < value_4
                else
                "Tie"
            )

        else:

            better_model = (
                "Model 4"
                if value_4 > value_5
                else
                "Model 5"
                if value_5 > value_4
                else
                "Tie"
            )

        scorecard_rows.append(
            {
                "split": split_name,
                "metric": metric,
                "model_4": value_4,
                "model_5": value_5,
                "difference_model5_minus_model4": (
                    value_5 - value_4
                ),
                "better_model": better_model,
            }
        )

scorecard = pd.DataFrame(
    scorecard_rows
)

scorecard.to_csv(
    OUTPUT_DIR
    / "scorecard.csv",
    index=False,
)

print(
    scorecard.to_string(
        index=False
    )
)


# =============================================================================
# WIN COUNTS
# =============================================================================

print_header(
    "METRIC WIN COUNTS"
)

win_rows = []

for split_name in [
    "validation",
    "test",
]:

    split_scorecard = (
        scorecard.loc[
            scorecard["split"]
            == split_name
        ]
    )

    model_4_wins = (
        split_scorecard[
            "better_model"
        ]
        == "Model 4"
    ).sum()

    model_5_wins = (
        split_scorecard[
            "better_model"
        ]
        == "Model 5"
    ).sum()

    ties = (
        split_scorecard[
            "better_model"
        ]
        == "Tie"
    ).sum()

    win_rows.append(
        {
            "split": split_name,
            "model_4_wins": model_4_wins,
            "model_5_wins": model_5_wins,
            "ties": ties,
        }
    )

    print(
        f"{split_name.capitalize()}: "
        f"Model 4 = {model_4_wins} | "
        f"Model 5 = {model_5_wins} | "
        f"Ties = {ties}"
    )

win_counts = pd.DataFrame(
    win_rows
)

win_counts.to_csv(
    OUTPUT_DIR
    / "metric_win_counts.csv",
    index=False,
)


# =============================================================================
# FINAL AUDIT SUMMARY
# =============================================================================

print_header(
    "FINAL AUDIT SUMMARY"
)

validation_log_loss_difference = (
    model_5_validation_metrics[
        "log_loss"
    ]
    -
    model_4_validation_metrics[
        "log_loss"
    ]
)

test_log_loss_difference = (
    model_5_test_metrics[
        "log_loss"
    ]
    -
    model_4_test_metrics[
        "log_loss"
    ]
)

validation_brier_difference = (
    model_5_validation_metrics[
        "brier_score"
    ]
    -
    model_4_validation_metrics[
        "brier_score"
    ]
)

test_brier_difference = (
    model_5_test_metrics[
        "brier_score"
    ]
    -
    model_4_test_metrics[
        "brier_score"
    ]
)

validation_auc_difference = (
    model_5_validation_metrics[
        "roc_auc"
    ]
    -
    model_4_validation_metrics[
        "roc_auc"
    ]
)

test_auc_difference = (
    model_5_test_metrics[
        "roc_auc"
    ]
    -
    model_4_test_metrics[
        "roc_auc"
    ]
)

print(
    "\nProbability-quality comparison:"
)

print(
    "  Validation Log Loss: "
    "Model 5 - Model 4 = "
    f"{validation_log_loss_difference:+.6f}"
)

print(
    "  Test Log Loss: "
    "Model 5 - Model 4 = "
    f"{test_log_loss_difference:+.6f}"
)

print(
    "  Validation Brier: "
    "Model 5 - Model 4 = "
    f"{validation_brier_difference:+.6f}"
)

print(
    "  Test Brier: "
    "Model 5 - Model 4 = "
    f"{test_brier_difference:+.6f}"
)

print(
    "  Validation AUC: "
    "Model 5 - Model 4 = "
    f"{validation_auc_difference:+.6f}"
)

print(
    "  Test AUC: "
    "Model 5 - Model 4 = "
    f"{test_auc_difference:+.6f}"
)

validation_alignment_row = (
    alignment_summary.loc[
        alignment_summary["split"]
        == "validation"
    ].iloc[0]
)

test_alignment_row = (
    alignment_summary.loc[
        alignment_summary["split"]
        == "test"
    ].iloc[0]
)

print(
    "\nPrediction stability:"
)

print(
    "  Validation agreement: "
    f"{validation_alignment_row['prediction_agreement_rate']:.2%}"
)

print(
    "  Test agreement: "
    f"{test_alignment_row['prediction_agreement_rate']:.2%}"
)

print(
    "  Validation probability correlation: "
    f"{validation_alignment_row['probability_correlation']:.6f}"
)

print(
    "  Test probability correlation: "
    f"{test_alignment_row['probability_correlation']:.6f}"
)

print(
    "\nAudit artifacts saved to:"
)

print(
    f"  {OUTPUT_DIR}"
)

print()
print(
    "=" * 78
)

print(
    "MODEL 4 VS MODEL 5 AUDIT COMPLETE"
)

print(
    "=" * 78
)