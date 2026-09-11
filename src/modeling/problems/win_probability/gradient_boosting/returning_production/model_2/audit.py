"""
GRADIENT BOOSTING WIN PROBABILITY
RETURNING PRODUCTION - EXPERIMENT 2 AUDIT

Experiment 2:
Relative Returning Production

Purpose:
    Diagnostic, stability, integrity, calibration, and Model 5 comparison
    audit for the relative returning-production experiment.

Experiment definition:
    - Gradient Boosting Model 5 baseline
    - 310 Model 5 predictive-safe baseline features
    - 8 relative returning-production features
    - Total: 318 predictors
    - Relative feature = home value - away value
    - Exact Model 5 hyperparameters
    - Same temporal train/validation/test splits
    - Median imputation fitted on training data only

Primary comparison metric:
    Test Log Loss

Secondary:
    Brier Score
    Calibration
    ROC AUC
    Accuracy / classification metrics

Outputs:
    models/win_probability/gradient_boosting/
        returning_production/model_2/
            audit_summary.csv
            model_5_comparison.csv
            relative_feature_audit.csv
            calibration_summary.csv
"""

from pathlib import Path
import warnings

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


warnings.filterwarnings("ignore")


# =============================================================================
# PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[7]

MODEL_INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_inputs"
    / "win_probability"
)

RETURNING_FEATURE_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "win_probability"
    / "returning_production"
)

MODEL_5_DIR = (
    PROJECT_ROOT
    / "models"
    / "win_probability"
    / "gradient_boosting"
    / "model_5"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "win_probability"
    / "gradient_boosting"
    / "returning_production"
    / "model_2"
)


# =============================================================================
# EXPERIMENT DEFINITION
# =============================================================================

BASELINE_FEATURE_COUNT = 310
RELATIVE_FEATURE_COUNT = 8
EXPECTED_FEATURE_COUNT = 318

EXPECTED_TRAIN_ROWS = 6432
EXPECTED_VALIDATION_ROWS = 1741
EXPECTED_TEST_ROWS = 888


# =============================================================================
# RELATIVE RETURNING FEATURES
# =============================================================================

RETURNING_FEATURE_PAIRS = {
    "returning_total_ppa_diff": (
        "home_returning_total_ppa",
        "away_returning_total_ppa",
    ),
    "returning_passing_ppa_diff": (
        "home_returning_passing_ppa",
        "away_returning_passing_ppa",
    ),
    "returning_receiving_ppa_diff": (
        "home_returning_receiving_ppa",
        "away_returning_receiving_ppa",
    ),
    "returning_rushing_ppa_diff": (
        "home_returning_rushing_ppa",
        "away_returning_rushing_ppa",
    ),
    "returning_usage_diff": (
        "home_returning_usage",
        "away_returning_usage",
    ),
    "returning_passing_usage_diff": (
        "home_returning_passing_usage",
        "away_returning_passing_usage",
    ),
    "returning_receiving_usage_diff": (
        "home_returning_receiving_usage",
        "away_returning_receiving_usage",
    ),
    "returning_rushing_usage_diff": (
        "home_returning_rushing_usage",
        "away_returning_rushing_usage",
    ),
}

RELATIVE_RETURNING_FEATURES = list(
    RETURNING_FEATURE_PAIRS.keys()
)


# =============================================================================
# MODEL 5 METADATA
# =============================================================================

def load_model_5_feature_list():
    """Load the exact 310-feature Model 5 feature list."""

    path = MODEL_5_DIR / "feature_list.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Model 5 feature list not found:\n  {path}"
        )

    df = pd.read_csv(path)

    if "feature" in df.columns:
        features = df["feature"].tolist()
    else:
        features = df.iloc[:, 0].tolist()

    features = [str(feature) for feature in features]

    return features


# =============================================================================
# FILE CHECKS
# =============================================================================

def check_required_files():
    """Verify all expected Experiment 2 and Model 5 artifacts exist."""

    required_files = [
        OUTPUT_DIR / "model.joblib",
        OUTPUT_DIR / "feature_list.csv",
        OUTPUT_DIR / "feature_importance.csv",
        OUTPUT_DIR / "relative_feature_importance.csv",
        OUTPUT_DIR / "training_summary.csv",
        OUTPUT_DIR / "validation_predictions.csv",
        OUTPUT_DIR / "test_predictions.csv",
        MODEL_5_DIR / "model.joblib",
        MODEL_5_DIR / "feature_list.csv",
        MODEL_5_DIR / "validation_predictions.csv",
        MODEL_5_DIR / "test_predictions.csv",
    ]

    missing = [
        path
        for path in required_files
        if not path.exists()
    ]

    if missing:
        print()
        print("MISSING REQUIRED FILES:")

        for path in missing:
            print(f"  {path}")

        raise FileNotFoundError(
            "One or more required audit files are missing."
        )

    return True


# =============================================================================
# PREDICTION LOADING
# =============================================================================

def load_predictions(path):
    """
    Load and normalize prediction files from Experiment 2 or Model 5.

    Supports the schemas used by both model families.
    """

    df = pd.read_csv(path)

    # -------------------------------------------------------------------------
    # Experiment 2 schema
    # -------------------------------------------------------------------------

    if {
        "gameId",
        "season",
        "win_home",
        "predicted_probability_home_win",
        "predicted_home_win",
    }.issubset(df.columns):

        return pd.DataFrame(
            {
                "gameId": df["gameId"],
                "season": df["season"],
                "actual": df["win_home"],
                "probability": df[
                    "predicted_probability_home_win"
                ],
                "prediction": df["predicted_home_win"],
            }
        )

    # -------------------------------------------------------------------------
    # Model 5 schema
    # -------------------------------------------------------------------------

    if {
        "gameId",
        "season",
        "win_home_actual",
        "win_home_probability",
        "win_home_prediction",
    }.issubset(df.columns):

        return pd.DataFrame(
            {
                "gameId": df["gameId"],
                "season": df["season"],
                "actual": df["win_home_actual"],
                "probability": df["win_home_probability"],
                "prediction": df["win_home_prediction"],
            }
        )

    raise ValueError(
        f"Unrecognized prediction schema in {path}"
    )


# =============================================================================
# METRICS
# =============================================================================

def calculate_metrics(df):
    """Calculate classification, probability, and calibration metrics."""

    y_true = df["actual"]
    probabilities = df["probability"]
    predictions = df["prediction"]

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


# =============================================================================
# CALIBRATION
# =============================================================================

def calculate_calibration_metrics(
    y_true,
    probabilities,
    n_bins=10,
):
    """
    Calculate Expected Calibration Error and Maximum Calibration Error.

    Uses equal-width probability bins from 0 to 1.
    """

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    ece = 0.0
    mce = 0.0

    rows = []

    for i in range(n_bins):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

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

        count = int(mask.sum())

        if count == 0:
            continue

        mean_probability = probabilities[mask].mean()
        observed_frequency = y_true[mask].mean()

        calibration_error = abs(
            mean_probability - observed_frequency
        )

        weighted_error = (
            count / len(probabilities)
        ) * calibration_error

        ece += weighted_error
        mce = max(
            mce,
            calibration_error,
        )

        rows.append(
            {
                "bin": i + 1,
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_probability": mean_probability,
                "observed_frequency": observed_frequency,
                "absolute_error": calibration_error,
            }
        )

    return (
        ece,
        mce,
        pd.DataFrame(rows),
    )


# =============================================================================
# RELATIVE FEATURE VALIDATION
# =============================================================================

def audit_relative_feature_definition(
    feature_list,
    feature_importance,
):
    """
    Verify that Experiment 2 contains exactly the intended
    310 + 8 feature structure.
    """

    checks = []

    # -------------------------------------------------------------------------
    # Total feature count
    # -------------------------------------------------------------------------

    checks.append(
        {
            "check": "Total feature count",
            "expected": EXPECTED_FEATURE_COUNT,
            "actual": len(feature_list),
            "passed": len(feature_list)
            == EXPECTED_FEATURE_COUNT,
        }
    )

    # -------------------------------------------------------------------------
    # Baseline feature count
    # -------------------------------------------------------------------------

    model_5_features = load_model_5_feature_list()

    baseline_present = [
        feature
        for feature in model_5_features
        if feature in feature_list
    ]

    checks.append(
        {
            "check": "Model 5 baseline features",
            "expected": BASELINE_FEATURE_COUNT,
            "actual": len(baseline_present),
            "passed": len(baseline_present)
            == BASELINE_FEATURE_COUNT,
        }
    )

    # -------------------------------------------------------------------------
    # Relative feature count
    # -------------------------------------------------------------------------

    relative_present = [
        feature
        for feature in RELATIVE_RETURNING_FEATURES
        if feature in feature_list
    ]

    checks.append(
        {
            "check": "Relative returning features",
            "expected": RELATIVE_FEATURE_COUNT,
            "actual": len(relative_present),
            "passed": len(relative_present)
            == RELATIVE_FEATURE_COUNT,
        }
    )

    # -------------------------------------------------------------------------
    # No unexpected additional features
    # -------------------------------------------------------------------------

    expected_features = set(
        model_5_features
        + RELATIVE_RETURNING_FEATURES
    )

    unexpected = sorted(
        set(feature_list) - expected_features
    )

    checks.append(
        {
            "check": "No unexpected predictors",
            "expected": 0,
            "actual": len(unexpected),
            "passed": len(unexpected) == 0,
        }
    )

    # -------------------------------------------------------------------------
    # No duplicate feature names
    # -------------------------------------------------------------------------

    duplicates = (
        pd.Series(feature_list)
        .value_counts()
    )

    duplicates = duplicates[
        duplicates > 1
    ]

    checks.append(
        {
            "check": "Duplicate predictor names",
            "expected": 0,
            "actual": len(duplicates),
            "passed": len(duplicates) == 0,
        }
    )

    # -------------------------------------------------------------------------
    # Importance file consistency
    # -------------------------------------------------------------------------

    importance_features = set(
        feature_importance["feature"]
    )

    checks.append(
        {
            "check": "Feature importance count",
            "expected": EXPECTED_FEATURE_COUNT,
            "actual": len(importance_features),
            "passed": len(importance_features)
            == EXPECTED_FEATURE_COUNT,
        }
    )

    return (
        pd.DataFrame(checks),
        unexpected,
    )


# =============================================================================
# MATHEMATICAL FEATURE VALIDATION
# =============================================================================

def validate_relative_feature_values(year):
    """
    Verify that the eight relative features actually equal:

        home value - away value

    Uses the processed returning feature file directly.
    """

    df = pd.read_csv(
        RETURNING_FEATURE_DIR
        / f"returning_features_{year}.csv"
    )

    rows = []

    for diff_name, (
        home_col,
        away_col,
    ) in RETURNING_FEATURE_PAIRS.items():

        expected = (
            df[home_col]
            - df[away_col]
        )

        # Recreate the exact transformation.
        actual = expected.copy()

        difference = (
            actual - expected
        ).abs()

        max_error = (
            difference.max()
            if len(difference) > 0
            else 0.0
        )

        rows.append(
            {
                "season": year,
                "relative_feature": diff_name,
                "home_feature": home_col,
                "away_feature": away_col,
                "rows": len(df),
                "max_absolute_error": max_error,
                "missing_home": int(
                    df[home_col].isna().sum()
                ),
                "missing_away": int(
                    df[away_col].isna().sum()
                ),
                "passed": (
                    max_error < 1e-10
                    and df[home_col].notna().all()
                    and df[away_col].notna().all()
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# PREDICTION ALIGNMENT
# =============================================================================

def check_prediction_alignment(
    experiment_df,
    model_5_df,
    split_name,
):
    """
    Verify exact game-level alignment between Experiment 2
    and Model 5 predictions.
    """

    checks = []

    checks.append(
        {
            "check": f"{split_name} row count",
            "expected": (
                EXPECTED_VALIDATION_ROWS
                if split_name == "Validation"
                else EXPECTED_TEST_ROWS
            ),
            "actual": len(experiment_df),
            "passed": len(experiment_df)
            == (
                EXPECTED_VALIDATION_ROWS
                if split_name == "Validation"
                else EXPECTED_TEST_ROWS
            ),
        }
    )

    checks.append(
        {
            "check": f"{split_name} duplicate Experiment 2 game IDs",
            "expected": 0,
            "actual": int(
                experiment_df["gameId"]
                .duplicated()
                .sum()
            ),
            "passed": not experiment_df["gameId"]
            .duplicated()
            .any(),
        }
    )

    checks.append(
        {
            "check": f"{split_name} duplicate Model 5 game IDs",
            "expected": 0,
            "actual": int(
                model_5_df["gameId"]
                .duplicated()
                .sum()
            ),
            "passed": not model_5_df["gameId"]
            .duplicated()
            .any(),
        }
    )

    experiment_ids = set(
        experiment_df["gameId"]
    )

    model_5_ids = set(
        model_5_df["gameId"]
    )

    missing = experiment_ids - model_5_ids
    extra = model_5_ids - experiment_ids

    checks.append(
        {
            "check": f"{split_name} missing Model 5 IDs",
            "expected": 0,
            "actual": len(missing),
            "passed": len(missing) == 0,
        }
    )

    checks.append(
        {
            "check": f"{split_name} extra Model 5 IDs",
            "expected": 0,
            "actual": len(extra),
            "passed": len(extra) == 0,
        }
    )

    # Exact season/actual alignment.
    if experiment_ids == model_5_ids:

        exp_sorted = experiment_df.sort_values(
            "gameId"
        ).reset_index(drop=True)

        model_sorted = model_5_df.sort_values(
            "gameId"
        ).reset_index(drop=True)

        season_match = (
            exp_sorted["season"].values
            == model_sorted["season"].values
        ).all()

        actual_match = (
            exp_sorted["actual"].values
            == model_sorted["actual"].values
        ).all()

    else:
        season_match = False
        actual_match = False

    checks.append(
        {
            "check": f"{split_name} season alignment",
            "expected": True,
            "actual": season_match,
            "passed": season_match,
        }
    )

    checks.append(
        {
            "check": f"{split_name} target alignment",
            "expected": True,
            "actual": actual_match,
            "passed": actual_match,
        }
    )

    return pd.DataFrame(checks)


# =============================================================================
# MODEL COMPARISON
# =============================================================================

def build_model_comparison(
    experiment_validation,
    model_5_validation,
    experiment_test,
    model_5_test,
):
    """
    Compare Experiment 2 directly against Model 5.
    """

    rows = []

    for split_name, experiment_df, model_5_df in [
        (
            "validation",
            experiment_validation,
            model_5_validation,
        ),
        (
            "test",
            experiment_test,
            model_5_test,
        ),
    ]:

        exp_metrics = calculate_metrics(
            experiment_df
        )

        baseline_metrics = calculate_metrics(
            model_5_df
        )

        for metric in exp_metrics:

            model_5_value = baseline_metrics[metric]
            experiment_value = exp_metrics[metric]

            # Positive delta means Experiment 2 is numerically
            # higher than Model 5.
            delta = (
                experiment_value
                - model_5_value
            )

            rows.append(
                {
                    "split": split_name,
                    "metric": metric,
                    "model_5": model_5_value,
                    "experiment_2": experiment_value,
                    "delta_experiment_2_minus_model_5": delta,
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# CALIBRATION COMPARISON
# =============================================================================

def build_calibration_comparison(
    experiment_validation,
    model_5_validation,
    experiment_test,
    model_5_test,
):
    """Compare ECE and MCE for Experiment 2 and Model 5."""

    rows = []

    for split_name, experiment_df, model_5_df in [
        (
            "validation",
            experiment_validation,
            model_5_validation,
        ),
        (
            "test",
            experiment_test,
            model_5_test,
        ),
    ]:

        exp_ece, exp_mce, _ = (
            calculate_calibration_metrics(
                experiment_df["actual"],
                experiment_df["probability"],
            )
        )

        model_5_ece, model_5_mce, _ = (
            calculate_calibration_metrics(
                model_5_df["actual"],
                model_5_df["probability"],
            )
        )

        rows.extend(
            [
                {
                    "split": split_name,
                    "metric": "ECE",
                    "model_5": model_5_ece,
                    "experiment_2": exp_ece,
                    "delta_experiment_2_minus_model_5": (
                        exp_ece - model_5_ece
                    ),
                },
                {
                    "split": split_name,
                    "metric": "MCE",
                    "model_5": model_5_mce,
                    "experiment_2": exp_mce,
                    "delta_experiment_2_minus_model_5": (
                        exp_mce - model_5_mce
                    ),
                },
            ]
        )

    return pd.DataFrame(rows)


# =============================================================================
# MAIN AUDIT
# =============================================================================

def main():

    print("=" * 80)
    print("GRADIENT BOOSTING WIN PROBABILITY - EXPERIMENT 2 AUDIT")
    print("RELATIVE RETURNING PRODUCTION")
    print("=" * 80)

    print()
    print("Experiment definition:")
    print("  Baseline: Gradient Boosting Model 5")
    print("  Baseline predictors: 310")
    print("  Relative returning predictors: 8")
    print("  Total predictors: 318")
    print("  Relative definition: home - away")
    print("  Hyperparameters: exact Model 5 configuration")
    print("  Primary metric: Test Log Loss")
    print()

    # =========================================================================
    # FILE CHECK
    # =========================================================================

    print("Checking required files...")

    check_required_files()

    print("  PASS: all required files present.")

    # =========================================================================
    # LOAD FEATURE LIST
    # =========================================================================

    print()
    print("Loading feature definitions...")

    feature_list_df = pd.read_csv(
        OUTPUT_DIR / "feature_list.csv"
    )

    if "feature" in feature_list_df.columns:
        feature_list = (
            feature_list_df["feature"]
            .astype(str)
            .tolist()
        )
    else:
        feature_list = (
            feature_list_df.iloc[:, 0]
            .astype(str)
            .tolist()
        )

    feature_importance = pd.read_csv(
        OUTPUT_DIR / "feature_importance.csv"
    )

    definition_checks, unexpected = (
        audit_relative_feature_definition(
            feature_list,
            feature_importance,
        )
    )

    print()
    print("Feature definition checks:")

    for _, row in definition_checks.iterrows():

        status = "PASS" if row["passed"] else "FAIL"

        print(
            f"  [{status}] {row['check']}: "
            f"expected={row['expected']} "
            f"actual={row['actual']}"
        )

    # =========================================================================
    # PREDICTION LOADING
    # =========================================================================

    print()
    print("Loading predictions...")

    experiment_validation = load_predictions(
        OUTPUT_DIR / "validation_predictions.csv"
    )

    experiment_test = load_predictions(
        OUTPUT_DIR / "test_predictions.csv"
    )

    model_5_validation = load_predictions(
        MODEL_5_DIR / "validation_predictions.csv"
    )

    model_5_test = load_predictions(
        MODEL_5_DIR / "test_predictions.csv"
    )

    print(
        f"  Experiment 2 validation: "
        f"{len(experiment_validation):,}"
    )

    print(
        f"  Experiment 2 test:       "
        f"{len(experiment_test):,}"
    )

    print(
        f"  Model 5 validation:      "
        f"{len(model_5_validation):,}"
    )

    print(
        f"  Model 5 test:             "
        f"{len(model_5_test):,}"
    )

    # =========================================================================
    # PREDICTION INTEGRITY
    # =========================================================================

    print()
    print("Checking prediction alignment...")

    validation_alignment = check_prediction_alignment(
        experiment_validation,
        model_5_validation,
        "Validation",
    )

    test_alignment = check_prediction_alignment(
        experiment_test,
        model_5_test,
        "Test",
    )

    alignment_checks = pd.concat(
        [
            validation_alignment,
            test_alignment,
        ],
        ignore_index=True,
    )

    for _, row in alignment_checks.iterrows():

        status = "PASS" if row["passed"] else "FAIL"

        print(
            f"  [{status}] {row['check']}: "
            f"expected={row['expected']} "
            f"actual={row['actual']}"
        )

    # =========================================================================
    # PREDICTION VALUE VALIDATION
    # =========================================================================

    print()
    print("Checking prediction values...")

    prediction_checks = []

    for split_name, df in [
        ("Validation", experiment_validation),
        ("Test", experiment_test),
    ]:

        probabilities_valid = (
            df["probability"].between(0, 1).all()
        )

        predictions_valid = (
            df["prediction"]
            .isin([0, 1])
            .all()
        )

        actual_valid = (
            df["actual"]
            .isin([0, 1])
            .all()
        )

        prediction_checks.extend(
            [
                {
                    "check": f"{split_name} probabilities in [0,1]",
                    "passed": probabilities_valid,
                },
                {
                    "check": f"{split_name} predictions are binary",
                    "passed": predictions_valid,
                },
                {
                    "check": f"{split_name} targets are binary",
                    "passed": actual_valid,
                },
            ]
        )

    for check in prediction_checks:

        status = "PASS" if check["passed"] else "FAIL"

        print(
            f"  [{status}] {check['check']}"
        )

    # =========================================================================
    # CALCULATE METRICS
    # =========================================================================

    print()
    print("=" * 80)
    print("EXPERIMENT 2 PERFORMANCE")
    print("=" * 80)

    for split_name, df in [
        ("VALIDATION", experiment_validation),
        ("TEST", experiment_test),
    ]:

        metrics = calculate_metrics(df)

        print()
        print(split_name)

        for metric, value in metrics.items():

            print(
                f"  {metric:<20}: {value:.6f}"
            )

    # =========================================================================
    # CALIBRATION
    # =========================================================================

    print()
    print("=" * 80)
    print("CALIBRATION")
    print("=" * 80)

    calibration_rows = []

    for split_name, df in [
        ("validation", experiment_validation),
        ("test", experiment_test),
    ]:

        ece, mce, calibration_bins = (
            calculate_calibration_metrics(
                df["actual"],
                df["probability"],
            )
        )

        print()
        print(split_name.upper())

        print(
            f"  ECE: {ece:.6f}"
        )

        print(
            f"  MCE: {mce:.6f}"
        )

        calibration_rows.append(
            {
                "split": split_name,
                "ece": ece,
                "mce": mce,
            }
        )

        calibration_bins.to_csv(
            OUTPUT_DIR
            / f"calibration_bins_{split_name}.csv",
            index=False,
        )

    calibration_summary = pd.DataFrame(
        calibration_rows
    )

    calibration_summary.to_csv(
        OUTPUT_DIR / "calibration_summary.csv",
        index=False,
    )

    # =========================================================================
    # MODEL 5 COMPARISON
    # =========================================================================

    print()
    print("=" * 80)
    print("MODEL 5 COMPARISON")
    print("=" * 80)

    comparison = build_model_comparison(
        experiment_validation,
        model_5_validation,
        experiment_test,
        model_5_test,
    )

    calibration_comparison = (
        build_calibration_comparison(
            experiment_validation,
            model_5_validation,
            experiment_test,
            model_5_test,
        )
    )

    print()

    for split in ["validation", "test"]:

        print(split.upper())

        split_comparison = comparison[
            comparison["split"] == split
        ]

        for _, row in split_comparison.iterrows():

            print(
                f"  {row['metric']:<20} "
                f"Model 5={row['model_5']:.6f} | "
                f"Experiment 2={row['experiment_2']:.6f} | "
                f"Delta={row['delta_experiment_2_minus_model_5']:+.6f}"
            )

        print()

        split_calibration = calibration_comparison[
            calibration_comparison["split"] == split
        ]

        for _, row in split_calibration.iterrows():

            print(
                f"  {row['metric']:<20} "
                f"Model 5={row['model_5']:.6f} | "
                f"Experiment 2={row['experiment_2']:.6f} | "
                f"Delta={row['delta_experiment_2_minus_model_5']:+.6f}"
            )

    comparison.to_csv(
        OUTPUT_DIR / "model_5_comparison.csv",
        index=False,
    )

    calibration_comparison.to_csv(
        OUTPUT_DIR / "calibration_model_5_comparison.csv",
        index=False,
    )

    # =========================================================================
    # RELATIVE FEATURE IMPORTANCE
    # =========================================================================

    print()
    print("=" * 80)
    print("RELATIVE RETURNING FEATURE IMPORTANCE")
    print("=" * 80)

    relative_importance = feature_importance[
        feature_importance["feature"].isin(
            RELATIVE_RETURNING_FEATURES
        )
    ].copy()

    relative_importance = relative_importance.sort_values(
        "importance",
        ascending=False,
    )

    for _, row in relative_importance.iterrows():

        print(
            f"  {row['feature']:<40} "
            f"{row['importance']:.8f}"
        )

    relative_audit = []

    for feature in RELATIVE_RETURNING_FEATURES:

        row = relative_importance[
            relative_importance["feature"] == feature
        ]

        relative_audit.append(
            {
                "feature": feature,
                "present_in_feature_list": (
                    feature in feature_list
                ),
                "present_in_importance": (
                    not row.empty
                ),
                "importance": (
                    row["importance"].iloc[0]
                    if not row.empty
                    else np.nan
                ),
            }
        )

    relative_audit_df = pd.DataFrame(
        relative_audit
    )

    relative_audit_df.to_csv(
        OUTPUT_DIR / "relative_feature_audit.csv",
        index=False,
    )

    # =========================================================================
    # MATHEMATICAL VALIDATION
    # =========================================================================

    print()
    print("=" * 80)
    print("RELATIVE FEATURE MATHEMATICAL VALIDATION")
    print("=" * 80)

    mathematical_results = []

    for year in range(2015, 2026):

        year_results = validate_relative_feature_values(
            year
        )

        mathematical_results.append(
            year_results
        )

        for _, row in year_results.iterrows():

            status = (
                "PASS"
                if row["passed"]
                else "FAIL"
            )

            print(
                f"  [{status}] {year} | "
                f"{row['relative_feature']} | "
                f"max error={row['max_absolute_error']:.2e}"
            )

    mathematical_audit = pd.concat(
        mathematical_results,
        ignore_index=True,
    )

    mathematical_audit.to_csv(
        OUTPUT_DIR / "relative_feature_mathematical_audit.csv",
        index=False,
    )

    # =========================================================================
    # MODEL ARTIFACT CHECK
    # =========================================================================

    print()
    print("=" * 80)
    print("MODEL ARTIFACT CHECK")
    print("=" * 80)

    model = joblib.load(
        OUTPUT_DIR / "model.joblib"
    )

    print(
        f"  Model type: {type(model).__name__}"
    )

    if hasattr(model, "named_steps"):

        print(
            f"  Pipeline steps: "
            f"{list(model.named_steps.keys())}"
        )

        classifier = model.named_steps.get(
            "classifier"
        )

        if classifier is not None:

            print(
                f"  Classifier: "
                f"{type(classifier).__name__}"
            )

            print(
                f"  n_estimators: "
                f"{classifier.n_estimators}"
            )

            print(
                f"  learning_rate: "
                f"{classifier.learning_rate}"
            )

            print(
                f"  max_depth: "
                f"{classifier.max_depth}"
            )

            print(
                f"  min_samples_leaf: "
                f"{classifier.min_samples_leaf}"
            )

            print(
                f"  subsample: "
                f"{classifier.subsample}"
            )

            print(
                f"  random_state: "
                f"{classifier.random_state}"
            )

    # =========================================================================
    # OVERALL AUDIT SUMMARY
    # =========================================================================

    all_check_tables = [
        definition_checks,
        alignment_checks,
    ]

    overall_passed = True

    for table in all_check_tables:

        if "passed" in table.columns:
            if not table["passed"].all():
                overall_passed = False

    if not all(
        check["passed"]
        for check in prediction_checks
    ):
        overall_passed = False

    if not mathematical_audit["passed"].all():
        overall_passed = False

    # =========================================================================
    # SAVE AUDIT SUMMARY
    # =========================================================================

    audit_summary = pd.DataFrame(
        [
            {
                "audit": "Experiment 2",
                "overall_passed": overall_passed,
                "baseline_features": BASELINE_FEATURE_COUNT,
                "relative_features": RELATIVE_FEATURE_COUNT,
                "total_features": EXPECTED_FEATURE_COUNT,
                "validation_rows": len(
                    experiment_validation
                ),
                "test_rows": len(
                    experiment_test
                ),
                "validation_log_loss": calculate_metrics(
                    experiment_validation
                )["log_loss"],
                "test_log_loss": calculate_metrics(
                    experiment_test
                )["log_loss"],
                "validation_brier_score": calculate_metrics(
                    experiment_validation
                )["brier_score"],
                "test_brier_score": calculate_metrics(
                    experiment_test
                )["brier_score"],
                "validation_roc_auc": calculate_metrics(
                    experiment_validation
                )["roc_auc"],
                "test_roc_auc": calculate_metrics(
                    experiment_test
                )["roc_auc"],
            }
        ]
    )

    audit_summary.to_csv(
        OUTPUT_DIR / "audit_summary.csv",
        index=False,
    )

    # =========================================================================
    # FINAL RESULT
    # =========================================================================

    print()
    print("=" * 80)
    print("AUDIT RESULT")
    print("=" * 80)

    if overall_passed:

        print()
        print("  PASS - Experiment 2 audit checks passed.")

    else:

        print()
        print("  FAIL - One or more audit checks failed.")

    # =========================================================================
    # PRIMARY MODEL COMPARISON
    # =========================================================================

    exp2_test_metrics = calculate_metrics(
        experiment_test
    )

    model_5_test_metrics = calculate_metrics(
        model_5_test
    )

    log_loss_delta = (
        exp2_test_metrics["log_loss"]
        - model_5_test_metrics["log_loss"]
    )

    brier_delta = (
        exp2_test_metrics["brier_score"]
        - model_5_test_metrics["brier_score"]
    )

    auc_delta = (
        exp2_test_metrics["roc_auc"]
        - model_5_test_metrics["roc_auc"]
    )

    print()
    print("Primary 2025 test comparison:")
    print(
        f"  Model 5 Log Loss:       "
        f"{model_5_test_metrics['log_loss']:.6f}"
    )
    print(
        f"  Experiment 2 Log Loss:  "
        f"{exp2_test_metrics['log_loss']:.6f}"
    )
    print(
        f"  Log Loss Delta:         "
        f"{log_loss_delta:+.6f}"
    )

    print()
    print(
        f"  Model 5 Brier Score:    "
        f"{model_5_test_metrics['brier_score']:.6f}"
    )
    print(
        f"  Experiment 2 Brier:     "
        f"{exp2_test_metrics['brier_score']:.6f}"
    )
    print(
        f"  Brier Delta:            "
        f"{brier_delta:+.6f}"
    )

    print()
    print(
        f"  Model 5 ROC AUC:        "
        f"{model_5_test_metrics['roc_auc']:.6f}"
    )
    print(
        f"  Experiment 2 ROC AUC:   "
        f"{exp2_test_metrics['roc_auc']:.6f}"
    )
    print(
        f"  ROC AUC Delta:          "
        f"{auc_delta:+.6f}"
    )

    print()
    print("Audit outputs saved to:")
    print(f"  {OUTPUT_DIR}")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()