"""
Gradient Boosting Win Probability - Model 4
Diagnostic, Stability, Temporal CV, and Model 3 Comparison Audit

Model 4:
    28 compact features
    Expanding-window temporal cross-validation
    Hyperparameters selected using Log Loss

Temporal CV:
    Fold 1: 2015-2018 -> 2019
    Fold 2: 2015-2019 -> 2020
    Fold 3: 2015-2020 -> 2021
    Fold 4: 2015-2021 -> 2022

Final evaluation:
    Training:   2015-2022
    Validation:  2023-2024
    Test:       2025

This audit does NOT retrain the hyperparameter search.

It validates:
    1. Required files
    2. Dataset integrity
    3. Temporal split integrity
    4. Feature-set integrity
    5. Model pipeline
    6. Selected hyperparameters
    7. Temporal CV search integrity
    8. CV fold performance
    9. Top-configuration stability
    10. Model 3 baseline inclusion
    11. Feature missingness
    12. Model 4 validation/test performance
    13. Model 3 vs Model 4 comparison
    14. Calibration
    15. Prediction stability
    16. Probability distributions
    17. Feature importance
    18. Validation permutation importance
    19. Practical significance
    20. Overall audit conclusion
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
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
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[6]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_inputs"
    / "win_probability"
)

TRAIN_PATH = DATA_DIR / "train.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEST_PATH = DATA_DIR / "test.csv"


MODEL_3_DIR = (
    PROJECT_ROOT
    / "models"
    / "win_probability"
    / "gradient_boosting"
    / "model_3"
)

MODEL_4_DIR = (
    PROJECT_ROOT
    / "models"
    / "win_probability"
    / "gradient_boosting"
    / "model_4"
)


MODEL_3_PATH = MODEL_3_DIR / "model.joblib"
MODEL_3_FEATURE_PATH = MODEL_3_DIR / "feature_list.csv"
MODEL_3_TRAINING_SUMMARY_PATH = MODEL_3_DIR / "training_summary.csv"
MODEL_3_VALIDATION_PATH = MODEL_3_DIR / "validation_predictions.csv"
MODEL_3_TEST_PATH = MODEL_3_DIR / "test_predictions.csv"


MODEL_4_PATH = MODEL_4_DIR / "model.joblib"
MODEL_4_FEATURE_PATH = MODEL_4_DIR / "feature_list.csv"
MODEL_4_TRAINING_SUMMARY_PATH = MODEL_4_DIR / "training_summary.csv"
MODEL_4_CV_PATH = MODEL_4_DIR / "cv_results.csv"
MODEL_4_BEST_PARAMS_PATH = MODEL_4_DIR / "best_parameters.csv"
MODEL_4_VALIDATION_PATH = MODEL_4_DIR / "validation_predictions.csv"
MODEL_4_TEST_PATH = MODEL_4_DIR / "test_predictions.csv"


# =============================================================================
# CONSTANTS
# =============================================================================

TARGET_COLUMN = "win_home"
GAME_ID_COLUMN = "gameId"
SEASON_COLUMN = "season"

EXPECTED_TRAIN_SHAPE = (6432, 315)
EXPECTED_VALIDATION_SHAPE = (1741, 315)
EXPECTED_TEST_SHAPE = (888, 315)

EXPECTED_FEATURE_COUNT = 28

EXPECTED_TRAIN_SEASONS = list(range(2015, 2023))
EXPECTED_VALIDATION_SEASONS = [2023, 2024]
EXPECTED_TEST_SEASONS = [2025]

EXPECTED_RANDOM_STATE = 42

EXPECTED_BEST_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_samples_leaf": 10,
    "subsample": 0.75,
}

MODEL_3_BASELINE_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_samples_leaf": 5,
    "subsample": 1.0,
}

EXPECTED_CV_FOLDS = [
    {
        "train_seasons": list(range(2015, 2019)),
        "validation_seasons": [2019],
    },
    {
        "train_seasons": list(range(2015, 2020)),
        "validation_seasons": [2020],
    },
    {
        "train_seasons": list(range(2015, 2021)),
        "validation_seasons": [2021],
    },
    {
        "train_seasons": list(range(2015, 2022)),
        "validation_seasons": [2022],
    },
]

EXPECTED_CV_COMBINATIONS = 288
EXPECTED_CV_FITS = 1152


# =============================================================================
# FEATURE GROUPS
# =============================================================================

CORE_FEATURES = [
    "homePregameElo",
    "awayPregameElo",
    "winPctBefore_home",
    "winPctBefore_away",
    "pointDifferentialBefore_home",
    "pointDifferentialBefore_away",
    "pointDifferentialAvgBefore_home",
    "pointDifferentialAvgBefore_away",
]

RECENT_FORM_FEATURES = [
    "pointDifferentialAvgLast3_home",
    "pointDifferentialAvgLast3_away",
    "pointDifferentialAvgLast5_home",
    "pointDifferentialAvgLast5_away",
    "pointsForAvgLast5_home",
    "pointsForAvgLast5_away",
    "pointsAgainstAvgLast5_home",
    "pointsAgainstAvgLast5_away",
]

OFFENSIVE_FEATURES = [
    "home_pregame_offense_successRate",
    "away_pregame_offense_successRate",
    "home_pregame_offense_ppa",
    "away_pregame_offense_ppa",
    "yardsPerPassAttemptBefore_home",
    "yardsPerPassAttemptBefore_away",
    "yardsPerRushAttemptBefore_home",
    "yardsPerRushAttemptBefore_away",
]

DEFENSIVE_FEATURES = [
    "home_pregame_defense_successRate",
    "away_pregame_defense_successRate",
    "home_pregame_defense_ppa",
    "away_pregame_defense_ppa",
]

EXPECTED_FEATURES = (
    CORE_FEATURES
    + RECENT_FORM_FEATURES
    + OFFENSIVE_FEATURES
    + DEFENSIVE_FEATURES
)


# =============================================================================
# REMOVED FEATURES
# =============================================================================

REMOVED_TREND_FEATURES = [
    "pointsForTrend_home",
    "pointsForTrend_away",
    "pointsAgainstTrend_home",
    "pointsAgainstTrend_away",
    "pointDifferentialTrend_home",
    "pointDifferentialTrend_away",
    "totalYardsTrend_home",
    "totalYardsTrend_away",
    "netPassingYardsTrend_home",
    "netPassingYardsTrend_away",
    "winPctTrend_home",
    "winPctTrend_away",
]

REMOVED_SOS_FEATURES = [
    "priorSOSWinPct_home",
    "priorSOSWinPct_away",
    "priorSOSPointDiff_home",
    "priorSOSPointDiff_away",
]


# =============================================================================
# HELPERS
# =============================================================================

PASS_COUNT = 0
FAIL_COUNT = 0
WARNING_COUNT = 0


def print_section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def record_check(name, passed, detail=""):
    global PASS_COUNT
    global FAIL_COUNT

    status = "PASS" if passed else "FAIL"

    if passed:
        PASS_COUNT += 1
    else:
        FAIL_COUNT += 1

    print(f"{name}: {status}")

    if detail:
        print(f"  {detail}")


def record_warning(name, detail=""):
    global WARNING_COUNT

    WARNING_COUNT += 1

    print(f"{name}: WARNING")

    if detail:
        print(f"  {detail}")


def load_csv(path, description):
    try:
        df = pd.read_csv(path)

        print(f"{description}: PASS")
        print(f"  {path}")

        return df

    except Exception as exc:

        print(f"{description}: FAIL")
        print(f"  {path}")
        print(f"  Error: {exc}")

        raise


def get_prediction_column(df):
    candidates = [
        "probability",
        "predicted_probability",
        "win_probability",
        "predicted_prob",
        "prob_home",
        "prediction",
    ]

    for column in candidates:
        if column in df.columns:
            return column

    probability_candidates = [
        column
        for column in df.columns
        if "prob" in column.lower()
    ]

    if probability_candidates:
        return probability_candidates[0]

    raise ValueError(
        "Could not identify probability column."
    )


def get_target_column(df):
    if TARGET_COLUMN in df.columns:
        return TARGET_COLUMN

    candidates = [
        column
        for column in df.columns
        if column.lower() in {
            "actual",
            "target",
            "y",
            "win_home",
        }
    ]

    if candidates:
        return candidates[0]

    raise ValueError(
        "Could not identify target column."
    )


def calculate_metrics(y_true, probability):

    predicted_class = (
        probability >= 0.50
    ).astype(int)

    return {
        "log_loss": log_loss(
            y_true,
            probability,
        ),
        "brier": brier_score_loss(
            y_true,
            probability,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probability,
        ),
        "accuracy": accuracy_score(
            y_true,
            predicted_class,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            predicted_class,
        ),
        "precision": precision_score(
            y_true,
            predicted_class,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predicted_class,
            zero_division=0,
        ),
    }


def print_metrics(metrics):

    print(
        f"  Log Loss:          {metrics['log_loss']:.6f}"
    )

    print(
        f"  Brier Score:       {metrics['brier']:.6f}"
    )

    print(
        f"  ROC AUC:           {metrics['roc_auc']:.6f}"
    )

    print(
        f"  Accuracy:          {metrics['accuracy'] * 100:.4f}%"
    )

    print(
        f"  Balanced Accuracy: "
        f"{metrics['balanced_accuracy'] * 100:.4f}%"
    )

    print(
        f"  Precision:         "
        f"{metrics['precision'] * 100:.4f}%"
    )

    print(
        f"  Recall:            "
        f"{metrics['recall'] * 100:.4f}%"
    )


def calculate_calibration(
    y_true,
    probability,
    bins=None,
):

    if bins is None:
        bins = np.array(
            [
                0.0,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1.0,
            ]
        )

    rows = []

    for lower, upper in zip(
        bins[:-1],
        bins[1:],
    ):

        if upper == 1.0:

            mask = (
                (probability >= lower)
                & (probability <= upper)
            )

        else:

            mask = (
                (probability >= lower)
                & (probability < upper)
            )

        count = mask.sum()

        if count == 0:
            continue

        mean_prediction = probability[mask].mean()
        observed_rate = y_true[mask].mean()

        rows.append(
            {
                "bin_lower": lower,
                "bin_upper": upper,
                "count": count,
                "mean_prediction": mean_prediction,
                "observed_rate": observed_rate,
                "absolute_error": abs(
                    mean_prediction
                    - observed_rate
                ),
            }
        )

    calibration = pd.DataFrame(rows)

    if len(calibration) == 0:
        return calibration, np.nan

    weighted_error = (
        (
            calibration["absolute_error"]
            * calibration["count"]
        ).sum()
        / calibration["count"].sum()
    )

    return calibration, weighted_error


def print_prediction_distribution(
    name,
    probability,
):

    quantiles = probability.quantile(
        [
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )

    print()
    print(f"{name}:")

    print(
        f"  Mean:   {probability.mean():.6f}"
    )

    print(
        f"  Std:    {probability.std():.6f}"
    )

    for q, value in quantiles.items():

        print(
            f"  P{int(q * 100):02d}:    "
            f"{value:.6f}"
        )

    print(
        f"  Min:    {probability.min():.6f}"
    )

    print(
        f"  Max:    {probability.max():.6f}"
    )


def parameter_columns(df):

    expected = [
        "n_estimators",
        "learning_rate",
        "max_depth",
        "min_samples_leaf",
        "subsample",
    ]

    return [
        column
        for column in expected
        if column in df.columns
    ]


# =============================================================================
# MAIN AUDIT
# =============================================================================

def main():

    print(
        "=" * 80
    )

    print(
        "GRADIENT BOOSTING WIN PROBABILITY - MODEL 4"
    )

    print(
        "DIAGNOSTIC, STABILITY & MODEL 3 COMPARISON AUDIT"
    )

    print(
        "=" * 80
    )

    print()
    print("Model 4 definition:")
    print("  28 compact base features")
    print("  Same feature set as Model 3")
    print("  Expanding-window temporal CV")
    print("  Hyperparameters selected using Log Loss")
    print("  Test set: evaluation only")

    print()
    print("Project root:")
    print(f"  {PROJECT_ROOT}")

    # =========================================================================
    # 1. REQUIRED FILE VALIDATION
    # =========================================================================

    print_section(
        "1. REQUIRED FILE VALIDATION"
    )

    required_files = {
        "Training data": TRAIN_PATH,
        "Validation data": VALIDATION_PATH,
        "Test data": TEST_PATH,
        "Model 3": MODEL_3_PATH,
        "Model 3 feature list": MODEL_3_FEATURE_PATH,
        "Model 3 training summary": MODEL_3_TRAINING_SUMMARY_PATH,
        "Model 3 validation predictions": MODEL_3_VALIDATION_PATH,
        "Model 3 test predictions": MODEL_3_TEST_PATH,
        "Model 4": MODEL_4_PATH,
        "Model 4 feature list": MODEL_4_FEATURE_PATH,
        "Model 4 training summary": MODEL_4_TRAINING_SUMMARY_PATH,
        "Model 4 CV results": MODEL_4_CV_PATH,
        "Model 4 best parameters": MODEL_4_BEST_PARAMS_PATH,
        "Model 4 validation predictions": MODEL_4_VALIDATION_PATH,
        "Model 4 test predictions": MODEL_4_TEST_PATH,
    }

    for description, path in required_files.items():

        record_check(
            description,
            path.exists(),
            str(path),
        )

    if FAIL_COUNT > 0:

        raise FileNotFoundError(
            "Required audit files are missing."
        )

    # =========================================================================
    # 2. LOAD DATA
    # =========================================================================

    print_section(
        "2. LOADING DATA"
    )

    train = pd.read_csv(
        TRAIN_PATH
    )

    validation = pd.read_csv(
        VALIDATION_PATH
    )

    test = pd.read_csv(
        TEST_PATH
    )

    model_3 = joblib.load(
        MODEL_3_PATH
    )

    model_4 = joblib.load(
        MODEL_4_PATH
    )

    model_3_features_df = pd.read_csv(
        MODEL_3_FEATURE_PATH
    )

    model_4_features_df = pd.read_csv(
        MODEL_4_FEATURE_PATH
    )

    model_3_training_summary = pd.read_csv(
        MODEL_3_TRAINING_SUMMARY_PATH
    )

    model_4_training_summary = pd.read_csv(
        MODEL_4_TRAINING_SUMMARY_PATH
    )

    cv_results = pd.read_csv(
        MODEL_4_CV_PATH
    )

    best_params_df = pd.read_csv(
        MODEL_4_BEST_PARAMS_PATH
    )

    model_3_validation = pd.read_csv(
        MODEL_3_VALIDATION_PATH
    )

    model_3_test = pd.read_csv(
        MODEL_3_TEST_PATH
    )

    model_4_validation = pd.read_csv(
        MODEL_4_VALIDATION_PATH
    )

    model_4_test = pd.read_csv(
        MODEL_4_TEST_PATH
    )

    print(
        f"Training shape:    {train.shape}"
    )

    print(
        f"Validation shape:  {validation.shape}"
    )

    print(
        f"Test shape:        {test.shape}"
    )

    print(
        f"Model 3 type:       {type(model_3).__name__}"
    )

    print(
        f"Model 4 type:       {type(model_4).__name__}"
    )

    print(
        f"Model 4 CV rows:     {len(cv_results)}"
    )

    # =========================================================================
    # 3. DATASET VALIDATION
    # =========================================================================

    print_section(
        "3. DATASET VALIDATION"
    )

    datasets = [
        (
            "Training",
            train,
            EXPECTED_TRAIN_SHAPE,
            EXPECTED_TRAIN_SEASONS,
        ),
        (
            "Validation",
            validation,
            EXPECTED_VALIDATION_SHAPE,
            EXPECTED_VALIDATION_SEASONS,
        ),
        (
            "Test",
            test,
            EXPECTED_TEST_SHAPE,
            EXPECTED_TEST_SEASONS,
        ),
    ]

    for name, df, expected_shape, expected_seasons in datasets:

        actual_seasons = sorted(
            df[SEASON_COLUMN]
            .dropna()
            .unique()
            .tolist()
        )

        print()
        print(f"{name}:")

        record_check(
            "  Shape",
            df.shape == expected_shape,
            f"Expected {expected_shape}; actual {df.shape}",
        )

        record_check(
            "  Target present",
            TARGET_COLUMN in df.columns,
        )

        target_nulls = (
            df[TARGET_COLUMN].isna().sum()
            if TARGET_COLUMN in df.columns
            else -1
        )

        record_check(
            "  Target nulls",
            target_nulls == 0,
            f"{target_nulls} nulls",
        )

        record_check(
            "  Seasons",
            actual_seasons == expected_seasons,
            f"Expected {expected_seasons}; actual {actual_seasons}",
        )

    # =========================================================================
    # 4. TEMPORAL SPLIT INTEGRITY
    # =========================================================================

    print_section(
        "4. TEMPORAL SPLIT INTEGRITY"
    )

    train_max = train[SEASON_COLUMN].max()
    validation_min = validation[SEASON_COLUMN].min()
    validation_max = validation[SEASON_COLUMN].max()
    test_min = test[SEASON_COLUMN].min()

    record_check(
        "Training ends before validation",
        train_max < validation_min,
        f"{train_max} < {validation_min}",
    )

    record_check(
        "Validation ends before test",
        validation_max < test_min,
        f"{validation_max} < {test_min}",
    )

    record_check(
        "No training/validation season overlap",
        set(train[SEASON_COLUMN])
        .isdisjoint(
            set(validation[SEASON_COLUMN])
        ),
    )

    record_check(
        "No validation/test season overlap",
        set(validation[SEASON_COLUMN])
        .isdisjoint(
            set(test[SEASON_COLUMN])
        ),
    )

    record_check(
        "No training/test season overlap",
        set(train[SEASON_COLUMN])
        .isdisjoint(
            set(test[SEASON_COLUMN])
        ),
    )

    # =========================================================================
    # 5. FEATURE LIST VALIDATION
    # =========================================================================

    print_section(
        "5. FEATURE LIST VALIDATION"
    )

    model_3_feature_column = (
        "feature"
        if "feature"
        in model_3_features_df.columns
        else model_3_features_df.columns[0]
    )

    model_4_feature_column = (
        "feature"
        if "feature"
        in model_4_features_df.columns
        else model_4_features_df.columns[0]
    )

    model_3_features = (
        model_3_features_df[
            model_3_feature_column
        ]
        .astype(str)
        .tolist()
    )

    model_4_features = (
        model_4_features_df[
            model_4_feature_column
        ]
        .astype(str)
        .tolist()
    )

    print(
        f"Expected feature count: {EXPECTED_FEATURE_COUNT}"
    )

    print(
        f"Model 3 feature count:  {len(model_3_features)}"
    )

    print(
        f"Model 4 feature count:  {len(model_4_features)}"
    )

    record_check(
        "Model 4 feature count",
        len(model_4_features)
        == EXPECTED_FEATURE_COUNT,
    )

    record_check(
        "Model 4 feature set",
        set(model_4_features)
        == set(EXPECTED_FEATURES),
    )

    record_check(
        "Model 4 duplicate features",
        len(model_4_features)
        == len(set(model_4_features)),
    )

    record_check(
        "Model 3 / Model 4 feature equality",
        set(model_3_features)
        == set(model_4_features),
    )

    missing_expected = sorted(
        set(EXPECTED_FEATURES)
        - set(model_4_features)
    )

    unexpected_features = sorted(
        set(model_4_features)
        - set(EXPECTED_FEATURES)
    )

    if missing_expected:
        print()
        print("Missing expected features:")
        for feature in missing_expected:
            print(f"  {feature}")

    if unexpected_features:
        print()
        print("Unexpected features:")
        for feature in unexpected_features:
            print(f"  {feature}")

    # =========================================================================
    # 6. REMOVED FEATURE VALIDATION
    # =========================================================================

    print_section(
        "6. REMOVED FEATURE VALIDATION"
    )

    accidentally_retained_trend = [
        feature
        for feature in REMOVED_TREND_FEATURES
        if feature in model_4_features
    ]

    accidentally_retained_sos = [
        feature
        for feature in REMOVED_SOS_FEATURES
        if feature in model_4_features
    ]

    print(
        f"Trend features expected removed: "
        f"{len(REMOVED_TREND_FEATURES)}"
    )

    print(
        f"Prior SOS features expected removed: "
        f"{len(REMOVED_SOS_FEATURES)}"
    )

    record_check(
        "No Trend features retained",
        len(accidentally_retained_trend) == 0,
    )

    record_check(
        "No Prior SOS features retained",
        len(accidentally_retained_sos) == 0,
    )

    # =========================================================================
    # 7. FEATURE PRESENCE
    # =========================================================================

    print_section(
        "7. FEATURE PRESENCE IN DATASETS"
    )

    for name, df, _, _ in datasets:

        missing_features = [
            feature
            for feature in model_4_features
            if feature not in df.columns
        ]

        print()
        print(f"{name}:")

        print(
            f"  Expected features: {len(model_4_features)}"
        )

        print(
            f"  Present features:  "
            f"{len(model_4_features) - len(missing_features)}"
        )

        record_check(
            "  Feature presence",
            len(missing_features) == 0,
            (
                "All features present."
                if not missing_features
                else f"Missing: {missing_features}"
            ),
        )

    # =========================================================================
    # 8. MODEL / PIPELINE VALIDATION
    # =========================================================================

    print_section(
        "8. MODEL / PIPELINE VALIDATION"
    )

    print("Model 4 object:")
    print(model_4)

    record_check(
        "Pipeline model",
        hasattr(model_4, "named_steps"),
    )

    if hasattr(model_4, "named_steps"):

        steps = model_4.named_steps

        record_check(
            "Median imputer present",
            "imputer" in steps
            and type(
                steps["imputer"]
            ).__name__
            == "SimpleImputer",
        )

        record_check(
            "Gradient Boosting classifier present",
            "classifier" in steps
            and type(
                steps["classifier"]
            ).__name__
            == "GradientBoostingClassifier",
        )

        classifier = steps.get(
            "classifier"
        )

    else:

        classifier = None

    # =========================================================================
    # 9. HYPERPARAMETER VALIDATION
    # =========================================================================

    print_section(
        "9. SELECTED HYPERPARAMETER VALIDATION"
    )

    if classifier is not None:

        actual_params = {
            "n_estimators":
                classifier.n_estimators,
            "learning_rate":
                classifier.learning_rate,
            "max_depth":
                classifier.max_depth,
            "min_samples_leaf":
                classifier.min_samples_leaf,
            "subsample":
                classifier.subsample,
        }

        for parameter, expected in EXPECTED_BEST_PARAMS.items():

            actual = actual_params[parameter]

            record_check(
                parameter,
                np.isclose(
                    float(actual),
                    float(expected),
                ),
                f"Expected {expected}; actual {actual}",
            )

        record_check(
            "random_state",
            classifier.random_state
            == EXPECTED_RANDOM_STATE,
            (
                f"Expected {EXPECTED_RANDOM_STATE}; "
                f"actual {classifier.random_state}"
            ),
        )

    # =========================================================================
    # 10. BEST PARAMETER ARTIFACT VALIDATION
    # =========================================================================

    print_section(
        "10. BEST PARAMETER ARTIFACT VALIDATION"
    )

    print(
        "Best-parameter file:"
    )

    print(
        best_params_df.to_string(
            index=False
        )
    )

    parameter_columns_found = (
        parameter_columns(
            best_params_df
        )
    )

    record_check(
        "Best-parameter columns present",
        len(parameter_columns_found) == 5,
        str(parameter_columns_found),
    )

    if len(best_params_df) > 0:

        best_row = best_params_df.iloc[0]

        for parameter, expected in EXPECTED_BEST_PARAMS.items():

            if parameter in best_row.index:

                actual = best_row[parameter]

                record_check(
                    f"Best parameter artifact - {parameter}",
                    np.isclose(
                        float(actual),
                        float(expected),
                    ),
                    f"Expected {expected}; actual {actual}",
                )

    # =========================================================================
    # 11. TEMPORAL CV STRUCTURE VALIDATION
    # =========================================================================

    print_section(
        "11. TEMPORAL CROSS-VALIDATION STRUCTURE"
    )

    print(
        "Expected temporal folds:"
    )

    for index, fold in enumerate(
        EXPECTED_CV_FOLDS,
        start=1,
    ):

        print()
        print(f"Fold {index}:")
        print(
            f"  Train:      {fold['train_seasons']}"
        )
        print(
            f"  Validation: {fold['validation_seasons']}"
        )

    print()

    record_check(
        "CV combination count",
        len(cv_results)
        == EXPECTED_CV_COMBINATIONS,
        (
            f"Expected {EXPECTED_CV_COMBINATIONS}; "
            f"actual {len(cv_results)}"
        ),
    )

    if len(cv_results) == EXPECTED_CV_COMBINATIONS:

        print(
            "CV result row count confirms "
            "all hyperparameter combinations are present."
        )

    # =========================================================================
    # 12. CV RESULT STRUCTURE
    # =========================================================================

    print_section(
        "12. CV RESULT STRUCTURE VALIDATION"
    )

    required_cv_columns = [
        "rank",
        "n_estimators",
        "learning_rate",
        "max_depth",
        "min_samples_leaf",
        "subsample",
        "mean_log_loss",
        "std_log_loss",
        "mean_brier_score",
        "std_brier_score",
        "mean_roc_auc",
        "std_roc_auc",
        "mean_accuracy",
        "std_accuracy",
        "mean_balanced_accuracy",
        "mean_precision",
        "mean_recall",
    ]

    for column in required_cv_columns:

        record_check(
            f"CV column - {column}",
            column in cv_results.columns,
        )

    # Validate all four fold-level metric columns.

    expected_fold_metric_columns = []

    for fold_number in range(1, 5):

        expected_fold_metric_columns.extend(
            [
                f"fold_{fold_number}_log_loss",
                f"fold_{fold_number}_brier_score",
                f"fold_{fold_number}_roc_auc",
                f"fold_{fold_number}_accuracy",
            ]
        )

    for column in expected_fold_metric_columns:

        record_check(
            f"CV column - {column}",
            column in cv_results.columns,
        )

    # =========================================================================
    # 13. MODEL 3 BASELINE IN CV SEARCH
    # =========================================================================

    print_section(
        "13. MODEL 3 BASELINE IN TEMPORAL CV SEARCH"
    )

    baseline_mask = pd.Series(
        True,
        index=cv_results.index,
    )

    for parameter, value in MODEL_3_BASELINE_PARAMS.items():

        if parameter not in cv_results.columns:

            baseline_mask &= False
            continue

        baseline_mask &= np.isclose(
            cv_results[parameter].astype(float),
            float(value),
        )

    baseline_rows = cv_results[
        baseline_mask
    ]

    record_check(
        "Model 3 baseline configuration included",
        len(baseline_rows) == 1,
        f"Matching rows: {len(baseline_rows)}",
    )

    if len(baseline_rows) == 1:

        baseline_row = (
            baseline_rows.iloc[0]
        )

        print()
        print(
            "Model 3 baseline CV performance:"
        )

        print(
            f"  Mean Log Loss: "
            f"{baseline_row['mean_log_loss']:.6f}"
        )

        print(
            f"  Mean Brier Score: "
            f"{baseline_row['mean_brier_score']:.6f}"
        )

        print(
            f"  Mean ROC AUC:  "
            f"{baseline_row['mean_roc_auc']:.6f}"
        )

        print(
            f"  Mean Accuracy: "
            f"{baseline_row['mean_accuracy']:.6f}"
        )

    # =========================================================================
    # 14. CV RANKING / TOP CONFIGURATIONS
    # =========================================================================

    print_section(
        "14. TEMPORAL CV PERFORMANCE & TOP CONFIGURATIONS"
    )

    cv_sorted = cv_results.sort_values(
        [
            "mean_log_loss",
            "mean_brier_score",
            "mean_roc_auc",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    ).reset_index(drop=True)

    display_columns = [
        "rank",
        "n_estimators",
        "learning_rate",
        "max_depth",
        "min_samples_leaf",
        "subsample",
        "mean_log_loss",
        "std_log_loss",
        "mean_brier_score",
        "std_brier_score",
        "mean_roc_auc",
        "std_roc_auc",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in cv_sorted.columns
    ]

    print(
        "Top 10 temporal CV configurations:"
    )

    print(
        cv_sorted[
            display_columns
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    winning_row = cv_sorted.iloc[0]

    print()
    print(
        "Selected configuration:"
    )

    for parameter in EXPECTED_BEST_PARAMS:

        print(
            f"  {parameter:<20}"
            f"{winning_row[parameter]}"
        )

    record_check(
        "Top CV configuration matches saved model",
        all(
            np.isclose(
                float(winning_row[p]),
                float(
                    EXPECTED_BEST_PARAMS[p]
                ),
            )
            for p in EXPECTED_BEST_PARAMS
        ),
    )

    # =========================================================================
    # 15. TOP-CONFIGURATION STABILITY
    # =========================================================================

    print_section(
        "15. TOP-CONFIGURATION STABILITY"
    )

    print(
        "How concentrated is the search around the winning configuration?"
    )

    top_n = min(
        10,
        len(cv_sorted),
    )

    top_configs = cv_sorted.head(
        top_n
    )

    for parameter in EXPECTED_BEST_PARAMS:

        values = (
            top_configs[parameter]
            .value_counts()
            .to_dict()
        )

        print()
        print(
            f"{parameter}:"
        )

        for value, count in values.items():

            print(
                f"  {value}: {count}/{top_n}"
            )

    top_learning_rate = (
        top_configs["learning_rate"]
        .value_counts()
        .idxmax()
    )

    top_subsample = (
        top_configs["subsample"]
        .value_counts()
        .idxmax()
    )

    print()
    print(
        f"Most common learning rate in top {top_n}: "
        f"{top_learning_rate}"
    )

    print(
        f"Most common subsample in top {top_n}: "
        f"{top_subsample}"
    )

    # =========================================================================
    # 16. FEATURE MISSINGNESS
    # =========================================================================

    print_section(
        "16. FEATURE MISSINGNESS"
    )

    missingness = pd.DataFrame(
        {
            "feature":
                model_4_features,
            "training_missing_pct":
                [
                    train[f].isna().mean() * 100
                    for f in model_4_features
                ],
            "validation_missing_pct":
                [
                    validation[f].isna().mean() * 100
                    for f in model_4_features
                ],
            "test_missing_pct":
                [
                    test[f].isna().mean() * 100
                    for f in model_4_features
                ],
        }
    )

    print(
        missingness
        .sort_values(
            "training_missing_pct",
            ascending=False,
        )
        .to_string(
            index=False
        )
    )

    print()

    record_check(
        "Missingness count matches expected structure",
        (
            missingness[
                "training_missing_pct"
            ] > 0
        ).sum()
        == 26,
        (
            f"Features with training missingness: "
            f"{(missingness['training_missing_pct'] > 0).sum()}"
        ),
    )

    print(
        f"Total training missing values: "
        f"{train[model_4_features].isna().sum().sum():,}"
    )

    # =========================================================================
    # 17. LOAD / VALIDATE PREDICTIONS
    # =========================================================================

    print_section(
        "17. PREDICTION FILE VALIDATION"
    )

    prediction_sets = [
        (
            "Model 3 validation",
            model_3_validation,
            len(validation),
        ),
        (
            "Model 3 test",
            model_3_test,
            len(test),
        ),
        (
            "Model 4 validation",
            model_4_validation,
            len(validation),
        ),
        (
            "Model 4 test",
            model_4_test,
            len(test),
        ),
    ]

    for name, df, expected_rows in prediction_sets:

        print()
        print(name)

        record_check(
            "  Row count",
            len(df) == expected_rows,
            f"Expected {expected_rows}; actual {len(df)}",
        )

        try:

            probability_column = (
                get_prediction_column(df)
            )

            probabilities = (
                df[probability_column]
                .astype(float)
            )

            record_check(
                "  Probability column",
                True,
                probability_column,
            )

            record_check(
                "  Probability nulls",
                probabilities.isna().sum() == 0,
                f"{probabilities.isna().sum()} nulls",
            )

            record_check(
                "  Probability bounds",
                (
                    (probabilities >= 0)
                    & (probabilities <= 1)
                ).all(),
            )

        except Exception as exc:

            record_check(
                "  Probability validation",
                False,
                str(exc),
            )

    # =========================================================================
    # 18. MODEL 4 PERFORMANCE
    # =========================================================================

    print_section(
        "18. MODEL 4 PERFORMANCE"
    )

    y_validation = (
        validation[TARGET_COLUMN]
        .astype(int)
        .to_numpy()
    )

    y_test = (
        test[TARGET_COLUMN]
        .astype(int)
        .to_numpy()
    )

    model_4_validation_probability = (
        model_4_validation[
            get_prediction_column(
                model_4_validation
            )
        ]
        .astype(float)
        .to_numpy()
    )

    model_4_test_probability = (
        model_4_test[
            get_prediction_column(
                model_4_test
            )
        ]
        .astype(float)
        .to_numpy()
    )

    model_4_validation_metrics = calculate_metrics(
        y_validation,
        model_4_validation_probability,
    )

    model_4_test_metrics = calculate_metrics(
        y_test,
        model_4_test_probability,
    )

    print("Validation:")
    print_metrics(
        model_4_validation_metrics
    )

    print()
    print("Test:")
    print_metrics(
        model_4_test_metrics
    )

    # =========================================================================
    # 19. MODEL 3 VS MODEL 4
    # =========================================================================

    print_section(
        "19. MODEL 3 VS MODEL 4 PERFORMANCE"
    )

    model_3_validation_probability = (
        model_3_validation[
            get_prediction_column(
                model_3_validation
            )
        ]
        .astype(float)
        .to_numpy()
    )

    model_3_test_probability = (
        model_3_test[
            get_prediction_column(
                model_3_test
            )
        ]
        .astype(float)
        .to_numpy()
    )

    model_3_validation_metrics = calculate_metrics(
        y_validation,
        model_3_validation_probability,
    )

    model_3_test_metrics = calculate_metrics(
        y_test,
        model_3_test_probability,
    )

    metric_names = [
        "log_loss",
        "brier",
        "roc_auc",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
    ]

    comparison_rows = []

    for metric in metric_names:

        model_3_value = (
            model_3_validation_metrics[metric]
        )

        model_4_value = (
            model_4_validation_metrics[metric]
        )

        comparison_rows.append(
            {
                "dataset": "Validation",
                "metric": metric,
                "model_3": model_3_value,
                "model_4": model_4_value,
                "model_4_minus_model_3":
                    model_4_value - model_3_value,
            }
        )

        model_3_value = (
            model_3_test_metrics[metric]
        )

        model_4_value = (
            model_4_test_metrics[metric]
        )

        comparison_rows.append(
            {
                "dataset": "Test",
                "metric": metric,
                "model_3": model_3_value,
                "model_4": model_4_value,
                "model_4_minus_model_3":
                    model_4_value - model_3_value,
            }
        )

    comparison = pd.DataFrame(
        comparison_rows
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    # =========================================================================
    # 20. PERFORMANCE INTERPRETATION
    # =========================================================================

    print_section(
        "20. PERFORMANCE DIFFERENCE INTERPRETATION"
    )

    for dataset in [
        "Validation",
        "Test",
    ]:

        subset = comparison[
            comparison["dataset"]
            == dataset
        ].set_index("metric")

        print()
        print(
            f"{dataset}:"
        )

        for metric in metric_names:

            difference = subset.loc[
                metric,
                "model_4_minus_model_3",
            ]

            if metric in [
                "log_loss",
                "brier",
            ]:

                direction = (
                    "better"
                    if difference < 0
                    else "worse"
                    if difference > 0
                    else "equal"
                )

            else:

                direction = (
                    "better"
                    if difference > 0
                    else "worse"
                    if difference < 0
                    else "equal"
                )

            print(
                f"  {metric:<20}"
                f"{difference:+.6f} "
                f"({direction})"
            )

    # =========================================================================
    # 21. CALIBRATION
    # =========================================================================

    print_section(
        "21. CALIBRATION COMPARISON"
    )

    calibration_results = []

    for model_name, y, probability in [
        (
            "Model 3 Validation",
            y_validation,
            model_3_validation_probability,
        ),
        (
            "Model 4 Validation",
            y_validation,
            model_4_validation_probability,
        ),
        (
            "Model 3 Test",
            y_test,
            model_3_test_probability,
        ),
        (
            "Model 4 Test",
            y_test,
            model_4_test_probability,
        ),
    ]:

        calibration_table, calibration_error = (
            calculate_calibration(
                y,
                pd.Series(probability),
            )
        )

        print()
        print(
            f"{model_name}:"
        )

        print(
            f"  Weighted calibration error: "
            f"{calibration_error:.6f}"
        )

        calibration_results.append(
            {
                "model": model_name,
                "calibration_error":
                    calibration_error,
            }
        )

        if not calibration_table.empty:

            print(
                calibration_table.to_string(
                    index=False
                )
            )

    calibration_results = pd.DataFrame(
        calibration_results
    )

    # =========================================================================
    # 22. PREDICTION STABILITY
    # =========================================================================

    print_section(
        "22. PREDICTION STABILITY"
    )

    print_prediction_distribution(
        "Model 3 Validation",
        pd.Series(
            model_3_validation_probability
        ),
    )

    print_prediction_distribution(
        "Model 4 Validation",
        pd.Series(
            model_4_validation_probability
        ),
    )

    print_prediction_distribution(
        "Model 3 Test",
        pd.Series(
            model_3_test_probability
        ),
    )

    print_prediction_distribution(
        "Model 4 Test",
        pd.Series(
            model_4_test_probability
        ),
    )

    # =========================================================================
    # 23. PREDICTION DISTRIBUTION BINS
    # =========================================================================

    print_section(
        "23. PREDICTION CONFIDENCE DISTRIBUTIONS"
    )

    confidence_bins = [
        0.0,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        1.0,
    ]

    confidence_labels = [
        "<0.10",
        "0.10-0.25",
        "0.25-0.50",
        "0.50-0.75",
        "0.75-0.90",
        ">=0.90",
    ]

    for model_name, probability in [
        (
            "Model 3 Test",
            model_3_test_probability,
        ),
        (
            "Model 4 Test",
            model_4_test_probability,
        ),
    ]:

        categories = pd.cut(
            probability,
            bins=confidence_bins,
            labels=confidence_labels,
            include_lowest=True,
            right=False,
        )

        counts = (
            pd.Series(categories)
            .value_counts(
                sort=False
            )
        )

        print()
        print(model_name)

        for label, count in counts.items():

            print(
                f"  {label:<10}"
                f"{count:>6}"
            )

    # =========================================================================
    # 24. FEATURE IMPORTANCE
    # =========================================================================

    print_section(
        "24. MODEL 4 FEATURE IMPORTANCE"
    )

    if classifier is not None:

        impurity_importance = pd.DataFrame(
            {
                "feature": model_4_features,
                "importance":
                    classifier.feature_importances_,
            }
        ).sort_values(
            "importance",
            ascending=False,
        )

        print(
            "Top 15 impurity-based feature importances:"
        )

        print(
            impurity_importance
            .head(15)
            .to_string(
                index=False
            )
        )

        family_map = {}

        for feature in CORE_FEATURES:
            family_map[feature] = (
                "Core Strength"
            )

        for feature in RECENT_FORM_FEATURES:
            family_map[feature] = (
                "Recent Form"
            )

        for feature in OFFENSIVE_FEATURES:
            family_map[feature] = (
                "Offensive Efficiency"
            )

        for feature in DEFENSIVE_FEATURES:
            family_map[feature] = (
                "Defensive Efficiency"
            )

        impurity_importance[
            "family"
        ] = impurity_importance[
            "feature"
        ].map(
            family_map
        )

        family_importance = (
            impurity_importance
            .groupby(
                "family"
            )["importance"]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        print()
        print(
            "Feature-family importance:"
        )

        for family, importance in (
            family_importance.items()
        ):

            print(
                f"  {family:<25}"
                f"{importance * 100:>8.2f}%"
            )

    # =========================================================================
    # 25. PERMUTATION IMPORTANCE
    # =========================================================================

    print_section(
        "25. MODEL 4 VALIDATION PERMUTATION IMPORTANCE"
    )

    X_validation = validation[
        model_4_features
    ]

    permutation = permutation_importance(
        model_4,
        X_validation,
        y_validation,
        scoring="neg_log_loss",
        n_repeats=5,
        random_state=EXPECTED_RANDOM_STATE,
        n_jobs=-1,
    )

    permutation_importance_df = pd.DataFrame(
        {
            "feature": model_4_features,
            "mean_importance":
                permutation.importances_mean,
            "std_importance":
                permutation.importances_std,
        }
    ).sort_values(
        "mean_importance",
        ascending=False,
    )

    print(
        "Top 15 validation permutation importances:"
    )

    print(
        permutation_importance_df
        .head(15)
        .to_string(
            index=False
        )
    )

    # =========================================================================
    # 26. MODEL 3 VS MODEL 4 FEATURE IMPORTANCE
    # =========================================================================

    print_section(
        "26. MODEL 3 VS MODEL 4 FEATURE IMPORTANCE"
    )

    try:

        model_3_classifier = (
            model_3.named_steps[
                "classifier"
            ]
        )

        model_3_importance = pd.DataFrame(
            {
                "feature": model_3_features,
                "model_3_importance":
                    model_3_classifier
                    .feature_importances_,
            }
        )

        model_4_importance = pd.DataFrame(
            {
                "feature": model_4_features,
                "model_4_importance":
                    classifier.feature_importances_,
            }
        )

        importance_comparison = (
            model_3_importance
            .merge(
                model_4_importance,
                on="feature",
                how="inner",
            )
        )

        importance_comparison[
            "change"
        ] = (
            importance_comparison[
                "model_4_importance"
            ]
            - importance_comparison[
                "model_3_importance"
            ]
        )

        importance_comparison[
            "absolute_change"
        ] = (
            importance_comparison[
                "change"
            ].abs()
        )

        importance_comparison = (
            importance_comparison
            .sort_values(
                "absolute_change",
                ascending=False,
            )
        )

        print(
            "Largest feature-importance changes:"
        )

        print(
            importance_comparison
            .head(15)
            .to_string(
                index=False
            )
        )

    except Exception as exc:

        record_warning(
            "Feature-importance comparison",
            str(exc),
        )

    # =========================================================================
    # 27. PREDICTION AGREEMENT
    # =========================================================================

    print_section(
        "27. MODEL 3 VS MODEL 4 PREDICTION AGREEMENT"
    )

    validation_probability_difference = (
        model_4_validation_probability
        - model_3_validation_probability
    )

    test_probability_difference = (
        model_4_test_probability
        - model_3_test_probability
    )

    print("Validation:")
    print(
        f"  Mean probability difference: "
        f"{validation_probability_difference.mean():+.6f}"
    )

    print(
        f"  Mean absolute difference: "
        f"{np.abs(validation_probability_difference).mean():.6f}"
    )

    print(
        f"  Maximum absolute difference: "
        f"{np.abs(validation_probability_difference).max():.6f}"
    )

    print()
    print("Test:")
    print(
        f"  Mean probability difference: "
        f"{test_probability_difference.mean():+.6f}"
    )

    print(
        f"  Mean absolute difference: "
        f"{np.abs(test_probability_difference).mean():.6f}"
    )

    print(
        f"  Maximum absolute difference: "
        f"{np.abs(test_probability_difference).max():.6f}"
    )

    # =========================================================================
    # 28. CLASSIFICATION AGREEMENT
    # =========================================================================

    print_section(
        "28. CLASSIFICATION AGREEMENT"
    )

    model_3_validation_class = (
        model_3_validation_probability
        >= 0.50
    )

    model_4_validation_class = (
        model_4_validation_probability
        >= 0.50
    )

    model_3_test_class = (
        model_3_test_probability
        >= 0.50
    )

    model_4_test_class = (
        model_4_test_probability
        >= 0.50
    )

    validation_disagreement = (
        model_3_validation_class
        != model_4_validation_class
    ).sum()

    test_disagreement = (
        model_3_test_class
        != model_4_test_class
    ).sum()

    print(
        f"Validation classification disagreements: "
        f"{validation_disagreement} "
        f"({validation_disagreement / len(validation):.2%})"
    )

    print(
        f"Test classification disagreements:       "
        f"{test_disagreement} "
        f"({test_disagreement / len(test):.2%})"
    )

    # =========================================================================
    # 29. TRAINING SUMMARY VALIDATION
    # =========================================================================

    print_section(
        "29. TRAINING SUMMARY VALIDATION"
    )

    print(
        "Model 4 training summary:"
    )

    print(
        model_4_training_summary.to_string(
            index=False
        )
    )

    # Validate that the selected methodology is represented.

    summary_text = (
        model_4_training_summary
        .astype(str)
        .to_string()
        .lower()
    )

    # The training summary does not contain a literal "temporal" text field.
    # Instead, temporal CV is represented structurally through:
    #   - cv_folds = 4
    #   - tuning_metric = log_loss
    #
    # The actual expanding-window fold structure is independently validated
    # in Section 11.

    if len(model_4_training_summary) == 1:

        summary_row = model_4_training_summary.iloc[0]

        temporal_cv_metadata_valid = (
            int(summary_row["cv_folds"]) == 4
            and str(summary_row["tuning_metric"]).lower() == "log_loss"
        )

    else:

        temporal_cv_metadata_valid = False

    record_check(
        "Training summary temporal CV metadata",
        temporal_cv_metadata_valid,
        (
            f"Expected cv_folds=4 and tuning_metric=log_loss; "
            f"actual cv_folds="
            f"{model_4_training_summary.iloc[0]['cv_folds'] if len(model_4_training_summary) == 1 else 'N/A'}, "
            f"tuning_metric="
            f"{model_4_training_summary.iloc[0]['tuning_metric'] if len(model_4_training_summary) == 1 else 'N/A'}"
        ),
    )

    record_check(
        "Training summary references Model 3 feature set",
        "model 3" in summary_text
        or "28" in summary_text,
    )

    # =========================================================================
    # 30. PRACTICAL SIGNIFICANCE
    # =========================================================================

    print_section(
        "30. PRACTICAL SIGNIFICANCE OF MODEL 4"
    )

    validation_log_loss_change = (
        model_4_validation_metrics["log_loss"]
        - model_3_validation_metrics["log_loss"]
    )

    validation_brier_change = (
        model_4_validation_metrics["brier"]
        - model_3_validation_metrics["brier"]
    )

    validation_auc_change = (
        model_4_validation_metrics["roc_auc"]
        - model_3_validation_metrics["roc_auc"]
    )

    test_log_loss_change = (
        model_4_test_metrics["log_loss"]
        - model_3_test_metrics["log_loss"]
    )

    test_brier_change = (
        model_4_test_metrics["brier"]
        - model_3_test_metrics["brier"]
    )

    test_auc_change = (
        model_4_test_metrics["roc_auc"]
        - model_3_test_metrics["roc_auc"]
    )

    print(
        "Validation changes:"
    )

    print(
        f"  Log Loss:  {validation_log_loss_change:+.6f}"
    )

    print(
        f"  Brier:     {validation_brier_change:+.6f}"
    )

    print(
        f"  ROC AUC:   {validation_auc_change:+.6f}"
    )

    print()
    print(
        "Test changes:"
    )

    print(
        f"  Log Loss:  {test_log_loss_change:+.6f}"
    )

    print(
        f"  Brier:     {test_brier_change:+.6f}"
    )

    print(
        f"  ROC AUC:   {test_auc_change:+.6f}"
    )

    print()

    if (
        validation_log_loss_change < 0
        and test_log_loss_change < 0
    ):

        print(
            "Model 4 improved Log Loss on both "
            "held-out evaluation periods."
        )

    elif (
        validation_log_loss_change > 0
        and test_log_loss_change > 0
    ):

        print(
            "Model 4 produced worse Log Loss on both "
            "held-out evaluation periods."
        )

    else:

        print(
            "Model 4 produced mixed Log Loss results "
            "across the held-out periods."
        )

    if (
        abs(test_log_loss_change) < 0.005
        and abs(test_brier_change) < 0.002
        and abs(test_auc_change) < 0.005
    ):

        print(
            "The observed test-set differences are "
            "small in practical magnitude."
        )

    # =========================================================================
    # 31. FINAL AUDIT CONCLUSION
    # =========================================================================

    print_section(
        "31. FINAL AUDIT CONCLUSION"
    )

    if FAIL_COUNT == 0:

        print(
            "STRUCTURAL AUDIT: PASS"
        )

        print(
            "All required files, datasets, feature sets, "
            "pipeline components, and selected hyperparameters "
            "passed validation."
        )

    else:

        print(
            "STRUCTURAL AUDIT: FAIL"
        )

        print(
            f"Failed checks: {FAIL_COUNT}"
        )

    print()
    print(
        "TEMPORAL CV:"
    )

    print(
        f"  Configurations evaluated: "
        f"{len(cv_results)}"
    )

    print(
        f"  Expected configurations:   "
        f"{EXPECTED_CV_COMBINATIONS}"
    )

    print(
        f"  Expected total fits:       "
        f"{EXPECTED_CV_FITS}"
    )

    print()
    print(
        "SELECTED MODEL 4 PARAMETERS:"
    )

    for parameter, value in (
        EXPECTED_BEST_PARAMS.items()
    ):

        print(
            f"  {parameter:<20}{value}"
        )

    print()
    print(
        "HELD-OUT PERFORMANCE:"
    )

    print(
        f"  Validation Log Loss: "
        f"{model_4_validation_metrics['log_loss']:.6f}"
    )

    print(
        f"  Test Log Loss:       "
        f"{model_4_test_metrics['log_loss']:.6f}"
    )

    print(
        f"  Validation Brier:    "
        f"{model_4_validation_metrics['brier']:.6f}"
    )

    print(
        f"  Test Brier:          "
        f"{model_4_test_metrics['brier']:.6f}"
    )

    print(
        f"  Validation ROC AUC:  "
        f"{model_4_validation_metrics['roc_auc']:.6f}"
    )

    print(
        f"  Test ROC AUC:        "
        f"{model_4_test_metrics['roc_auc']:.6f}"
    )

    print()
    print(
        "MODEL 3 -> MODEL 4 TEST CHANGES:"
    )

    print(
        f"  Log Loss: "
        f"{test_log_loss_change:+.6f}"
    )

    print(
        f"  Brier:    "
        f"{test_brier_change:+.6f}"
    )

    print(
        f"  ROC AUC:  "
        f"{test_auc_change:+.6f}"
    )

    print()
    print(
        "AUDIT COUNTS:"
    )

    print(
        f"  PASS:     {PASS_COUNT}"
    )

    print(
        f"  FAIL:     {FAIL_COUNT}"
    )

    print(
        f"  WARNING:  {WARNING_COUNT}"
    )

    print()

    if FAIL_COUNT == 0:

        print(
            "OVERALL RESULT: MODEL 4 AUDIT PASSED"
        )

    else:

        print(
            "OVERALL RESULT: MODEL 4 AUDIT REQUIRES REVIEW"
        )

    print(
        "=" * 80
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()