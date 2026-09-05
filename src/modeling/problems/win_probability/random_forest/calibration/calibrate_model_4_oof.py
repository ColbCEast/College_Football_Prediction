"""
Random Forest Model 4 — Temporal OOF Probability Calibration

Purpose
-------
Evaluate post-hoc probability calibration for the existing Random Forest
Model 4 using temporally ordered out-of-fold predictions.

The original Model 4 remains completely unchanged.

Model 4 preprocessing
---------------------
- 28-feature specification
- Median imputation
- RandomForestClassifier
- Existing Model 4 hyperparameters recovered from model.joblib

OOF design
----------
2016 prediction: train on 2015
2017 prediction: train on 2015-2016
2018 prediction: train on 2015-2017
...
2022 prediction: train on 2015-2021

Calibration
-----------
1. Generate OOF probabilities for 2016-2022.
2. Fit sigmoid and isotonic calibrators on OOF predictions.
3. Evaluate both on untouched 2023-2024 validation predictions.
4. Select calibration method using validation log loss.
5. Refit the selected calibrator using OOF + validation data.
6. Apply the final calibrator to untouched 2025 test predictions.

The 2025 test set is never used for calibration fitting or method selection.

Outputs
-------
models/win_probability/random_forest/calibration/model_4/oof/
    oof_predictions.csv
    validation_calibration_comparison.csv
    test_calibration_comparison.csv
    validation_calibrated_predictions.csv
    test_calibrated_predictions.csv
    oof_calibration_bins.csv
    validation_calibration_bins.csv
    test_calibration_bins.csv
    calibration_summary.csv
    sigmoid_calibrator.joblib
    isotonic_calibrator.joblib
    final_calibrator.joblib
"""


from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42

MODEL_NUMBER = 4

# Column names
TARGET = "win_home"
GAME_ID = "gameId"
SEASON = "season"

# Prediction-output column names
OUTPUT_TARGET = "win_home_actual"
OUTPUT_PROBABILITY = "win_home_probability"
OUTPUT_PREDICTION = "win_home_prediction"
OUTPUT_SPLIT = "split"

# Calibration settings
N_CALIBRATION_BINS = 10

# Temporal OOF settings
INITIAL_TRAIN_SEASON = 2015
OOF_START_SEASON = 2016
OOF_END_SEASON = 2022

VALIDATION_START_SEASON = 2023
VALIDATION_END_SEASON = 2024

TEST_SEASON = 2025

# =============================================================================
# PATHS
# =============================================================================

MODEL_DIR = Path(
    "models/win_probability/random_forest/model_4"
)

MODEL_PATH = MODEL_DIR / "model.joblib"
FEATURE_LIST_PATH = MODEL_DIR / "feature_list.csv"

TRAIN_PATH = Path(
    "data/processed/model_inputs/win_probability/train.csv"
)

VALIDATION_PATH = Path(
    "data/processed/model_inputs/win_probability/validation.csv"
)

TEST_PATH = Path(
    "data/processed/model_inputs/win_probability/test.csv"
)

OUTPUT_DIR = Path(
    "models/win_probability/random_forest/"
    "calibration/model_4/oof"
)


# =============================================================================
# FEATURE SPECIFICATION
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
# DATA LOADING
# =============================================================================

def load_existing_model():
    """
    Load the existing Model 4 pipeline.

    Expected structure:

        Pipeline
        ├── imputer
        └── model

    The fitted pipeline is never modified or refit.
    """

    print("Loading existing Model 4 artifact...")

    model = joblib.load(MODEL_PATH)

    if not isinstance(model, Pipeline):
        raise TypeError(
            "Expected Model 4 artifact to be a sklearn Pipeline. "
            f"Found: {type(model)}"
        )

    if "imputer" not in model.named_steps:
        raise ValueError(
            "Model 4 pipeline does not contain an 'imputer' step."
        )

    if "model" not in model.named_steps:
        raise ValueError(
            "Model 4 pipeline does not contain a 'model' step."
        )

    if not isinstance(
        model.named_steps["imputer"],
        SimpleImputer,
    ):
        raise TypeError(
            "Model 4 'imputer' step is not a SimpleImputer."
        )

    if not isinstance(
        model.named_steps["model"],
        RandomForestClassifier,
    ):
        raise TypeError(
            "Model 4 'model' step is not a RandomForestClassifier."
        )

    print("  Pipeline loaded successfully.")

    return model


def load_feature_list():
    """
    Load Model 4's saved feature list and verify that it matches the
    feature specification in train.py.
    """

    print("\nLoading Model 4 feature list...")

    feature_list = pd.read_csv(
        FEATURE_LIST_PATH
    )

    if "feature" not in feature_list.columns:
        raise ValueError(
            f"Expected 'feature' column in {FEATURE_LIST_PATH}. "
            f"Found: {list(feature_list.columns)}"
        )

    saved_features = (
        feature_list["feature"]
        .dropna()
        .astype(str)
        .tolist()
    )

    if saved_features != FEATURES:
        raise ValueError(
            "Saved Model 4 feature list does not exactly match "
            "the Model 4 feature specification.\n\n"
            f"Saved features:\n{saved_features}\n\n"
            f"Expected features:\n{FEATURES}"
        )

    print(
        f"  Features loaded: {len(saved_features)}"
    )

    return saved_features


def load_data():
    """Load the complete temporal modeling datasets."""

    print("\nLoading modeling datasets...")

    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    test = pd.read_csv(TEST_PATH)

    print(
        f"  Training:   {train.shape}"
    )

    print(
        f"  Validation: {validation.shape}"
    )

    print(
        f"  Test:       {test.shape}"
    )

    return train, validation, test


# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_dataset(
    df,
    name,
    expected_seasons=None,
):
    """Validate one modeling dataset."""

    required_columns = set(
        FEATURES
        + [
            TARGET,
            GAME_ID,
            SEASON,
        ]
    )

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{name.capitalize()} dataset is missing required columns: "
            f"{sorted(missing)}"
        )

    if df[GAME_ID].duplicated().any():
        raise ValueError(
            f"{name.capitalize()} dataset contains duplicate game IDs."
        )

    if df[TARGET].isna().any():
        raise ValueError(
            f"{name.capitalize()} target contains missing values."
        )

    if not df[TARGET].isin([0, 1]).all():
        raise ValueError(
            f"{name.capitalize()} target contains values other than 0/1."
        )

    seasons = sorted(
        df[SEASON].unique()
    )

    print(
        f"  {name.capitalize()} seasons: {seasons}"
    )

    if expected_seasons is not None:

        if seasons != list(expected_seasons):
            raise ValueError(
                f"Unexpected {name} seasons. "
                f"Expected {list(expected_seasons)}, "
                f"found {seasons}."
            )


def validate_all_data(
    train,
    validation,
    test,
):
    """Validate the complete temporal dataset structure."""

    print("\nValidating datasets...")

    validate_dataset(
        train,
        "training",
        expected_seasons=range(
            INITIAL_TRAIN_SEASON,
            OOF_END_SEASON + 1,
        ),
    )

    validate_dataset(
        validation,
        "validation",
        expected_seasons=range(
            VALIDATION_START_SEASON,
            VALIDATION_END_SEASON + 1,
        ),
    )

    validate_dataset(
        test,
        "test",
        expected_seasons=[TEST_SEASON],
    )

    # -------------------------------------------------------------------------
    # Check that all splits are mutually exclusive.
    # -------------------------------------------------------------------------

    train_ids = set(
        train[GAME_ID]
    )

    validation_ids = set(
        validation[GAME_ID]
    )

    test_ids = set(
        test[GAME_ID]
    )

    if train_ids & validation_ids:
        raise ValueError(
            "Training and validation contain overlapping game IDs."
        )

    if train_ids & test_ids:
        raise ValueError(
            "Training and test contain overlapping game IDs."
        )

    if validation_ids & test_ids:
        raise ValueError(
            "Validation and test contain overlapping game IDs."
        )

    print(
        "  Dataset validation PASSED."
    )


# =============================================================================
# MODEL CONSTRUCTION
# =============================================================================

def create_oof_pipeline(
    existing_model,
):
    """
    Create a fresh Model 4-equivalent pipeline.

    The hyperparameters are copied from the existing fitted Model 4.

    We intentionally create a new pipeline for every OOF fold.
    """

    existing_imputer = (
        existing_model.named_steps["imputer"]
    )

    existing_rf = (
        existing_model.named_steps["model"]
    )

    # -------------------------------------------------------------------------
    # Copy the preprocessing configuration.
    #
    # Model 4 uses:
    #     SimpleImputer(strategy="median")
    # -------------------------------------------------------------------------

    imputer = SimpleImputer(
        strategy=existing_imputer.strategy,
        missing_values=existing_imputer.missing_values,
        add_indicator=existing_imputer.add_indicator,
        keep_empty_features=existing_imputer.keep_empty_features,
    )

    # -------------------------------------------------------------------------
    # Copy the exact Random Forest configuration.
    # -------------------------------------------------------------------------

    rf = RandomForestClassifier(
        **existing_rf.get_params()
    )

    return Pipeline(
        steps=[
            (
                "imputer",
                imputer,
            ),
            (
                "model",
                rf,
            ),
        ]
    )


# =============================================================================
# OOF PREDICTIONS
# =============================================================================

def create_oof_predictions(
    train,
    existing_model,
):
    """
    Generate expanding-window temporal OOF predictions.

    Each season is predicted using only seasons that occurred before it.
    """

    print("\n" + "=" * 80)
    print("GENERATING TEMPORAL OOF PREDICTIONS")
    print("=" * 80)

    print(
        "\nTemporal structure:"
    )

    print(
        f"  Initial training season: {INITIAL_TRAIN_SEASON}"
    )

    print(
        f"  OOF seasons:             "
        f"{OOF_START_SEASON}-{OOF_END_SEASON}"
    )

    print(
        "  Method:                  expanding temporal window"
    )

    oof_rows = []

    for target_season in range(
        OOF_START_SEASON,
        OOF_END_SEASON + 1,
    ):

        training_seasons = list(
            range(
                INITIAL_TRAIN_SEASON,
                target_season,
            )
        )

        print("\n" + "-" * 80)

        print(
            f"OOF FOLD — PREDICT {target_season}"
        )

        print(
            f"  Training seasons: "
            f"{training_seasons[0]}-{training_seasons[-1]}"
        )

        # ---------------------------------------------------------------------
        # Select training data from prior seasons only.
        # ---------------------------------------------------------------------

        train_fold = train[
            train[SEASON].isin(
                training_seasons
            )
        ].copy()

        target_fold = train[
            train[SEASON] == target_season
        ].copy()

        if train_fold.empty:
            raise ValueError(
                f"No training rows available for OOF fold "
                f"{target_season}."
            )

        if target_fold.empty:
            raise ValueError(
                f"No prediction rows available for OOF fold "
                f"{target_season}."
            )

        # ---------------------------------------------------------------------
        # Build fresh Model 4-equivalent pipeline.
        # ---------------------------------------------------------------------

        fold_model = create_oof_pipeline(
            existing_model
        )

        X_train = train_fold[
            FEATURES
        ]

        y_train = train_fold[
            TARGET
        ]

        X_target = target_fold[
            FEATURES
        ]

        print(
            f"  Training rows:   {len(train_fold):,}"
        )

        print(
            f"  Prediction rows: {len(target_fold):,}"
        )

        print(
            f"  Training home win rate: "
            f"{y_train.mean():.6f}"
        )

        # ---------------------------------------------------------------------
        # Fit only on historical seasons.
        # ---------------------------------------------------------------------

        fold_model.fit(
            X_train,
            y_train,
        )

        probabilities = (
            fold_model.predict_proba(
                X_target
            )[:, 1]
        )

        predictions = (
            probabilities >= 0.50
        ).astype(int)

        # ---------------------------------------------------------------------
        # Store OOF predictions.
        # ---------------------------------------------------------------------

        fold_output = pd.DataFrame(
            {
                GAME_ID: target_fold[
                    GAME_ID
                ].values,

                SEASON: target_fold[
                    SEASON
                ].values,

                OUTPUT_TARGET: target_fold[
                    TARGET
                ].values,

                OUTPUT_PROBABILITY: probabilities,

                OUTPUT_PREDICTION: predictions,

                OUTPUT_SPLIT: "oof",

                "oof_training_start_season":
                    training_seasons[0],

                "oof_training_end_season":
                    training_seasons[-1],
            }
        )

        oof_rows.append(
            fold_output
        )

        # ---------------------------------------------------------------------
        # Fold diagnostics.
        # ---------------------------------------------------------------------

        fold_metrics = calculate_metrics(
            target_fold[TARGET].to_numpy(),
            probabilities,
        )

        print(
            f"  Log Loss: "
            f"{fold_metrics['log_loss']:.6f}"
        )

        print(
            f"  Brier:    "
            f"{fold_metrics['brier_score']:.6f}"
        )

        print(
            f"  ROC AUC:  "
            f"{fold_metrics['roc_auc']:.6f}"
        )

    # -------------------------------------------------------------------------
    # Combine folds.
    # -------------------------------------------------------------------------

    oof = pd.concat(
        oof_rows,
        ignore_index=True,
    )

    # -------------------------------------------------------------------------
    # Validate OOF predictions.
    # -------------------------------------------------------------------------

    if oof[GAME_ID].duplicated().any():
        raise ValueError(
            "OOF predictions contain duplicate game IDs."
        )

    expected_seasons = list(
        range(
            OOF_START_SEASON,
            OOF_END_SEASON + 1,
        )
    )

    actual_seasons = sorted(
        oof[SEASON].unique()
    )

    if actual_seasons != expected_seasons:
        raise ValueError(
            "OOF predictions contain unexpected seasons. "
            f"Expected {expected_seasons}, "
            f"found {actual_seasons}."
        )

    if oof[OUTPUT_PROBABILITY].isna().any():
        raise ValueError(
            "OOF predictions contain missing probabilities."
        )

    if not (
        (oof[OUTPUT_PROBABILITY] >= 0.0)
        & (oof[OUTPUT_PROBABILITY] <= 1.0)
    ).all():
        raise ValueError(
            "OOF probabilities contain values outside [0, 1]."
        )

    print("\n" + "=" * 80)
    print("OOF GENERATION COMPLETE")
    print("=" * 80)

    print(
        f"\nOOF rows: "
        f"{len(oof):,}"
    )

    print(
        f"OOF seasons: "
        f"{actual_seasons}"
    )

    print(
        f"OOF home win rate: "
        f"{oof[OUTPUT_TARGET].mean():.6f}"
    )

    return oof


# =============================================================================
# SIGMOID CALIBRATION
# =============================================================================

def fit_sigmoid_calibrator(
    probabilities,
    targets,
):
    """
    Fit sigmoid / Platt calibration.

    This is the same logit-transformation approach used in the
    original Model 4 calibration experiment.
    """

    probabilities = np.clip(
        probabilities,
        1e-6,
        1 - 1e-6,
    )

    logits = np.log(
        probabilities
        / (1.0 - probabilities)
    ).reshape(-1, 1)

    calibrator = LogisticRegression(
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )

    calibrator.fit(
        logits,
        targets,
    )

    return calibrator


def apply_sigmoid_calibrator(
    calibrator,
    probabilities,
):
    """Apply sigmoid calibration."""

    probabilities = np.clip(
        probabilities,
        1e-6,
        1 - 1e-6,
    )

    logits = np.log(
        probabilities
        / (1.0 - probabilities)
    ).reshape(-1, 1)

    calibrated = calibrator.predict_proba(
        logits
    )[:, 1]

    return np.clip(
        calibrated,
        1e-6,
        1 - 1e-6,
    )


# =============================================================================
# ISOTONIC CALIBRATION
# =============================================================================

def fit_isotonic_calibrator(
    probabilities,
    targets,
):
    """Fit isotonic regression calibration."""

    calibrator = IsotonicRegression(
        y_min=1e-6,
        y_max=1 - 1e-6,
        out_of_bounds="clip",
    )

    calibrator.fit(
        probabilities,
        targets,
    )

    return calibrator


def apply_isotonic_calibrator(
    calibrator,
    probabilities,
):
    """Apply isotonic calibration."""

    calibrated = calibrator.predict(
        probabilities
    )

    return np.clip(
        calibrated,
        1e-6,
        1 - 1e-6,
    )


# =============================================================================
# METRICS
# =============================================================================

def calculate_calibration_bins(
    y_true,
    probabilities,
    n_bins=N_CALIBRATION_BINS,
):
    """Calculate equal-width calibration bins."""

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

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

        count = int(
            mask.sum()
        )

        if count == 0:

            rows.append(
                {
                    "bin": i + 1,
                    "bin_lower": lower,
                    "bin_upper": upper,
                    "count": 0,
                    "mean_predicted_probability": np.nan,
                    "observed_win_rate": np.nan,
                    "absolute_error": np.nan,
                    "squared_error": np.nan,
                }
            )

            continue

        mean_probability = (
            probabilities[mask].mean()
        )

        observed_rate = (
            y_true[mask].mean()
        )

        error = (
            mean_probability
            - observed_rate
        )

        rows.append(
            {
                "bin": i + 1,
                "bin_lower": lower,
                "bin_upper": upper,
                "count": count,
                "mean_predicted_probability":
                    mean_probability,
                "observed_win_rate":
                    observed_rate,
                "absolute_error":
                    abs(error),
                "squared_error":
                    error ** 2,
            }
        )

    return pd.DataFrame(rows)


def calculate_ece(
    y_true,
    probabilities,
    n_bins=N_CALIBRATION_BINS,
):
    """Calculate Expected Calibration Error."""

    bins = calculate_calibration_bins(
        y_true,
        probabilities,
        n_bins,
    )

    total = bins["count"].sum()

    if total == 0:
        return np.nan

    valid = (
        bins["count"] > 0
    )

    return (
        (
            bins.loc[
                valid,
                "count",
            ]
            * bins.loc[
                valid,
                "absolute_error",
            ]
        ).sum()
        / total
    )


def calculate_mce(
    y_true,
    probabilities,
    n_bins=N_CALIBRATION_BINS,
):
    """Calculate Maximum Calibration Error."""

    bins = calculate_calibration_bins(
        y_true,
        probabilities,
        n_bins,
    )

    valid = (
        bins["count"] > 0
    )

    if not valid.any():
        return np.nan

    return bins.loc[
        valid,
        "absolute_error",
    ].max()


def calculate_metrics(
    y_true,
    probabilities,
):
    """Calculate probability and classification metrics."""

    probabilities = np.asarray(
        probabilities
    )

    predictions = (
        probabilities >= 0.50
    ).astype(int)

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

        "ece": calculate_ece(
            y_true,
            probabilities,
        ),

        "mce": calculate_mce(
            y_true,
            probabilities,
        ),
    }


# =============================================================================
# OUTPUT HELPERS
# =============================================================================

def create_calibration_predictions(
    df,
    raw_probabilities,
    sigmoid_probabilities,
    isotonic_probabilities,
    final_probabilities=None,
):
    """Create standardized calibration prediction output."""

    output = df[
        [
            GAME_ID,
            SEASON,
            TARGET,
        ]
    ].copy()

    output.rename(
        columns={
            TARGET: OUTPUT_TARGET,
        },
        inplace=True,
    )

    output["raw_probability"] = (
        raw_probabilities
    )

    output["sigmoid_probability"] = (
        sigmoid_probabilities
    )

    output["isotonic_probability"] = (
        isotonic_probabilities
    )

    output["raw_prediction"] = (
        raw_probabilities >= 0.50
    ).astype(int)

    output["sigmoid_prediction"] = (
        sigmoid_probabilities >= 0.50
    ).astype(int)

    output["isotonic_prediction"] = (
        isotonic_probabilities >= 0.50
    ).astype(int)

    if final_probabilities is not None:

        output[
            "final_calibrated_probability"
        ] = final_probabilities

        output[
            "final_calibrated_prediction"
        ] = (
            final_probabilities >= 0.50
        ).astype(int)

    return output


def create_bin_output(
    dataset_name,
    y_true,
    probability_sets,
):
    """Create calibration-bin diagnostics."""

    outputs = []

    for method, probabilities in probability_sets:

        bins = calculate_calibration_bins(
            y_true,
            probabilities,
        )

        bins.insert(
            0,
            "method",
            method,
        )

        bins.insert(
            0,
            "dataset",
            dataset_name,
        )

        outputs.append(
            bins
        )

    return pd.concat(
        outputs,
        ignore_index=True,
    )


def add_changes_relative_to_raw(
    comparison,
):
    """Add changes relative to raw Model 4 probabilities."""

    comparison = comparison.copy()

    change_metrics = [
        "log_loss",
        "brier_score",
        "ece",
        "mce",
        "roc_auc",
    ]

    for dataset in [
        "validation",
        "test",
    ]:

        raw_mask = (
            (comparison["dataset"] == dataset)
            & (
                comparison["method"]
                == "raw_model_4"
            )
        )

        if not raw_mask.any():
            continue

        raw_row = comparison.loc[
            raw_mask
        ].iloc[0]

        for metric in change_metrics:

            comparison.loc[
                comparison["dataset"] == dataset,
                f"{metric}_change",
            ] = (
                comparison.loc[
                    comparison["dataset"] == dataset,
                    metric,
                ]
                - raw_row[metric]
            )

    return comparison


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print(
        "RANDOM FOREST MODEL 4 — "
        "TEMPORAL OOF PROBABILITY CALIBRATION"
    )
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================================
    # Load existing Model 4
    # =========================================================================

    existing_model = (
        load_existing_model()
    )

    features = (
        load_feature_list()
    )

    train, validation, test = (
        load_data()
    )

    validate_all_data(
        train,
        validation,
        test,
    )

    # =========================================================================
    # Verify existing Model 4 configuration
    # =========================================================================

    print("\n" + "=" * 80)
    print("MODEL 4 CONFIGURATION")
    print("=" * 80)

    imputer = (
        existing_model.named_steps[
            "imputer"
        ]
    )

    rf = (
        existing_model.named_steps[
            "model"
        ]
    )

    print(
        f"\nImputer:"
    )

    print(
        f"  strategy: "
        f"{imputer.strategy}"
    )

    print(
        "\nRandom Forest parameters:"
    )

    important_params = [
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "bootstrap",
        "class_weight",
        "criterion",
        "random_state",
        "n_jobs",
    ]

    rf_params = rf.get_params()

    for parameter in important_params:

        print(
            f"  {parameter:<22}: "
            f"{rf_params[parameter]}"
        )

    print(
        "\nThese exact hyperparameters will be reused "
        "for every OOF fold."
    )

    print(
        "\nThe original Model 4 pipeline will NOT be modified."
    )

    # =========================================================================
    # Generate OOF predictions
    # =========================================================================

    oof = create_oof_predictions(
        train,
        existing_model,
    )

    oof.to_csv(
        OUTPUT_DIR
        / "oof_predictions.csv",
        index=False,
    )

    # =========================================================================
    # OOF evaluation
    # =========================================================================

    y_oof = oof[
        OUTPUT_TARGET
    ].to_numpy()

    p_oof = oof[
        OUTPUT_PROBABILITY
    ].to_numpy()

    print("\n" + "=" * 80)
    print("OOF RAW MODEL 4 PERFORMANCE")
    print("=" * 80)

    oof_metrics = calculate_metrics(
        y_oof,
        p_oof,
    )

    for metric, value in oof_metrics.items():

        print(
            f"  {metric:<22}: "
            f"{value:.6f}"
        )

    # =========================================================================
    # Fit OOF calibrators
    # =========================================================================

    print("\n" + "=" * 80)
    print("FITTING CALIBRATORS ON OOF DATA")
    print("=" * 80)

    print(
        f"\nCalibration rows: "
        f"{len(oof):,}"
    )

    print(
        "Calibration seasons: "
        f"{OOF_START_SEASON}-{OOF_END_SEASON}"
    )

    print(
        "\n2023-2024 validation and 2025 test "
        "are NOT used here."
    )

    sigmoid_oof = (
        fit_sigmoid_calibrator(
            p_oof,
            y_oof,
        )
    )

    isotonic_oof = (
        fit_isotonic_calibrator(
            p_oof,
            y_oof,
        )
    )

    print(
        "\nOOF sigmoid parameters:"
    )

    print(
        f"  Intercept: "
        f"{sigmoid_oof.intercept_[0]:.6f}"
    )

    print(
        f"  Coefficient: "
        f"{sigmoid_oof.coef_[0, 0]:.6f}"
    )

    # =========================================================================
    # Apply OOF calibrators to validation
    # =========================================================================

    y_validation = validation[
        TARGET
    ].to_numpy()

    p_validation = (
        # Use the existing Model 4 predictions exactly as they were
        # originally generated.
        pd.read_csv(
            MODEL_DIR
            / "validation_predictions.csv"
        )[OUTPUT_PROBABILITY]
        .to_numpy()
    )

    # -------------------------------------------------------------------------
    # Confirm prediction alignment.
    # -------------------------------------------------------------------------

    existing_validation_predictions = pd.read_csv(
        MODEL_DIR
        / "validation_predictions.csv"
    )

    if not np.array_equal(
        validation[GAME_ID].to_numpy(),
        existing_validation_predictions[
            GAME_ID
        ].to_numpy(),
    ):
        raise ValueError(
            "Validation game ordering differs between modeling validation.csv "
            "and Model 4 validation_predictions.csv."
        )

    validation_sigmoid = (
        apply_sigmoid_calibrator(
            sigmoid_oof,
            p_validation,
        )
    )

    validation_isotonic = (
        apply_isotonic_calibrator(
            isotonic_oof,
            p_validation,
        )
    )

    # =========================================================================
    # Validation evaluation
    # =========================================================================

    print("\n" + "=" * 80)
    print("2023-2024 VALIDATION — OOF CALIBRATION")
    print("=" * 80)

    validation_raw_metrics = (
        calculate_metrics(
            y_validation,
            p_validation,
        )
    )

    validation_sigmoid_metrics = (
        calculate_metrics(
            y_validation,
            validation_sigmoid,
        )
    )

    validation_isotonic_metrics = (
        calculate_metrics(
            y_validation,
            validation_isotonic,
        )
    )

    validation_comparison = pd.DataFrame(
        [
            {
                "dataset": "validation",
                "method": "raw_model_4",
                **validation_raw_metrics,
            },
            {
                "dataset": "validation",
                "method": "sigmoid",
                **validation_sigmoid_metrics,
            },
            {
                "dataset": "validation",
                "method": "isotonic",
                **validation_isotonic_metrics,
            },
        ]
    )

    validation_comparison = (
        add_changes_relative_to_raw(
            validation_comparison
        )
    )

    print(
        validation_comparison[
            [
                "dataset",
                "method",
                "log_loss",
                "brier_score",
                "roc_auc",
                "accuracy",
                "balanced_accuracy",
                "ece",
                "mce",
            ]
        ].to_string(index=False)
    )

    # =========================================================================
    # Select calibration method
    # =========================================================================

    candidates = (
        validation_comparison[
            validation_comparison["method"]
            != "raw_model_4"
        ]
        .sort_values(
            [
                "log_loss",
                "brier_score",
            ]
        )
        .reset_index(drop=True)
    )

    validation_winner = (
        candidates.loc[
            0,
            "method"
        ]
    )

    print("\n" + "=" * 80)
    print("CALIBRATION METHOD SELECTION")
    print("=" * 80)

    print(
        "\nPrimary criterion: validation Log Loss"
    )

    print(
        "Secondary criterion: validation Brier Score"
    )

    print(
        f"\nSelected method: "
        f"{validation_winner}"
    )

    # =========================================================================
    # Save initial OOF-trained calibrators
    # =========================================================================

    joblib.dump(
        sigmoid_oof,
        OUTPUT_DIR
        / "sigmoid_calibrator.joblib",
    )

    joblib.dump(
        isotonic_oof,
        OUTPUT_DIR
        / "isotonic_calibrator.joblib",
    )

    # =========================================================================
    # Final calibrator
    # =========================================================================

    print("\n" + "=" * 80)
    print("FITTING FINAL SELECTED CALIBRATOR")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # The final calibrator gets:
    #
    #   OOF 2016-2022
    #   +
    #   Validation 2023-2024
    #
    # It does NOT see 2025.
    # -------------------------------------------------------------------------

    final_probabilities = np.concatenate(
        [
            p_oof,
            p_validation,
        ]
    )

    final_targets = np.concatenate(
        [
            y_oof,
            y_validation,
        ]
    )

    print(
        f"\nOOF rows: "
        f"{len(p_oof):,}"
    )

    print(
        f"Validation rows: "
        f"{len(p_validation):,}"
    )

    print(
        f"Final calibration rows: "
        f"{len(final_probabilities):,}"
    )

    print(
        "\n2025 test data is NOT used."
    )

    if validation_winner == "sigmoid":

        final_calibrator = (
            fit_sigmoid_calibrator(
                final_probabilities,
                final_targets,
            )
        )

        final_calibration_type = "sigmoid"

        print(
            "\nFinal calibrator: sigmoid"
        )

        print(
            f"  Intercept: "
            f"{final_calibrator.intercept_[0]:.6f}"
        )

        print(
            f"  Coefficient: "
            f"{final_calibrator.coef_[0, 0]:.6f}"
        )

    else:

        final_calibrator = (
            fit_isotonic_calibrator(
                final_probabilities,
                final_targets,
            )
        )

        final_calibration_type = "isotonic"

        print(
            "\nFinal calibrator: isotonic"
        )

    joblib.dump(
        final_calibrator,
        OUTPUT_DIR
        / "final_calibrator.joblib",
    )

    # =========================================================================
    # Load existing Model 4 test predictions
    # =========================================================================

    existing_test_predictions = pd.read_csv(
        MODEL_DIR
        / "test_predictions.csv"
    )

    if not np.array_equal(
        test[GAME_ID].to_numpy(),
        existing_test_predictions[
            GAME_ID
        ].to_numpy(),
    ):
        raise ValueError(
            "Test game ordering differs between modeling test.csv "
            "and Model 4 test_predictions.csv."
        )

    y_test = test[
        TARGET
    ].to_numpy()

    p_test = (
        existing_test_predictions[
            OUTPUT_PROBABILITY
        ]
        .to_numpy()
    )

    # =========================================================================
    # Apply OOF-trained calibrators to test
    # =========================================================================

    test_sigmoid = (
        apply_sigmoid_calibrator(
            sigmoid_oof,
            p_test,
        )
    )

    test_isotonic = (
        apply_isotonic_calibrator(
            isotonic_oof,
            p_test,
        )
    )

    # =========================================================================
    # Apply final selected calibrator to test
    # =========================================================================

    if final_calibration_type == "sigmoid":

        final_test_probability = (
            apply_sigmoid_calibrator(
                final_calibrator,
                p_test,
            )
        )

    else:

        final_test_probability = (
            apply_isotonic_calibrator(
                final_calibrator,
                p_test,
            )
        )

    # =========================================================================
    # Final test evaluation
    # =========================================================================

    print("\n" + "=" * 80)
    print("2025 TEST — FINAL OUT-OF-SAMPLE EVALUATION")
    print("=" * 80)

    print(
        "\n2025 was not used for:"
        "\n  - OOF model fitting"
        "\n  - calibrator fitting"
        "\n  - calibration method selection"
    )

    test_raw_metrics = calculate_metrics(
        y_test,
        p_test,
    )

    test_sigmoid_metrics = calculate_metrics(
        y_test,
        test_sigmoid,
    )

    test_isotonic_metrics = calculate_metrics(
        y_test,
        test_isotonic,
    )

    test_final_metrics = calculate_metrics(
        y_test,
        final_test_probability,
    )

    test_comparison = pd.DataFrame(
        [
            {
                "dataset": "test",
                "method": "raw_model_4",
                **test_raw_metrics,
            },
            {
                "dataset": "test",
                "method": "sigmoid",
                **test_sigmoid_metrics,
            },
            {
                "dataset": "test",
                "method": "isotonic",
                **test_isotonic_metrics,
            },
            {
                "dataset": "test",
                "method": "final_selected",
                **test_final_metrics,
            },
        ]
    )

    test_comparison = (
        add_changes_relative_to_raw(
            test_comparison
        )
    )

    print(
        test_comparison[
            [
                "dataset",
                "method",
                "log_loss",
                "brier_score",
                "roc_auc",
                "accuracy",
                "balanced_accuracy",
                "ece",
                "mce",
            ]
        ].to_string(index=False)
    )

    # =========================================================================
    # Save comparison files
    # =========================================================================

    validation_comparison.to_csv(
        OUTPUT_DIR
        / "validation_calibration_comparison.csv",
        index=False,
    )

    test_comparison.to_csv(
        OUTPUT_DIR
        / "test_calibration_comparison.csv",
        index=False,
    )

    # =========================================================================
    # Save calibrated predictions
    # =========================================================================

    validation_output = (
        create_calibration_predictions(
            validation,
            p_validation,
            validation_sigmoid,
            validation_isotonic,
        )
    )

    test_output = (
        create_calibration_predictions(
            test,
            p_test,
            test_sigmoid,
            test_isotonic,
            final_test_probability,
        )
    )

    validation_output.to_csv(
        OUTPUT_DIR
        / "validation_calibrated_predictions.csv",
        index=False,
    )

    test_output.to_csv(
        OUTPUT_DIR
        / "test_calibrated_predictions.csv",
        index=False,
    )

    # =========================================================================
    # Calibration bins
    # =========================================================================

    print("\nSaving calibration-bin diagnostics...")

    oof_sigmoid_probabilities = (
        apply_sigmoid_calibrator(
            sigmoid_oof,
            p_oof,
        )
    )

    oof_isotonic_probabilities = (
        apply_isotonic_calibrator(
            isotonic_oof,
            p_oof,
        )
    )

    oof_bins = create_bin_output(
        "oof",
        y_oof,
        [
            (
                "raw_model_4",
                p_oof,
            ),
            (
                "sigmoid",
                oof_sigmoid_probabilities,
            ),
            (
                "isotonic",
                oof_isotonic_probabilities,
            ),
        ],
    )

    validation_bins = create_bin_output(
        "validation",
        y_validation,
        [
            (
                "raw_model_4",
                p_validation,
            ),
            (
                "sigmoid",
                validation_sigmoid,
            ),
            (
                "isotonic",
                validation_isotonic,
            ),
        ],
    )

    test_bins = create_bin_output(
        "test",
        y_test,
        [
            (
                "raw_model_4",
                p_test,
            ),
            (
                "sigmoid",
                test_sigmoid,
            ),
            (
                "isotonic",
                test_isotonic,
            ),
            (
                "final_selected",
                final_test_probability,
            ),
        ],
    )

    oof_bins.to_csv(
        OUTPUT_DIR
        / "oof_calibration_bins.csv",
        index=False,
    )

    validation_bins.to_csv(
        OUTPUT_DIR
        / "validation_calibration_bins.csv",
        index=False,
    )

    test_bins.to_csv(
        OUTPUT_DIR
        / "test_calibration_bins.csv",
        index=False,
    )

    # =========================================================================
    # Calibration summary
    # =========================================================================

    summary = pd.DataFrame(
        [
            {
                "model":
                    "random_forest_model_4",

                "calibration_type":
                    "temporal_oof",

                "oof_training_period":
                    f"{OOF_START_SEASON}-"
                    f"{OOF_END_SEASON}",

                "validation_period":
                    f"{VALIDATION_START_SEASON}-"
                    f"{VALIDATION_END_SEASON}",

                "test_period":
                    str(TEST_SEASON),

                "oof_rows":
                    len(oof),

                "validation_rows":
                    len(validation),

                "test_rows":
                    len(test),

                "validation_winner_by_log_loss":
                    validation_winner,

                "final_calibrator_training_rows":
                    len(final_probabilities),

                "final_calibration_type":
                    final_calibration_type,

                "test_final_log_loss":
                    test_final_metrics[
                        "log_loss"
                    ],

                "test_final_brier_score":
                    test_final_metrics[
                        "brier_score"
                    ],

                "test_final_roc_auc":
                    test_final_metrics[
                        "roc_auc"
                    ],

                "test_final_accuracy":
                    test_final_metrics[
                        "accuracy"
                    ],

                "test_final_balanced_accuracy":
                    test_final_metrics[
                        "balanced_accuracy"
                    ],

                "test_final_ece":
                    test_final_metrics[
                        "ece"
                    ],

                "test_final_mce":
                    test_final_metrics[
                        "mce"
                    ],

                "random_state":
                    RANDOM_STATE,

                "calibration_bins":
                    N_CALIBRATION_BINS,
            }
        ]
    )

    summary.to_csv(
        OUTPUT_DIR
        / "calibration_summary.csv",
        index=False,
    )

    # =========================================================================
    # Final output
    # =========================================================================

    print("\n" + "=" * 80)
    print("MODEL 4 TEMPORAL OOF CALIBRATION COMPLETE")
    print("=" * 80)

    print(
        "\nOriginal Model 4 was NOT modified."
    )

    print(
        f"\nValidation-selected method: "
        f"{validation_winner}"
    )

    print(
        "\nFinal 2025 performance:"
    )

    for metric, value in test_final_metrics.items():

        print(
            f"  {metric:<22}: "
            f"{value:.6f}"
        )

    print(
        "\nArtifacts saved to:"
    )

    print(
        f"  {OUTPUT_DIR.resolve()}"
    )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()